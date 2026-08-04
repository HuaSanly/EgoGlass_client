from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import queue
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable, Coroutine, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn

from ingest_gateway.app import create_app
from ingest_gateway.discovery import DISCOVERY_PORT, LanDiscoveryService
from ingest_gateway.imu_preview import ImuPreviewRuntime
from ingest_gateway.live_frames import LiveFrame, LiveFrameBuffer
from ingest_gateway.recording import RecordingRuntime
from ingest_gateway.recording_models import RecordingLibrary, RecordingState
from ingest_gateway.webrtc_models import StreamControlAction, StreamControlCommand
from ingest_gateway.webrtc_runtime import WebRtcSessionRuntime
from perception.configuration import (
    ConfigApplyResult,
    ConfigImpact,
    ConfigSnapshot,
    ConfigurationApplyRequest,
    ConfigurationProvenance,
    ConfigurationService,
    ValidationIssue,
)
from perception.runtime import HandTrackingRuntime, HandTrackingRuntimeConfig
from perception.sensor_preprocessing import SensorCalibration
from perception.video_processing import (
    ProcessingRunInfo,
    SessionProcessingRunner,
    VideoProcessingService,
)

from .state import CommandResult, RuntimeSnapshot

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    host: str = "0.0.0.0"
    port: int = 8770
    discovery_port: int = DISCOVERY_PORT
    recordings_root: Path = Path("local-data/recordings")
    pairing_token: str | None = None
    enable_discovery: bool = True

    def __post_init__(self) -> None:
        if self.port not in range(1, 65_536):
            raise ValueError("port must be between 1 and 65535")
        if self.discovery_port not in range(1, 65_536):
            raise ValueError("discovery_port must be between 1 and 65535")
        if self.pairing_token is not None and len(self.pairing_token) < 16:
            raise ValueError("pairing_token must contain at least 16 characters")


@dataclass(frozen=True, slots=True)
class _CommandDetail:
    detail: str


class UnifiedRuntimeHost:
    """Run gateway, recording, perception, and UI state collection in one process."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        configuration_service: ConfigurationService | None = None,
    ) -> None:
        self.config = config
        self.pairing_token = config.pairing_token or secrets.token_urlsafe(24)
        self.frame_buffer = LiveFrameBuffer()
        self.imu_preview = ImuPreviewRuntime()
        self.webrtc = WebRtcSessionRuntime(
            self.pairing_token,
            display_frame_sink=self.frame_buffer,
            display_imu_sink=self.imu_preview,
        )
        recordings_root = config.recordings_root.expanduser().resolve()
        self.configuration_service = configuration_service or ConfigurationService(
            recordings_root=recordings_root,
        )
        self.recording = RecordingRuntime(
            recordings_root,
            lambda: self.webrtc.recording_source(),
        )
        config_directory = self.configuration_service.config_directory
        sensor_config_path = config_directory / "sensor-preprocessing.yaml"
        self.perception = HandTrackingRuntime(
            runtime_config_path=config_directory / "perception-runtime.yaml",
            sensor_config_path=sensor_config_path,
            hand_tracking_config_path=config_directory / "live-hand-tracking.yaml",
        )
        self.video_processing = VideoProcessingService(
            recordings_root,
            runner=SessionProcessingRunner(
                sensor_config_path=sensor_config_path,
                offline_hand_tracking_config_path=config_directory
                / "offline-hand-tracking.yaml",
            ),
            on_gpu_job_changed=self._on_gpu_job_changed,
            configuration_provenance_provider=self._configuration_provenance,
        )
        self._apply_video_processing_defaults(
            self.configuration_service.snapshot()
            .require_module("video_processing")
            .values
        )
        self.discovery = (
            LanDiscoveryService(
                self.pairing_token,
                config.port,
                discovery_port=config.discovery_port,
            )
            if config.enable_discovery
            else None
        )
        self.app = create_app(
            webrtc_runtime=self.webrtc,
            discovery_service=self.discovery,
            recording_runtime=self.recording,
            perception_runtime=self.perception,
            live_frame_buffer=self.frame_buffer,
            imu_preview_runtime=self.imu_preview,
            recordings_root=recordings_root,
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._snapshot_lock = threading.Lock()
        self._snapshot = RuntimeSnapshot()
        self._command_results: queue.SimpleQueue[CommandResult] = queue.SimpleQueue()
        self._perception_results: queue.Queue[dict[str, object]] = queue.Queue(maxsize=1)
        self._startup_error: BaseException | None = None
        self._recent_events: deque[str] = deque(maxlen=50)
        self._library: RecordingLibrary | None = None
        self._library_refresh_lock = asyncio.Lock()

    def start(self, timeout_seconds: float = 20.0) -> None:
        if self._thread is not None:
            raise RuntimeError("runtime is already started")
        self._thread = threading.Thread(
            target=self._thread_main,
            name="egoglass-runtime",
            daemon=False,
        )
        self._thread.start()
        if not self._ready.wait(timeout_seconds):
            self.stop()
            raise TimeoutError("EgoGlass runtime did not become ready")
        if self._startup_error is not None:
            raise RuntimeError("EgoGlass runtime failed to start") from self._startup_error
        self._record_event(f"runtime listening on {self.config.host}:{self.config.port}")

    def stop(self, timeout_seconds: float = 20.0) -> None:
        if self._thread is None:
            asyncio.run(self._close_local_resources())
            return
        server = self._server
        if server is not None and not server.should_exit:
            try:
                future = self.submit(self._request_session("finalize"))
                future.result(timeout=min(timeout_seconds, 15.0))
            except Exception:
                LOGGER.exception("capture session finalization failed during shutdown")
            server.should_exit = True
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout_seconds)
        if thread is not None and thread.is_alive():
            raise TimeoutError("EgoGlass runtime did not stop cleanly")

    async def _close_local_resources(self) -> None:
        """Release resources for hosts constructed by tests or embedding code."""

        await self.webrtc.close()
        await self.frame_buffer.close()
        await self.perception.close()
        await self.imu_preview.close()
        await self.recording.close()
        await asyncio.to_thread(self.video_processing.close)

    def snapshot(self) -> RuntimeSnapshot:
        with self._snapshot_lock:
            return self._snapshot

    def latest_frame(self) -> LiveFrame | None:
        return self.frame_buffer.next_for_display()

    def take_latest_perception_result(self) -> dict[str, object] | None:
        try:
            return self._perception_results.get_nowait()
        except queue.Empty:
            return None

    def command_results(self) -> tuple[CommandResult, ...]:
        results: list[CommandResult] = []
        while True:
            try:
                results.append(self._command_results.get_nowait())
            except queue.Empty:
                return tuple(results)

    def submit(self, coroutine: Coroutine[Any, Any, Any]) -> concurrent.futures.Future[Any]:
        loop = self._loop
        if loop is None or loop.is_closed():
            coroutine.close()
            raise RuntimeError("runtime event loop is unavailable")
        return asyncio.run_coroutine_threadsafe(coroutine, loop)

    def request_stream(self, action: StreamControlAction) -> None:
        self._track_command("stream", self._request_stream(action))

    def request_recording(self, action: str) -> None:
        if action == "start":
            operation = self.recording.start()
        elif action == "stop":
            operation = self.recording.stop()
        else:
            raise ValueError("recording action must be start or stop")
        self._track_command(f"recording-{action}", operation)

    def request_session(self, action: str) -> None:
        if action not in {"new", "finalize"}:
            raise ValueError("session action must be new or finalize")
        self._track_command(f"session-{action}", self._request_session(action))

    def request_processing(
        self,
        session_id: str,
        *,
        clip_id: str | None = None,
        preset_id: str | None = None,
    ) -> None:
        self._track_sync_command(
            "start-processing",
            lambda: self.video_processing.enqueue(
                session_id,
                clip_id=clip_id,
                preset_id=preset_id,
            ),
        )

    def request_processing_cancel(self, job_id: str) -> None:
        self._track_sync_command(
            "cancel-processing", lambda: self.video_processing.cancel(job_id)
        )

    def request_processing_retry(self, job_id: str) -> None:
        self._track_sync_command(
            "retry-processing", lambda: self.video_processing.retry(job_id)
        )

    def request_live_inference(self, enabled: bool) -> None:
        self._track_command("live-inference", self.perception.set_live_enabled(enabled))

    def request_processing_export(
        self,
        session_id: str,
        run_id: str,
        clip_id: str,
    ) -> None:
        self._track_sync_command(
            "export-processing",
            lambda: self.video_processing.export_annotated_clip(
                session_id,
                run_id,
                clip_id,
            ),
        )

    def configuration_snapshot(self) -> ConfigSnapshot:
        return self.configuration_service.snapshot()

    def stage_configuration(self, values: Mapping[str, object]) -> ConfigSnapshot:
        return self.configuration_service.stage(values)

    def validate_configuration(
        self,
        values: Mapping[str, object] | None = None,
    ) -> tuple[ValidationIssue, ...]:
        return self.configuration_service.validate(values)

    def discard_configuration(self) -> ConfigSnapshot:
        self.configuration_service.discard()
        return self.configuration_service.snapshot()

    def restore_configuration_defaults(self, module_id: str) -> ConfigSnapshot:
        self.configuration_service.restore_defaults(module_id)
        return self.configuration_service.snapshot()

    def request_configuration_save(
        self,
        values: Mapping[str, object] | None = None,
        *,
        apply: bool,
    ) -> concurrent.futures.Future[ConfigApplyResult]:
        async def save() -> ConfigApplyResult:
            result = await asyncio.to_thread(self.configuration_service.save, values)
            if apply and result.changed_modules:
                request = self.configuration_service.apply_request(result)
                result = await self._apply_configuration(request)
            return result

        return self._track_command(
            "configuration-save",
            save(),
            detail_provider=lambda result: _configuration_result_detail(
                result,
                applied=apply,
            ),
        )

    def apply_configuration(
        self,
        request: ConfigurationApplyRequest,
    ) -> concurrent.futures.Future[ConfigApplyResult]:
        return self.submit(self._apply_configuration(request))

    def request_library_refresh(self) -> None:
        self._track_command("refresh-library", self._refresh_library())

    def request_imu_pose_reset(self) -> None:
        self._track_command("imu-pose-reset", self._reset_imu_pose())

    def session_directory(self, session_id: str) -> Path:
        return self.video_processing.session_directory(session_id)

    def processing_runs(
        self, session_id: str
    ) -> concurrent.futures.Future[tuple[ProcessingRunInfo, ...]]:
        return self.submit(asyncio.to_thread(self.video_processing.list_runs, session_id))

    def processing_result(
        self,
        session_id: str,
        run_id: str,
        clip_id: str,
        frame_index: int,
        session_time_ns: int,
    ) -> concurrent.futures.Future[dict[str, object] | None]:
        return self.submit(
            asyncio.to_thread(
                self.video_processing.result_for_frame,
                session_id,
                run_id,
                clip_id,
                frame_index,
                session_time_ns,
            )
        )

    async def _request_stream(self, action: StreamControlAction) -> object:
        if action is StreamControlAction.STOP:
            recording = await self.recording.status()
            if recording.state in {
                RecordingState.COUNTDOWN,
                RecordingState.RECORDING,
                RecordingState.FINALIZING,
            }:
                raise RuntimeError("stop the active recording before stopping the stream")
        return await self.webrtc.send_control_command(
            StreamControlCommand(
                command_id=secrets.token_hex(16),
                action=action,
            )
        )

    async def _request_session(self, action: str) -> object:
        previous = await self.recording.status()
        completed_session_id = previous.session_id
        result = await self.recording.session_command(action)
        if completed_session_id is not None:
            await asyncio.to_thread(
                self.video_processing.enqueue_completed_session,
                completed_session_id,
            )
        return result

    async def _reset_imu_pose(self) -> object:
        self.imu_preview.reset_orientation()
        return _CommandDetail("IMU pose reset")

    async def _apply_configuration(
        self,
        request: ConfigurationApplyRequest,
    ) -> ConfigApplyResult:
        warnings: list[str] = []
        live_values = request.values.get("live_hand_tracking")
        if isinstance(live_values, Mapping):
            runtime_values = live_values.get("runtime")
            if isinstance(runtime_values, Mapping):
                await self.perception.apply_runtime_config(
                    HandTrackingRuntimeConfig.model_validate(dict(runtime_values))
                )
            algorithm_changed = any(
                change.module_id == "live_hand_tracking"
                and change.field_path.startswith("algorithm.")
                for change in request.changes
            )
            if algorithm_changed:
                perception_status = await self.perception.status()
                if perception_status["offline_processing"]:
                    warnings.append(
                        "离线任务正在占用 GPU，实时手部模型配置将在实时推理恢复时加载"
                    )
                else:
                    await self.perception.reload_tracker_configuration()

        video_values = request.values.get("video_processing")
        if isinstance(video_values, Mapping):
            self._apply_video_processing_defaults(video_values)

        return _configuration_result_from_request(
            request,
            self.configuration_service.provenance(),
            tuple(warnings),
        )

    def _apply_video_processing_defaults(self, values: Mapping[str, object]) -> None:
        self.video_processing.set_configuration_defaults(
            default_preset_id=str(values["default_preset_id"]),
            auto_enqueue_on_session_complete=bool(
                values["auto_enqueue_on_session_complete"]
            ),
            default_output_result_type=str(values["default_output_result_type"]),
        )

    def _configuration_provenance(self) -> tuple[int, Mapping[str, str], str]:
        provenance = self.configuration_service.provenance()
        snapshot = self.configuration_service.snapshot()
        sensor = _mutable_configuration_values(
            snapshot.require_module("sensor_preprocessing").values
        )
        offline_hand = _mutable_configuration_values(
            snapshot.require_module("offline_hand_tracking").values
        )
        calibration_path = Path(str(sensor["calibration_file"]))
        if not calibration_path.is_absolute():
            calibration_path = (
                self.configuration_service.config_directory / calibration_path
            )
        sensor["calibration_file"] = str(calibration_path.resolve())
        model_directory = Path(str(offline_hand["model_directory"]))
        if not model_directory.is_absolute():
            model_directory = (
                self.configuration_service.config_directory / model_directory
            )
        offline_hand["model_directory"] = str(model_directory.resolve())
        execution_snapshot = {
            "sensor_preprocessing": sensor,
            "sensor_calibration": SensorCalibration.load(
                calibration_path
            ).model_dump(mode="json"),
            "offline_hand_tracking": offline_hand,
        }
        return (
            provenance.revision,
            provenance.sha256_by_file,
            json.dumps(execution_snapshot, ensure_ascii=False, sort_keys=True),
        )

    def _track_command(
        self,
        name: str,
        coroutine: Coroutine[Any, Any, Any],
        *,
        detail_provider: Callable[[Any], str] | None = None,
    ) -> concurrent.futures.Future[Any]:
        future = self.submit(coroutine)

        def completed(done: concurrent.futures.Future[Any]) -> None:
            try:
                result = done.result()
                detail = (
                    detail_provider(result)
                    if detail_provider is not None
                    else getattr(result, "detail", None) or "completed"
                )
                self._record_event(f"{name}: {detail}")
                self._command_results.put(CommandResult(name, True, str(detail)))
            except Exception as error:
                LOGGER.exception("runtime command failed: %s", name)
                self._record_event(f"{name} failed: {error}")
                self._command_results.put(CommandResult(name, False, str(error)))

        future.add_done_callback(completed)
        return future

    def _track_sync_command(self, name: str, operation: Callable[[], object]) -> None:
        async def run() -> object:
            return await asyncio.to_thread(operation)

        self._track_command(name, run())

    def _on_gpu_job_changed(self, active: bool) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            raise RuntimeError("runtime event loop is unavailable for GPU ownership change")
        future = asyncio.run_coroutine_threadsafe(
            self.perception.set_offline_processing(active),
            loop,
        )
        future.result(timeout=120.0)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as error:
            self._startup_error = error
            LOGGER.exception("unified EgoGlass runtime stopped with an error")
        finally:
            self._ready.set()
            self._stopped.set()

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.video_processing.start()
        config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            access_log=False,
            log_config=None,
        )
        self._server = uvicorn.Server(config)
        status_task = asyncio.create_task(self._collect_status())
        perception_task = asyncio.create_task(self._forward_perception_results())
        library_task = asyncio.create_task(self._initial_library_refresh())
        server_task = asyncio.create_task(self._server.serve())
        while not self._server.started and not server_task.done():
            await asyncio.sleep(0.02)
        if not self._server.started:
            await server_task
            raise RuntimeError("ingest server stopped before becoming ready")
        self._ready.set()
        try:
            await server_task
        finally:
            status_task.cancel()
            perception_task.cancel()
            await asyncio.gather(
                status_task,
                perception_task,
                library_task,
                return_exceptions=True,
            )
            await asyncio.to_thread(self.video_processing.close)

    async def _initial_library_refresh(self) -> None:
        try:
            await self._refresh_library()
        except Exception:
            LOGGER.exception("initial recording library scan failed")

    async def _refresh_library(self) -> RecordingLibrary:
        async with self._library_refresh_lock:
            self._library = await self.recording.library()
            return self._library

    async def _collect_status(self) -> None:
        revision = 0
        while True:
            try:
                now_ns = time.perf_counter_ns()
                webrtc, stream_control, imu, recording, perception = await asyncio.gather(
                    self.webrtc.status(),
                    self.webrtc.control_status(),
                    self.webrtc.imu_status(),
                    self.recording.status(),
                    self.perception.status(),
                )
                processing = self.video_processing.snapshot()
                revision += 1
                snapshot = RuntimeSnapshot(
                    revision=revision,
                    captured_at_client_monotonic_ns=now_ns,
                    server_ready=True,
                    webrtc=webrtc,
                    stream_control=stream_control,
                    imu=imu,
                    imu_pose=self.imu_preview.snapshot(),
                    recording=recording,
                    library=self._library,
                    perception=perception,
                    processing=processing,
                    display=self.frame_buffer.status(),
                    recent_events=tuple(self._recent_events),
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                LOGGER.exception("failed to collect UI runtime status")
                previous = self.snapshot()
                snapshot = RuntimeSnapshot(
                    revision=previous.revision + 1,
                    captured_at_client_monotonic_ns=time.perf_counter_ns(),
                    server_ready=True,
                    webrtc=previous.webrtc,
                    stream_control=previous.stream_control,
                    imu=previous.imu,
                    imu_pose=self.imu_preview.snapshot(),
                    recording=previous.recording,
                    library=previous.library,
                    perception=previous.perception,
                    processing=previous.processing,
                    display=self.frame_buffer.status(),
                    last_error=str(error),
                    recent_events=tuple(self._recent_events),
                )
            with self._snapshot_lock:
                self._snapshot = snapshot
            await asyncio.sleep(0.1)

    async def _forward_perception_results(self) -> None:
        last_result_key: tuple[str, str, int] | None = None
        async for payload in self.perception.status_events():
            if payload is None:
                continue
            result = payload.get("latest_result")
            if not isinstance(result, dict):
                continue
            result_key = _perception_result_key(result)
            if result_key is None or result_key == last_result_key:
                continue
            last_result_key = result_key
            with suppress(queue.Empty):
                self._perception_results.get_nowait()
            self._perception_results.put_nowait(result)

    def _record_event(self, detail: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self._recent_events.appendleft(f"{timestamp}  {detail}")


def _perception_result_key(
    result: dict[str, object],
) -> tuple[str, str, int] | None:
    session_id = result.get("session_id")
    sequence_id = result.get("sequence_id")
    frame_index = result.get("frame_index")
    if (
        not isinstance(session_id, str)
        or not isinstance(sequence_id, str)
        or not isinstance(frame_index, int)
        or isinstance(frame_index, bool)
    ):
        return None
    return session_id, sequence_id, frame_index


def _configuration_result_from_request(
    request: ConfigurationApplyRequest,
    provenance: ConfigurationProvenance,
    warnings: tuple[str, ...],
) -> ConfigApplyResult:
    module_order = (
        "client_runtime",
        "sensor_preprocessing",
        "live_hand_tracking",
        "offline_hand_tracking",
        "video_processing",
    )

    def modules_for(impact: ConfigImpact) -> tuple[str, ...]:
        selected = {
            change.module_id for change in request.changes if change.impact is impact
        }
        return tuple(
            module_id
            for module_id in module_order
            if module_id in selected
        )

    changed = {
        change.module_id for change in request.changes
    }
    return ConfigApplyResult(
        changed_modules=tuple(
            module_id
            for module_id in module_order
            if module_id in changed
        ),
        immediate_applied=modules_for(ConfigImpact.IMMEDIATE),
        pending_restart=modules_for(ConfigImpact.RESTART_CLIENT),
        pending_next_task=modules_for(ConfigImpact.NEXT_TASK),
        pending_next_session=modules_for(ConfigImpact.NEXT_SESSION),
        warnings=warnings,
        changes=request.changes,
        provenance=provenance,
    )


def _configuration_result_detail(result: ConfigApplyResult, *, applied: bool) -> str:
    if not result.changed_modules:
        return "配置没有变化"
    details = [f"已保存 {len(result.changed_modules)} 个模块"]
    if applied and result.immediate_applied:
        details.append("可热更新参数已应用")
    if result.pending_next_session:
        details.append("部分参数将在下次会话生效")
    if result.pending_next_task:
        details.append("部分参数将在下次任务生效")
    if result.pending_restart:
        details.append("部分参数需要重启客户端")
    return "；".join(details)


def _configuration_result_detail(
    result: ConfigApplyResult,
    *,
    applied: bool,
) -> str:
    if not result.changed_modules:
        return "No configuration changes"
    details = [f"Saved {len(result.changed_modules)} module(s)"]
    if applied and result.immediate_applied:
        details.append("Immediate values applied")
    if result.pending_next_session:
        details.append("Some values apply next session")
    if result.pending_next_task:
        details.append("Some values apply next task")
    if result.pending_restart:
        details.append("Some values require a client restart")
    return "; ".join(details)


def _mutable_configuration_values(value: object) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _mutable_configuration_values(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_mutable_configuration_values(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
