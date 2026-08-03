from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import queue
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable, Coroutine
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
from perception.runtime import HandTrackingRuntime
from perception.video_processing import VideoProcessingService

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

    def __init__(self, config: RuntimeConfig) -> None:
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
        self.recording = RecordingRuntime(
            recordings_root,
            lambda: self.webrtc.recording_source(),
        )
        self.perception = HandTrackingRuntime()
        self.video_processing = VideoProcessingService(
            recordings_root,
            on_gpu_job_changed=self._on_gpu_job_changed,
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
        preset_id: str = "hand-tracking-quality",
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

    def request_processing_auto_queue(self, enabled: bool) -> None:
        self._track_sync_command(
            "processing-auto-queue",
            lambda: _CommandDetail(
                "自动入队已开启"
                if self.video_processing.set_auto_enqueue(enabled)
                else "自动入队已关闭"
            ),
        )

    def request_library_refresh(self) -> None:
        self._track_command("refresh-library", self._refresh_library())

    def request_imu_pose_reset(self) -> None:
        self._track_command("imu-pose-reset", self._reset_imu_pose())

    def session_directory(self, session_id: str) -> Path:
        return self.video_processing.session_directory(session_id)

    def processing_runs(
        self, session_id: str
    ) -> concurrent.futures.Future[tuple[dict[str, object], ...]]:
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

    def _track_command(self, name: str, coroutine: Coroutine[Any, Any, Any]) -> None:
        future = self.submit(coroutine)

        def completed(done: concurrent.futures.Future[Any]) -> None:
            try:
                result = done.result()
                detail = getattr(result, "detail", None) or "completed"
                self._record_event(f"{name}: {detail}")
                self._command_results.put(CommandResult(name, True, str(detail)))
            except Exception as error:
                LOGGER.exception("runtime command failed: %s", name)
                self._record_event(f"{name} failed: {error}")
                self._command_results.put(CommandResult(name, False, str(error)))

        future.add_done_callback(completed)

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
