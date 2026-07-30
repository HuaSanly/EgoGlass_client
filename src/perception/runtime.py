"""Shared live and recorded hand-tracking orchestration for the client."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
import uuid
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import yaml
from av import VideoFrame
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from perception.sensor_preprocessing import (
    CaptureSessionReader,
    ClockId,
    ClockMappingSegment,
    LiveFrameInput,
    SegmentedClockMapper,
    SensorPreprocessingConfig,
    SensorPreprocessingPipeline,
    TimeObservation,
    TimestampSemantic,
    TimeStatus,
    client_perf_source_instance_id,
    derive_recorded_clock_mapping,
    persist_recorded_clock_mapping,
)
from perception.spatial_perception.hand_tracking import (
    HandTrackingResult,
    HumanEgoHandTrackingPipeline,
    render_hand_tracking_overlay,
)

LOGGER = logging.getLogger(__name__)
_MAX_CLIENT_PERF_COUNTER_NS = 2**63 - 1


class PerceptionRuntimeError(RuntimeError):
    """A live frame or replay cannot be processed by the perception runtime."""


class HandTrackingRuntimeState(StrEnum):
    """State shown to the native UI for the shared HaMeR worker."""

    DISABLED = "disabled"
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class ReplayState(StrEnum):
    """Lifecycle of a recorded session replay job."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"


class ReplayConfig(BaseModel):
    """Sampling control for a stored-session visual replay."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    inference_stride_frames: int = Field(default=5, ge=1, le=120)


class HandTrackingRuntimeConfig(BaseModel):
    """Small operational controls around the model's fixed algorithm settings."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str = "1.0"
    enabled: bool = True
    max_live_inference_fps: float = Field(default=6.0, gt=0.0, le=60.0)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)

    @classmethod
    def load(cls, path: str | Path) -> HandTrackingRuntimeConfig:
        try:
            payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("runtime config must be a mapping")
            return cls.model_validate(payload)
        except (OSError, UnicodeError, ValueError, ValidationError, yaml.YAMLError) as exc:
            raise PerceptionRuntimeError("invalid perception runtime config") from exc


@dataclass(frozen=True, slots=True)
class LiveHandTrackingFrame:
    """One decoded WebRTC frame submitted without blocking the media callback."""

    session_id: str
    connection_session_id: str
    frame_index: int
    received_at_client_monotonic_ns: int
    decoded_frame: VideoFrame


@dataclass(frozen=True, slots=True)
class ReplayReport:
    """Persistent locations and counters created by one offline replay."""

    session_id: str
    run_id: str
    output_directory: Path
    clock_mapping_path: Path
    result_path: Path
    video_paths: tuple[Path, ...]
    input_frame_count: int
    inferred_frame_count: int
    detected_hand_count: int

    def to_json_dict(self, recordings_root: Path) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "output_relative_path": self.output_directory.relative_to(recordings_root).as_posix(),
            "clock_mapping_relative_path": self.clock_mapping_path.relative_to(
                recordings_root
            ).as_posix(),
            "result_relative_path": self.result_path.relative_to(recordings_root).as_posix(),
            "videos": [
                {
                    "clip_id": path.stem.removeprefix("hand-tracking-"),
                    "relative_path": path.relative_to(recordings_root).as_posix(),
                }
                for path in self.video_paths
            ],
            "input_frame_count": self.input_frame_count,
            "inferred_frame_count": self.inferred_frame_count,
            "detected_hand_count": self.detected_hand_count,
        }


class _H264ReplayWriter:
    """Synchronous browser-compatible H.264 MP4 writer for annotated frames."""

    _TIME_BASE = Fraction(1, 90_000)

    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        rate = Fraction(str(fps)).limit_denominator(1001)
        if rate <= 0:
            raise ValueError("replay frame rate must be positive")
        self._container = av.open(
            str(path),
            mode="w",
            format="mp4",
            options={"movflags": "+faststart"},
        )
        self._stream = self._container.add_stream(
            "libx264",
            rate=rate,
            options={"crf": "18", "preset": "veryfast"},
        )
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._stream.time_base = self._TIME_BASE
        self._stream.codec_context.time_base = self._TIME_BASE
        self._width = width
        self._height = height
        self._frame_count = 0
        self._origin_time_ns: int | None = None
        self._last_pts: int | None = None
        self._closed = False

    def write(self, image_bgr: np.ndarray, presentation_time_ns: int) -> None:
        """Encode one BGR frame using its strictly increasing session time."""

        if self._closed:
            raise RuntimeError("replay writer is closed")
        if image_bgr.shape != (self._height, self._width, 3):
            raise ValueError("replay frame dimensions changed during encoding")
        if type(presentation_time_ns) is not int:
            raise TypeError("replay presentation time must be an integer")
        if self._origin_time_ns is None:
            self._origin_time_ns = presentation_time_ns
        relative_time_ns = presentation_time_ns - self._origin_time_ns
        if relative_time_ns < 0:
            raise ValueError("replay presentation time precedes its clip origin")
        pts = (relative_time_ns * self._TIME_BASE.denominator + 500_000_000) // 1_000_000_000
        if self._last_pts is not None and pts <= self._last_pts:
            raise ValueError("replay presentation time must strictly increase")
        frame = VideoFrame.from_ndarray(image_bgr, format="bgr24")
        frame.pts = pts
        frame.time_base = self._TIME_BASE
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self._last_pts = pts
        self._frame_count += 1

    def close(self) -> None:
        """Flush delayed encoder packets and finalize the fast-start MP4."""

        if self._closed:
            return
        self._closed = True
        try:
            for packet in self._stream.encode(None):
                self._container.mux(packet)
        finally:
            self._container.close()


def render_recorded_hand_tracking_replay(
    session_directory: str | Path,
    output_directory: str | Path,
    *,
    sensor_config_path: str | Path,
    tracker: HumanEgoHandTrackingPipeline,
    inference_stride_frames: int,
    progress: Callable[[int, int], None] | None = None,
) -> ReplayReport:
    """Render one completed capture session with the exact same hand pipeline as live mode."""

    if inference_stride_frames < 1:
        raise ValueError("replay inference stride must be positive")
    session_path = Path(session_directory).resolve()
    output_path = Path(output_directory).resolve()
    reader = CaptureSessionReader.open(session_path)
    frame_evidence = tuple(
        frame
        for clip in reader.session.clips
        for frame in reader.iter_frames(clip.clip_id)
    )
    imu_evidence = tuple(reader.iter_imu_samples())
    recorded_mapping = derive_recorded_clock_mapping(
        reader.session.session_id,
        frame_evidence,
        imu_evidence,
    )
    clock_mapping_path = persist_recorded_clock_mapping(recorded_mapping, session_path)
    preprocessing = SensorPreprocessingPipeline.from_config_file(
        sensor_config_path,
        recorded_mapping.mapper,
    )
    output_path.mkdir(parents=True, exist_ok=False)
    result_path = output_path / "results.jsonl"
    fps_by_clip = {clip.clip_id: clip.nominal_fps for clip in reader.session.clips}
    total_frames = sum(clip.frame_count for clip in reader.session.clips)
    writers: dict[str, _H264ReplayWriter] = {}
    videos: dict[str, Path] = {}
    latest_by_clip: dict[str, HandTrackingResult] = {}
    input_frame_count = 0
    inferred_frame_count = 0
    detected_hand_count = 0

    try:
        with (
            ExitStack() as writer_stack,
            result_path.open("x", encoding="utf-8") as result_stream,
        ):
            for bundle in preprocessing.iter_recorded_session(session_path):
                input_frame_count += 1
                clip_id = bundle.sequence_id
                if bundle.frame_index % inference_stride_frames == 0:
                    result = tracker.process_frame(bundle)
                    latest_by_clip[clip_id] = result
                    inferred_frame_count += 1
                    detected_hand_count += len(result.hands)
                    result_stream.write(
                        json.dumps(
                            result.to_json_dict(),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                result = latest_by_clip.get(clip_id)
                image = (
                    render_hand_tracking_overlay(bundle.image_bgr, result)
                    if result is not None
                    else bundle.image_bgr.copy()
                )
                writer = writers.get(clip_id)
                if writer is None:
                    video_path = output_path / f"hand-tracking-{clip_id}.mp4"
                    writer = _H264ReplayWriter(
                        video_path,
                        fps_by_clip[clip_id],
                        image.shape[1],
                        image.shape[0],
                    )
                    writer_stack.callback(writer.close)
                    writers[clip_id] = writer
                    videos[clip_id] = video_path
                writer.write(image, bundle.session_time_ns)
                if progress is not None:
                    progress(input_frame_count, total_frames)
    except Exception as exc:
        try:
            shutil.rmtree(output_path)
        except OSError:
            LOGGER.exception("failed to remove incomplete replay output %s", output_path)
        if isinstance(exc, PerceptionRuntimeError):
            raise
        raise PerceptionRuntimeError(
            f"failed to render hand-tracking replay: {exc}"
        ) from exc

    return ReplayReport(
        session_id=reader.session.session_id,
        run_id=output_path.name,
        output_directory=output_path,
        clock_mapping_path=clock_mapping_path,
        result_path=result_path,
        video_paths=tuple(videos.values()),
        input_frame_count=input_frame_count,
        inferred_frame_count=inferred_frame_count,
        detected_hand_count=detected_hand_count,
    )


class HandTrackingRuntime:
    """One serialized CUDA worker shared by live WebRTC and offline replay paths."""

    def __init__(
        self,
        *,
        recordings_root: str | Path,
        runtime_config_path: str | Path = "config/perception-runtime.yaml",
        sensor_config_path: str | Path = "config/sensor-preprocessing.yaml",
        hand_tracking_config_path: str | Path = "config/hand-tracking.yaml",
    ) -> None:
        self.recordings_root = Path(recordings_root).resolve()
        self.config = HandTrackingRuntimeConfig.load(runtime_config_path)
        self.sensor_config_path = Path(sensor_config_path).resolve()
        self.hand_tracking_config_path = Path(hand_tracking_config_path).resolve()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hand-tracking")
        self._lock = asyncio.Lock()
        self._status_condition = asyncio.Condition(self._lock)
        self._status_revision = 0
        self._pending_live_frame: LiveHandTrackingFrame | None = None
        self._live_worker: asyncio.Task[None] | None = None
        self._state = (
            HandTrackingRuntimeState.IDLE
            if self.config.enabled
            else HandTrackingRuntimeState.DISABLED
        )
        self._detail = "waiting for the first decoded frame" if self.config.enabled else "disabled"
        self._last_live_submission_ns: int | None = None
        self._latest_result: HandTrackingResult | None = None
        self._last_error: str | None = None
        self._live_frames_received = 0
        self._live_frames_dropped = 0
        self._live_inferences = 0
        self._replay_state = ReplayState.IDLE
        self._replay_detail = "no replay requested"
        self._replay_progress = (0, 0)
        self._latest_replay: ReplayReport | None = None
        self._replay_task: asyncio.Task[None] | None = None
        self._tracker: HumanEgoHandTrackingPipeline | None = None
        self._live_preprocessor: SensorPreprocessingPipeline | None = None
        self._live_session_id: str | None = None
        self._live_connection_session_id: str | None = None

    async def submit_live_frame(self, frame: LiveHandTrackingFrame) -> None:
        """Keep at most one newest decoded frame while CUDA inference is in flight."""

        if not self.config.enabled:
            return
        minimum_interval_ns = round(1_000_000_000 / self.config.max_live_inference_fps)
        async with self._lock:
            self._live_frames_received += 1
            if (
                self._last_live_submission_ns is not None
                and frame.received_at_client_monotonic_ns - self._last_live_submission_ns
                < minimum_interval_ns
            ):
                self._live_frames_dropped += 1
                return
            self._last_live_submission_ns = frame.received_at_client_monotonic_ns
            if self._pending_live_frame is not None:
                self._live_frames_dropped += 1
            self._pending_live_frame = frame
            if self._live_worker is None or self._live_worker.done():
                self._live_worker = asyncio.create_task(self._run_live_worker())

    async def submit_gateway_frame(
        self,
        *,
        session_id: str,
        connection_session_id: str,
        frame_index: int,
        received_at_client_monotonic_ns: int,
        decoded_frame: VideoFrame,
    ) -> None:
        """Accept one gateway-decoded frame without making WebRTC wait for CUDA work."""

        await self.submit_live_frame(
            LiveHandTrackingFrame(
                session_id=session_id,
                connection_session_id=connection_session_id,
                frame_index=frame_index,
                received_at_client_monotonic_ns=received_at_client_monotonic_ns,
                decoded_frame=decoded_frame,
            )
        )

    async def status(self) -> dict[str, object]:
        """Return the latest JSON-ready state snapshot."""

        async with self._lock:
            return self._status_payload_locked()

    async def status_events(
        self,
        *,
        heartbeat_seconds: float = 15.0,
    ) -> AsyncIterator[dict[str, object] | None]:
        """Push changed status snapshots and yield ``None`` for SSE heartbeats."""

        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat interval must be positive")
        last_revision = -1
        while True:
            heartbeat = False
            async with self._status_condition:
                if self._status_revision == last_revision:
                    expected_revision = last_revision
                    try:
                        await asyncio.wait_for(
                            self._status_condition.wait_for(
                                lambda revision=expected_revision: (
                                    self._status_revision != revision
                                )
                            ),
                            timeout=heartbeat_seconds,
                        )
                    except TimeoutError:
                        heartbeat = True
                if heartbeat:
                    payload = None
                else:
                    last_revision = self._status_revision
                    payload = self._status_payload_locked()
            yield payload

    async def start_replay(self, session_id: str) -> None:
        """Start one stored-session replay and reject concurrent GPU replay requests."""

        session_path = self._session_path(session_id)
        async with self._lock:
            if self._replay_task is not None and not self._replay_task.done():
                raise PerceptionRuntimeError("a hand-tracking replay is already running")
            self._replay_state = ReplayState.RUNNING
            self._replay_detail = "loading stored session"
            self._replay_progress = (0, 0)
            self._latest_replay = None
            self._replay_task = asyncio.create_task(self._run_replay(session_path))
            self._publish_status_locked()

    async def replay_video_path(self, session_id: str, run_id: str, clip_id: str) -> Path:
        """Resolve one generated replay MP4 beneath its owning recording directory."""

        if not session_id or not run_id or not clip_id:
            raise PerceptionRuntimeError("replay path fields cannot be empty")
        candidate = (
            self._session_path(session_id)
            / "perception"
            / "hand-tracking"
            / run_id
            / f"hand-tracking-{clip_id}.mp4"
        ).resolve()
        root = self._session_path(session_id)
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise FileNotFoundError("hand-tracking replay video is unavailable")
        return candidate

    async def close(self) -> None:
        """Drain worker tasks before gateway shutdown releases the process tree."""

        tasks = tuple(task for task in (self._live_worker, self._replay_task) if task is not None)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._executor.shutdown(wait=True, cancel_futures=True)

    async def _run_live_worker(self) -> None:
        while True:
            async with self._lock:
                if self._replay_state is ReplayState.RUNNING:
                    if self._pending_live_frame is not None:
                        self._live_frames_dropped += 1
                        self._pending_live_frame = None
                    return
                frame = self._pending_live_frame
                self._pending_live_frame = None
                if frame is None:
                    return
                self._state = HandTrackingRuntimeState.LOADING
                self._detail = "loading HaMeR" if self._tracker is None else "running HaMeR"
                self._publish_status_locked()
            try:
                result = await asyncio.get_running_loop().run_in_executor(
                    self._executor,
                    self._process_live_frame,
                    frame,
                )
            except Exception as exc:
                LOGGER.exception("live hand tracking failed")
                async with self._lock:
                    self._state = HandTrackingRuntimeState.ERROR
                    self._detail = "latest frame failed"
                    self._last_error = str(exc)
                    self._publish_status_locked()
                continue
            async with self._lock:
                self._state = HandTrackingRuntimeState.READY
                self._detail = "latest result is ready"
                self._last_error = None
                self._latest_result = result
                self._live_inferences += 1
                self._publish_status_locked()

    async def _run_replay(self, session_path: Path) -> None:
        run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
        output_path = session_path / "perception" / "hand-tracking" / run_id
        loop = asyncio.get_running_loop()
        try:
            report = await loop.run_in_executor(
                self._executor,
                self._process_replay,
                session_path,
                output_path,
                loop,
            )
        except Exception as exc:
            LOGGER.exception("hand-tracking replay failed")
            async with self._lock:
                self._replay_state = ReplayState.ERROR
                self._replay_detail = str(exc)
                self._publish_status_locked()
            return
        async with self._lock:
            self._replay_state = ReplayState.COMPLETE
            self._replay_detail = "annotated replay is ready"
            self._latest_replay = report
            self._publish_status_locked()

    def _process_live_frame(
        self,
        frame: LiveHandTrackingFrame,
    ) -> HandTrackingResult:
        preprocessing = self._live_preprocessing_for(frame)
        bundle = preprocessing.process_live_frame(
            frame.decoded_frame,
            LiveFrameInput(
                session_id=frame.session_id,
                stream_id=frame.connection_session_id,
                frame_index=frame.frame_index,
                time_observation=TimeObservation(
                    session_id=frame.session_id,
                    source_clock_id=ClockId.CLIENT_PERF_COUNTER_NS,
                    source_instance_id=client_perf_source_instance_id(
                        frame.session_id,
                        frame.connection_session_id,
                    ),
                    source_timestamp=frame.received_at_client_monotonic_ns,
                    timestamp_semantic=TimestampSemantic.CLIENT_RECEIPT,
                ),
                rotation_degrees=preprocessing.calibration.rotation_degrees,
                capture_config_id=preprocessing.calibration.capture_config_id,
                association_uncertainty_ns=10_000_000,
                association_status=TimeStatus.ESTIMATED,
            ),
        )
        tracker = self._tracker_for_current_thread()
        return tracker.process_frame(bundle)

    def _process_replay(
        self,
        session_path: Path,
        output_path: Path,
        loop: asyncio.AbstractEventLoop,
    ) -> ReplayReport:
        tracker = self._tracker_for_current_thread()
        last_progress_reported_ns = 0

        def update_progress(current: int, total: int) -> None:
            nonlocal last_progress_reported_ns
            now_ns = time.perf_counter_ns()
            if current != total and now_ns - last_progress_reported_ns < 250_000_000:
                return
            last_progress_reported_ns = now_ns
            loop.call_soon_threadsafe(
                asyncio.create_task,
                self._set_replay_progress(current, total),
            )

        return render_recorded_hand_tracking_replay(
            session_path,
            output_path,
            sensor_config_path=self.sensor_config_path,
            tracker=tracker,
            inference_stride_frames=self.config.replay.inference_stride_frames,
            progress=update_progress,
        )

    async def _set_replay_progress(self, current: int, total: int) -> None:
        async with self._lock:
            if self._replay_state is not ReplayState.RUNNING:
                return
            self._replay_progress = (current, total)
            self._replay_detail = f"processing {current}/{total} frames"
            self._publish_status_locked()

    def _status_payload_locked(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "status_revision": self._status_revision,
            "state": self._state.value,
            "detail": self._detail,
            "live_frames_received": self._live_frames_received,
            "live_frames_dropped": self._live_frames_dropped,
            "live_inferences": self._live_inferences,
            "latest_result": (
                self._latest_result.to_json_dict()
                if self._latest_result is not None
                else None
            ),
            "last_error": self._last_error,
            "replay": {
                "state": self._replay_state.value,
                "detail": self._replay_detail,
                "frames_processed": self._replay_progress[0],
                "frame_total": self._replay_progress[1],
                "report": (
                    self._latest_replay.to_json_dict(self.recordings_root)
                    if self._latest_replay is not None
                    else None
                ),
            },
        }

    def _publish_status_locked(self) -> None:
        self._status_revision += 1
        self._status_condition.notify_all()

    def _tracker_for_current_thread(self) -> HumanEgoHandTrackingPipeline:
        if self._tracker is None:
            self._tracker = HumanEgoHandTrackingPipeline.from_config_file(
                str(self.hand_tracking_config_path)
            )
        return self._tracker

    def _live_preprocessing_for(
        self,
        frame: LiveHandTrackingFrame,
    ) -> SensorPreprocessingPipeline:
        if (
            self._live_preprocessor is not None
            and self._live_session_id == frame.session_id
            and self._live_connection_session_id == frame.connection_session_id
        ):
            return self._live_preprocessor
        source_instance_id = client_perf_source_instance_id(
            frame.session_id,
            frame.connection_session_id,
        )
        mapper = SegmentedClockMapper(
            frame.session_id,
            (
                ClockMappingSegment(
                    session_id=frame.session_id,
                    source_clock_id=ClockId.CLIENT_PERF_COUNTER_NS,
                    source_instance_id=source_instance_id,
                    segment_index=0,
                    source_from=frame.received_at_client_monotonic_ns,
                    source_to=_MAX_CLIENT_PERF_COUNTER_NS,
                    source_anchor=frame.received_at_client_monotonic_ns,
                    target_anchor_ns=0,
                    scale_numerator_ns=1,
                    scale_denominator_source_units=1,
                    uncertainty_ns=10_000_000,
                    status=TimeStatus.ESTIMATED,
                    fit_method="client_receipt_identity",
                    provenance_id="gateway-live-hand-tracking-v1",
                    uncertainty_basis="client_receipt_only",
                ),
            ),
        )
        config = SensorPreprocessingConfig.load(self.sensor_config_path)
        self._live_preprocessor = SensorPreprocessingPipeline.from_config_file(
            self.sensor_config_path,
            mapper,
        )
        self._live_session_id = frame.session_id
        self._live_connection_session_id = frame.connection_session_id
        LOGGER.info(
            "hand_tracking_live_preprocessing session_id=%s calibration=%s",
            frame.session_id,
            config.calibration_file,
        )
        return self._live_preprocessor

    def _session_path(self, session_id: str) -> Path:
        if not session_id.strip():
            raise PerceptionRuntimeError("session_id cannot be empty")
        path = (self.recordings_root / session_id).resolve()
        if not path.is_relative_to(self.recordings_root):
            raise PerceptionRuntimeError("session path escapes recordings root")
        if not path.is_dir():
            raise FileNotFoundError("recording session is unavailable")
        return path
