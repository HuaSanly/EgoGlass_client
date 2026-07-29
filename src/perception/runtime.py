"""Shared live and recorded hand-tracking orchestration for the client."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import cv2
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
    """State shown to the operator console for the shared HaMeR worker."""

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
    empty_mapper = SegmentedClockMapper(reader.session.session_id, ())
    preprocessing = SensorPreprocessingPipeline.from_config_file(
        sensor_config_path,
        empty_mapper,
    )
    output_path.mkdir(parents=True, exist_ok=False)
    result_path = output_path / "results.jsonl"
    fps_by_clip = {clip.clip_id: clip.nominal_fps for clip in reader.session.clips}
    total_frames = sum(clip.frame_count for clip in reader.session.clips)
    writers: dict[str, cv2.VideoWriter] = {}
    videos: dict[str, Path] = {}
    latest_by_clip: dict[str, HandTrackingResult] = {}
    input_frame_count = 0
    inferred_frame_count = 0
    detected_hand_count = 0

    try:
        with result_path.open("x", encoding="utf-8") as result_stream:
            for bundle in preprocessing.iter_recorded_session(session_path):
                input_frame_count += 1
                clip_id = bundle.sequence_id
                if bundle.frame_index % inference_stride_frames == 0:
                    result = tracker.process_frame(bundle)
                    latest_by_clip[clip_id] = result
                    inferred_frame_count += 1
                    detected_hand_count += len(result.hands)
                    result_stream.write(
                        json.dumps(result.to_json_dict(), ensure_ascii=False, sort_keys=True) + "\n"
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
                    writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps_by_clip[clip_id],
                        (image.shape[1], image.shape[0]),
                    )
                    if not writer.isOpened():
                        raise PerceptionRuntimeError("failed to open hand-tracking replay writer")
                    writers[clip_id] = writer
                    videos[clip_id] = video_path
                writer.write(image)
                if progress is not None:
                    progress(input_frame_count, total_frames)
    except Exception as exc:
        try:
            shutil.rmtree(output_path)
        except OSError:
            LOGGER.exception("failed to remove incomplete replay output %s", output_path)
        if isinstance(exc, PerceptionRuntimeError):
            raise
        raise PerceptionRuntimeError("failed to render hand-tracking replay") from exc
    finally:
        for writer in writers.values():
            writer.release()

    return ReplayReport(
        session_id=reader.session.session_id,
        run_id=output_path.name,
        output_directory=output_path,
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
        """Return JSON-ready state consumed by the operator console polling loop."""

        async with self._lock:
            return {
                "schema_version": "1.0",
                "state": self._state.value,
                "detail": self._detail,
                "live_frames_received": self._live_frames_received,
                "live_frames_dropped": self._live_frames_dropped,
                "live_inferences": self._live_inferences,
                "latest_result": (
                    self._latest_result.to_json_dict() if self._latest_result is not None else None
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
                continue
            async with self._lock:
                self._state = HandTrackingRuntimeState.READY
                self._detail = "latest result is ready"
                self._last_error = None
                self._latest_result = result
                self._live_inferences += 1

    async def _run_replay(self, session_path: Path) -> None:
        run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
        output_path = session_path / "perception" / "hand-tracking" / run_id
        try:
            report = await asyncio.get_running_loop().run_in_executor(
                self._executor,
                self._process_replay,
                session_path,
                output_path,
            )
        except Exception as exc:
            LOGGER.exception("hand-tracking replay failed")
            async with self._lock:
                self._replay_state = ReplayState.ERROR
                self._replay_detail = str(exc)
            return
        async with self._lock:
            self._replay_state = ReplayState.COMPLETE
            self._replay_detail = "annotated replay is ready"
            self._latest_replay = report

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

    def _process_replay(self, session_path: Path, output_path: Path) -> ReplayReport:
        tracker = self._tracker_for_current_thread()

        def update_progress(current: int, total: int) -> None:
            self._replay_progress = (current, total)
            self._replay_detail = f"processing {current}/{total} frames"

        return render_recorded_hand_tracking_replay(
            session_path,
            output_path,
            sensor_config_path=self.sensor_config_path,
            tracker=tracker,
            inference_stride_frames=self.config.replay.inference_stride_frames,
            progress=update_progress,
        )

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
