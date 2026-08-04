"""Optional live hand-tracking orchestration for the client."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import yaml
from av import VideoFrame
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from perception.sensor_preprocessing import (
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
    release_pipeline_resources,
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


class HandTrackingRuntimeConfig(BaseModel):
    """Small operational controls around the model's fixed algorithm settings."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str = "1.0"
    enabled: bool = False
    max_live_inference_fps: float = Field(default=6.0, gt=0.0, le=60.0)

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
    """One decoded WebRTC or canonical RGB frame queued for hand tracking."""

    session_id: str
    connection_session_id: str
    frame_index: int
    received_at_client_monotonic_ns: int
    decoded_frame: VideoFrame | None = None
    image_rgb: np.ndarray | None = None

    def __post_init__(self) -> None:
        if (self.decoded_frame is None) == (self.image_rgb is None):
            raise ValueError("exactly one decoded_frame or image_rgb input is required")
        if self.image_rgb is not None:
            if (
                self.image_rgb.dtype != np.uint8
                or self.image_rgb.ndim != 3
                or self.image_rgb.shape[2] != 3
            ):
                raise TypeError("image_rgb must be an uint8 HxWx3 array")
            if not self.image_rgb.flags.c_contiguous:
                raise ValueError("image_rgb must be C-contiguous")
            if self.image_rgb.flags.writeable:
                raise ValueError("image_rgb must be read-only")


class HandTrackingRuntime:
    """One serialized CUDA worker for optional live hand tracking."""

    def __init__(
        self,
        *,
        runtime_config_path: str | Path = "config/perception-runtime.yaml",
        sensor_config_path: str | Path = "config/sensor-preprocessing.yaml",
        hand_tracking_config_path: str | Path = "config/live-hand-tracking.yaml",
    ) -> None:
        self.runtime_config_path = Path(runtime_config_path).resolve()
        self.config = HandTrackingRuntimeConfig.load(self.runtime_config_path)
        self.sensor_config_path = Path(sensor_config_path).resolve()
        self.hand_tracking_config_path = Path(hand_tracking_config_path).resolve()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hand-tracking")
        self._lock = asyncio.Lock()
        self._status_condition = asyncio.Condition(self._lock)
        self._status_revision = 0
        self._pending_live_frame: LiveHandTrackingFrame | None = None
        self._live_worker: asyncio.Task[None] | None = None
        self._live_enabled = self.config.enabled
        self._offline_processing = False
        self._reloading_tracker = False
        self._state = (
            HandTrackingRuntimeState.IDLE
            if self._live_enabled
            else HandTrackingRuntimeState.DISABLED
        )
        self._detail = "waiting for the first decoded frame" if self._live_enabled else "disabled"
        self._last_live_submission_ns: int | None = None
        self._latest_result: HandTrackingResult | None = None
        self._last_error: str | None = None
        self._live_frames_received = 0
        self._live_frames_dropped = 0
        self._live_inferences = 0
        self._tracker: HumanEgoHandTrackingPipeline | None = None
        self._live_preprocessor: SensorPreprocessingPipeline | None = None
        self._live_session_id: str | None = None
        self._live_connection_session_id: str | None = None

    async def submit_live_frame(self, frame: LiveHandTrackingFrame) -> None:
        """Keep at most one newest decoded frame while CUDA inference is in flight."""

        if not self._live_enabled or self._offline_processing or self._reloading_tracker:
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

    async def submit_rgb_frame(
        self,
        *,
        session_id: str,
        connection_session_id: str,
        frame_index: int,
        received_at_client_monotonic_ns: int,
        image_rgb: np.ndarray,
    ) -> None:
        """Accept the immutable RGB frame already prepared for the native display."""

        await self.submit_live_frame(
            LiveHandTrackingFrame(
                session_id=session_id,
                connection_session_id=connection_session_id,
                frame_index=frame_index,
                received_at_client_monotonic_ns=received_at_client_monotonic_ns,
                image_rgb=image_rgb,
            )
        )

    async def status(self) -> dict[str, object]:
        """Return the latest JSON-ready state snapshot."""

        async with self._lock:
            return self._status_payload_locked()

    async def set_live_enabled(self, enabled: bool) -> None:
        """Enable optional live inference without changing preview or recording."""

        live_worker: asyncio.Task[None] | None = None
        async with self._lock:
            if enabled and self._offline_processing:
                raise PerceptionRuntimeError("offline processing currently owns the GPU")
            self._live_enabled = enabled
            self._pending_live_frame = None
            if not enabled:
                live_worker = self._live_worker
            self._state = (
                HandTrackingRuntimeState.IDLE
                if enabled
                else HandTrackingRuntimeState.DISABLED
            )
            self._detail = "waiting for the first decoded frame" if enabled else "disabled"
            self._publish_status_locked()
        if not enabled:
            if live_worker is not None and live_worker is not asyncio.current_task():
                await asyncio.gather(live_worker, return_exceptions=True)
            await self._release_tracker()

    async def apply_runtime_config(self, config: HandTrackingRuntimeConfig) -> None:
        """Apply live controls without reconstructing the media path."""

        self.config = config
        if config.enabled != self._live_enabled:
            await self.set_live_enabled(config.enabled)
            return
        async with self._lock:
            self._publish_status_locked()

    async def reload_tracker_configuration(self) -> None:
        """Drain one in-flight inference and recreate the tracker on demand."""

        live_worker: asyncio.Task[None] | None
        async with self._lock:
            self._reloading_tracker = True
            if self._pending_live_frame is not None:
                self._live_frames_dropped += 1
                self._pending_live_frame = None
            live_worker = self._live_worker
        try:
            if live_worker is not None and live_worker is not asyncio.current_task():
                await asyncio.gather(live_worker, return_exceptions=True)
            await self._release_tracker()
            async with self._lock:
                self._latest_result = None
                self._detail = (
                    "waiting for the first decoded frame"
                    if self._live_enabled
                    else "disabled"
                )
                self._publish_status_locked()
        finally:
            async with self._lock:
                self._reloading_tracker = False

    async def set_offline_processing(self, active: bool) -> None:
        """Pause live inference while the offline worker exclusively owns the GPU."""

        live_worker: asyncio.Task[None] | None = None
        async with self._lock:
            self._offline_processing = active
            if active:
                self._pending_live_frame = None
                live_worker = self._live_worker
                self._state = HandTrackingRuntimeState.DISABLED
                self._detail = "paused while offline processing owns the GPU"
            else:
                self._state = (
                    HandTrackingRuntimeState.IDLE
                    if self._live_enabled
                    else HandTrackingRuntimeState.DISABLED
                )
                self._detail = (
                    "waiting for the first decoded frame" if self._live_enabled else "disabled"
                )
            self._publish_status_locked()
        if active and live_worker is not None and live_worker is not asyncio.current_task():
            await asyncio.gather(live_worker, return_exceptions=True)
        if active:
            await self._release_tracker()

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

    async def close(self) -> None:
        """Drain worker tasks before gateway shutdown releases the process tree."""

        if self._live_worker is not None:
            await asyncio.gather(self._live_worker, return_exceptions=True)
        await self._release_tracker()
        self._executor.shutdown(wait=True, cancel_futures=True)

    async def _run_live_worker(self) -> None:
        while True:
            async with self._lock:
                if self._offline_processing or not self._live_enabled or self._reloading_tracker:
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
                if self._offline_processing or not self._live_enabled:
                    return
                self._state = HandTrackingRuntimeState.READY
                self._detail = "latest result is ready"
                self._last_error = None
                self._latest_result = result
                self._live_inferences += 1
                self._publish_status_locked()

    def _process_live_frame(
        self,
        frame: LiveHandTrackingFrame,
    ) -> HandTrackingResult:
        preprocessing = self._live_preprocessing_for(frame)
        live_input = LiveFrameInput(
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
        )
        if frame.image_rgb is not None:
            bundle = preprocessing.process_live_rgb_frame(frame.image_rgb, live_input)
        else:
            assert frame.decoded_frame is not None
            bundle = preprocessing.process_live_frame(frame.decoded_frame, live_input)
        tracker = self._tracker_for_current_thread()
        return tracker.process_frame(bundle)

    def _status_payload_locked(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "status_revision": self._status_revision,
            "state": self._state.value,
            "detail": self._detail,
            "live_enabled": self._live_enabled,
            "offline_processing": self._offline_processing,
            "live_frames_received": self._live_frames_received,
            "live_frames_dropped": self._live_frames_dropped,
            "live_inferences": self._live_inferences,
            "latest_result": (
                self._latest_result.to_json_dict()
                if self._latest_result is not None
                else None
            ),
            "last_error": self._last_error,
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

    async def _release_tracker(self) -> None:
        await asyncio.get_running_loop().run_in_executor(
            self._executor,
            self._release_tracker_for_current_thread,
        )

    def _release_tracker_for_current_thread(self) -> None:
        tracker = self._tracker
        self._tracker = None
        release_pipeline_resources(tracker)

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
