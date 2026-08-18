from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Protocol

from av import VideoFrame

from schemas.recording import (
    CameraFrameRow,
    RecordingImuRow,
    RecordingLibrary,
    RecordingOutput,
    RecordingState,
    RecordingStatus,
)
from schemas.recording import (
    ImuSensorType as RecordingImuSensorType,
)

from .adapters.mp4_recorder import PyAvH264Mp4Recorder, RecordedVideoFrame
from .adapters.webrtc import WebRtcVideoRecordingSource
from .capture_recording import (
    CaptureRecordingError,
    CaptureRecordingReader,
    CaptureRecordingWriter,
    StagedCameraFrame,
    StagedImuSample,
    recover_completed_recordings,
)
from .webrtc_matcher import FrameMetadataMatch
from .webrtc_models import ImuCapabilities, ImuSample, VideoFrameMetadata

COUNTDOWN_SECONDS = 3.0
OUTPUT_FPS = 30
FRAME_METADATA_WAIT_SECONDS = 0.5
IMU_TAIL_COVERAGE_TIMEOUT_SECONDS = 1.0
TELEMETRY_DRAIN_TIMEOUT_SECONDS = 30.0
_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
LOGGER = logging.getLogger(__name__)


class RecordingUnavailableError(RuntimeError):
    """Raised when there is no compatible live Glass3 source."""


class RecordingConflictError(RuntimeError):
    """Raised when a command conflicts with the current recording state."""


class RecordingFailureError(RuntimeError):
    """Raised when a recording cannot be finalized without data loss."""


class RecordingNotFoundError(RuntimeError):
    """Raised when a completed recording does not exist or fails validation."""


class _ImuTailCoverageTimeout(CaptureRecordingError):
    """Raised when live IMU cannot cover the final encoded camera frame."""


class RecordingWriter(Protocol):
    @property
    def frames_received(self) -> int: ...

    @property
    def frame_records(self) -> Sequence[RecordedVideoFrame]: ...

    async def start(self) -> None: ...

    async def wait(self) -> None: ...

    async def stop(self) -> None: ...

    async def trim_to_frame_count(self, frame_count: int) -> None: ...


RecordingSourceProvider = Callable[[], Awaitable[WebRtcVideoRecordingSource | None]]


class RecordingWriterFactory(Protocol):
    def __call__(
        self,
        path: Path,
        track: object,
        *,
        width: int,
        height: int,
        fps: int,
    ) -> RecordingWriter: ...


FrameMetadataEligibility = Callable[[int], Awaitable[bool]]


class MetadataMatchedVideoTrack:
    """Expose only decoded frames that have authoritative Glass3 metadata."""

    def __init__(
        self,
        track: object,
        metadata_eligibility: FrameMetadataEligibility,
    ) -> None:
        self._track = track
        self._metadata_eligibility = metadata_eligibility
        self.skipped_frame_count = 0

    async def recv(self) -> VideoFrame:
        while True:
            frame = await self._track.recv()  # type: ignore[attr-defined]
            if not isinstance(frame, VideoFrame):
                raise TypeError("recording track returned a non-video frame")
            if frame.pts is not None and await self._metadata_eligibility(frame.pts):
                return frame
            self.skipped_frame_count += 1


class RecordingRuntime:
    """Own independent, atomically published capture recordings."""

    def __init__(
        self,
        root: Path,
        source_provider: RecordingSourceProvider,
        *,
        recorder_factory: RecordingWriterFactory = PyAvH264Mp4Recorder,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        unix_clock_ns: Callable[[], int] = time.time_ns,
        monotonic_clock_ns: Callable[[], int] = time.perf_counter_ns,
        recording_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
        telemetry_queue_size: int = 32_768,
        imu_tail_coverage_timeout_seconds: float = IMU_TAIL_COVERAGE_TIMEOUT_SECONDS,
    ) -> None:
        if telemetry_queue_size < 1:
            raise ValueError("telemetry_queue_size must be positive")
        if imu_tail_coverage_timeout_seconds < 0:
            raise ValueError("IMU tail coverage timeout cannot be negative")
        self._root = root.expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._source_provider = source_provider
        self._recorder_factory = recorder_factory
        self._sleep = sleep
        self._unix_clock_ns = unix_clock_ns
        self._monotonic_clock_ns = monotonic_clock_ns
        self._recording_id_factory = recording_id_factory
        self._telemetry_queue_size = telemetry_queue_size
        self._imu_tail_coverage_timeout_seconds = imu_tail_coverage_timeout_seconds
        self._command_lock = asyncio.Lock()
        self._state = RecordingState.UNAVAILABLE
        self._detail = "Glass3 video is not ready"
        self._output = RecordingOutput()
        self._recording_id: str | None = None
        self._connection_session_id: str | None = None
        self._camera_start_generation: int | None = None
        self._countdown_started_at_unix_ns: int | None = None
        self._countdown_started_at_monotonic_ns: int | None = None
        self._recording_starts_at_unix_ms: int | None = None
        self._recording_started_at_unix_ms: int | None = None
        self._recording_started_at_monotonic_ns: int | None = None
        self._recording_duration_ms = 0
        self._imu_sample_count = 0
        self._telemetry_queue_overflow_count = 0
        self._writer: CaptureRecordingWriter | None = None
        self._recorder: RecordingWriter | None = None
        self._countdown_task: asyncio.Task[None] | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._telemetry_task: asyncio.Task[None] | None = None
        self._telemetry_queue: asyncio.Queue[StagedImuSample] | None = None
        self._telemetry_error: BaseException | None = None
        self._accepting_imu = False
        self._first_imu_timestamp_ns: dict[RecordingImuSensorType, int] = {}
        self._latest_imu_timestamp_ns: dict[RecordingImuSensorType, int] = {}
        self._imu_sample_available = asyncio.Event()
        self._matches_by_pts: dict[int, FrameMetadataMatch] = {}
        self._metadata_match_available = asyncio.Event()
        self._metadata_matched_track: MetadataMatchedVideoTrack | None = None
        self._recover_partial_directories()

    @property
    def root(self) -> Path:
        return self._root

    async def status(self) -> RecordingStatus:
        async with self._command_lock:
            if self._state in {RecordingState.UNAVAILABLE, RecordingState.READY}:
                await self._refresh_availability_locked()
            return self._status_locked()

    async def start(self) -> RecordingStatus:
        async with self._command_lock:
            if self._state in {
                RecordingState.COUNTDOWN,
                RecordingState.RECORDING,
                RecordingState.FINALIZING,
            }:
                raise RecordingConflictError("a recording command is already active")
            source = await self._compatible_source()
            recording_id = self._recording_id_factory()
            if not _ID_PATTERN.fullmatch(recording_id):
                raise RecordingFailureError("recording ID factory returned an invalid ID")
            countdown_unix_ns = self._unix_clock_ns()
            countdown_monotonic_ns = self._monotonic_clock_ns()
            output = RecordingOutput(
                width=source.width,
                height=source.height,
                fps=float(OUTPUT_FPS),
            )
            try:
                writer = await asyncio.to_thread(
                    CaptureRecordingWriter.create,
                    self._root,
                    recording_id=recording_id,
                    video_profile=output,
                )
            except Exception as error:
                raise RecordingFailureError("recording workspace could not be created") from error

            self._recording_id = recording_id
            self._connection_session_id = source.connection_session_id
            self._camera_start_generation = source.camera_start_generation
            self._countdown_started_at_unix_ns = countdown_unix_ns
            self._countdown_started_at_monotonic_ns = countdown_monotonic_ns
            self._recording_starts_at_unix_ms = (
                countdown_unix_ns // 1_000_000 + round(COUNTDOWN_SECONDS * 1000)
            )
            self._recording_started_at_unix_ms = None
            self._recording_started_at_monotonic_ns = None
            self._recording_duration_ms = 0
            self._imu_sample_count = 0
            self._telemetry_queue_overflow_count = 0
            self._output = output
            self._writer = writer
            self._matches_by_pts.clear()
            self._metadata_match_available.clear()
            self._telemetry_error = None
            self._first_imu_timestamp_ns.clear()
            self._latest_imu_timestamp_ns.clear()
            self._imu_sample_available.clear()
            self._telemetry_queue = asyncio.Queue(maxsize=self._telemetry_queue_size)
            self._telemetry_task = asyncio.create_task(self._write_telemetry(writer))
            self._accepting_imu = True
            self._state = RecordingState.COUNTDOWN
            self._detail = "recording starts after the countdown"
            self._countdown_task = asyncio.create_task(
                self._complete_countdown(
                    recording_id,
                    source.connection_session_id,
                    source.camera_start_generation,
                )
            )
            return self._status_locked()

    async def stop(self) -> RecordingStatus:
        async with self._command_lock:
            if self._state is RecordingState.COUNTDOWN:
                await self._cancel_countdown_locked()
                return self._status_locked()
            if self._state is not RecordingState.RECORDING:
                raise RecordingConflictError("there is no active recording to stop")
            self._state = RecordingState.FINALIZING
            self._detail = "finalizing recording"
            try:
                await self._finalize_locked()
            except Exception as error:
                await self._fail_locked(error)
                raise RecordingFailureError(self._detail) from error
            return self._status_locked()

    async def library(self) -> RecordingLibrary:
        return await asyncio.to_thread(self._scan_library)

    async def media_path(self, recording_id: str) -> Path | None:
        reader = await asyncio.to_thread(self._reader_or_none, recording_id)
        return None if reader is None else reader.video_path

    async def artifact_path(self, recording_id: str, artifact: str) -> Path | None:
        if artifact not in {"camera.csv", "imu.csv", "calibration.yaml"}:
            return None
        reader = await asyncio.to_thread(self._reader_or_none, recording_id)
        return None if reader is None else reader.directory / artifact

    async def reader(self, recording_id: str) -> CaptureRecordingReader:
        reader = await asyncio.to_thread(self._reader_or_none, recording_id)
        if reader is None:
            raise RecordingNotFoundError("recording not found or failed validation")
        return reader

    async def delete(self, recording_id: str) -> RecordingLibrary:
        async with self._command_lock:
            if not _ID_PATTERN.fullmatch(recording_id):
                raise RecordingNotFoundError("recording not found")
            if recording_id == self._recording_id:
                raise RecordingConflictError("active recording cannot be deleted")
            recording_directory = self._root / recording_id
            reader = await asyncio.to_thread(self._reader_or_none, recording_id)
            if reader is None or reader.directory != recording_directory:
                raise RecordingNotFoundError("recording not found or failed validation")
            tombstone = self._root / f".{recording_id}.deleting-{uuid.uuid4().hex}"
            try:
                os.replace(recording_directory, tombstone)
                await asyncio.to_thread(shutil.rmtree, tombstone)
            except Exception as error:
                raise RecordingFailureError("recording deletion is incomplete") from error
        return await self.library()

    async def close(self) -> None:
        async with self._command_lock:
            if self._state is RecordingState.COUNTDOWN:
                await self._cancel_countdown_locked()
            elif self._state is RecordingState.RECORDING:
                self._state = RecordingState.FINALIZING
                self._detail = "finalizing recording during shutdown"
                try:
                    await self._finalize_locked()
                except Exception as error:
                    await self._fail_locked(error)
                    LOGGER.exception("recording could not be finalized during shutdown")
            elif self._writer is not None:
                await self._close_incomplete_writer()

    async def on_connection_started(
        self,
        connection_session_id: str,
        device_session_id: str,
        observed_at_client_monotonic_ns: int,
    ) -> None:
        del connection_session_id, device_session_id, observed_at_client_monotonic_ns

    async def on_connection_state(
        self,
        connection_session_id: str,
        state: str,
        observed_at_client_monotonic_ns: int,
    ) -> None:
        del connection_session_id, state, observed_at_client_monotonic_ns

    async def on_imu_capabilities(
        self,
        connection_session_id: str,
        capabilities: ImuCapabilities,
        received_at_client_monotonic_ns: int,
    ) -> None:
        del connection_session_id, capabilities, received_at_client_monotonic_ns

    async def on_imu_sample(
        self,
        connection_session_id: str,
        sample: ImuSample,
        received_at_client_monotonic_ns: int,
    ) -> None:
        queue = self._telemetry_queue
        origin_ns = self._countdown_started_at_monotonic_ns
        if (
            queue is None
            or not self._accepting_imu
            or origin_ns is None
            or self._writer is None
            or self._state
            not in {
                RecordingState.COUNTDOWN,
                RecordingState.RECORDING,
                RecordingState.FINALIZING,
            }
            or connection_session_id != self._connection_session_id
        ):
            return
        row = StagedImuSample(
            row=RecordingImuRow(
                sensor_type=RecordingImuSensorType(sample.sensor_type.value),
                sequence=sample.sequence_number,
                timestamp_ns=sample.sensor_event_monotonic_ns,
                x=sample.values[0],
                y=sample.values[1],
                z=sample.values[2],
            ),
            received_at_client_monotonic_ns=received_at_client_monotonic_ns,
        )
        try:
            queue.put_nowait(row)
        except asyncio.QueueFull:
            self._telemetry_queue_overflow_count += 1
        else:
            self._imu_sample_count += 1
            self._first_imu_timestamp_ns.setdefault(
                row.row.sensor_type,
                row.row.timestamp_ns,
            )
            self._latest_imu_timestamp_ns[row.row.sensor_type] = row.row.timestamp_ns
            self._imu_sample_available.set()

    async def on_frame_metadata_match(
        self,
        connection_session_id: str,
        match: FrameMetadataMatch,
    ) -> None:
        if (
            self._writer is not None
            and connection_session_id == self._connection_session_id
            and self._state
            in {RecordingState.COUNTDOWN, RecordingState.RECORDING, RecordingState.FINALIZING}
        ):
            self._matches_by_pts[match.decoded_frame_pts] = match
            self._metadata_match_available.set()

    async def on_video_frame_metadata(
        self,
        connection_session_id: str,
        metadata: VideoFrameMetadata,
        received_at_client_monotonic_ns: int,
        camera_start_generation: int,
        ingest_status: str,
    ) -> None:
        del (
            connection_session_id,
            metadata,
            received_at_client_monotonic_ns,
            camera_start_generation,
            ingest_status,
        )

    async def _complete_countdown(
        self,
        recording_id: str,
        connection_session_id: str,
        camera_start_generation: int,
    ) -> None:
        try:
            await self._sleep(COUNTDOWN_SECONDS)
            async with self._command_lock:
                if (
                    self._state is not RecordingState.COUNTDOWN
                    or self._recording_id != recording_id
                ):
                    return
                source = await self._compatible_source()
                if (
                    source.connection_session_id != connection_session_id
                    or source.camera_start_generation != camera_start_generation
                ):
                    raise RecordingUnavailableError(
                        "Glass3 stream changed during the countdown"
                    )
                writer = self._require_writer()
                metadata_matched_track = MetadataMatchedVideoTrack(
                    source.source.subscribe(buffered=True),
                    self._wait_for_frame_metadata,
                )
                recorder = self._recorder_factory(
                    writer.video_path,
                    metadata_matched_track,
                    width=source.width,
                    height=source.height,
                    fps=OUTPUT_FPS,
                )
                self._metadata_matched_track = metadata_matched_track
                self._recorder = recorder
                try:
                    await recorder.start()
                except Exception:
                    with suppress(Exception):
                        await recorder.stop()
                    self._recorder = None
                    raise
                self._recording_started_at_unix_ms = self._unix_clock_ns() // 1_000_000
                self._recording_started_at_monotonic_ns = self._monotonic_clock_ns()
                self._state = RecordingState.RECORDING
                self._detail = ""
                self._countdown_task = None
                self._monitor_task = asyncio.create_task(self._monitor_recorder(recorder))
        except asyncio.CancelledError:
            return
        except Exception as error:
            LOGGER.exception("recording countdown failed")
            async with self._command_lock:
                if self._recording_id == recording_id:
                    await self._fail_locked(error)

    async def _monitor_recorder(self, recorder: RecordingWriter) -> None:
        failure: BaseException | None = None
        try:
            await recorder.wait()
        except asyncio.CancelledError:
            return
        except BaseException as error:
            failure = error
        async with self._command_lock:
            if self._recorder is not recorder:
                return
            self._state = RecordingState.FINALIZING
            self._detail = "video source ended; finalizing recording"
            try:
                if failure is not None:
                    raise failure
                await self._finalize_locked()
            except BaseException as error:
                await self._fail_locked(error)

    async def _finalize_locked(self) -> None:
        recorder, self._recorder = self._recorder, None
        monitor, self._monitor_task = self._monitor_task, None
        if monitor is not None and monitor is not asyncio.current_task():
            monitor.cancel()
        if recorder is None:
            raise CaptureRecordingError("recording has no active MP4 writer")
        LOGGER.info("recording finalization stage=video_stop_started")
        await recorder.stop()
        frame_records = tuple(recorder.frame_records)
        LOGGER.info(
            "recording finalization stage=video_stop_completed frames=%d",
            len(frame_records),
        )
        metadata_matched_track, self._metadata_matched_track = (
            self._metadata_matched_track,
            None,
        )
        LOGGER.info(
            "recording frame metadata gate encoded=%d metadata_skipped=%d",
            recorder.frames_received,
            0
            if metadata_matched_track is None
            else metadata_matched_track.skipped_frame_count,
        )
        if recorder.frames_received < 1 or len(frame_records) != recorder.frames_received:
            raise CaptureRecordingError("MP4 writer did not preserve every frame timestamp")
        rows = self._camera_frames(frame_records)
        writer = self._require_writer()
        await asyncio.to_thread(writer.stage_camera, rows)
        trimmed_frame_count = 0
        try:
            await self._wait_for_imu_tail_coverage(rows[-1].row.device_monotonic_ns)
        except _ImuTailCoverageTimeout as error:
            keep_count = self._recoverable_camera_prefix_length(rows)
            if keep_count < 1 or keep_count >= len(rows):
                raise
            trimmed_frame_count = len(rows) - keep_count
            trimmed_duration_ns = (
                rows[-1].row.device_monotonic_ns
                - rows[keep_count - 1].row.device_monotonic_ns
            )
            LOGGER.warning(
                "recording IMU tail stalled; trimming frames=%d duration_ms=%.3f cause=%s",
                trimmed_frame_count,
                trimmed_duration_ns / 1_000_000,
                error,
            )
            await recorder.trim_to_frame_count(keep_count)
            frame_records = tuple(recorder.frame_records)
            if len(frame_records) != keep_count:
                raise CaptureRecordingError(
                    "trimmed MP4 frame index does not match the recoverable camera prefix"
                ) from error
            rows = self._camera_frames(frame_records)
            await asyncio.to_thread(writer.stage_camera, rows)
        self._accepting_imu = False
        LOGGER.info(
            "recording finalization stage=telemetry_drain_started queued=%d",
            0 if self._telemetry_queue is None else self._telemetry_queue.qsize(),
        )
        await self._drain_telemetry()
        LOGGER.info("recording finalization stage=telemetry_drain_completed")
        if self._telemetry_queue_overflow_count:
            raise CaptureRecordingError("IMU CSV queue overflowed during recording")
        LOGGER.info(
            "recording finalization stage=protocol_write_started camera_rows=%d",
            len(rows),
        )
        reader = await asyncio.to_thread(
            writer.finalize,
            rows,
        )
        LOGGER.info("recording finalization stage=protocol_write_completed")
        self._recording_duration_ms = reader.summary().duration_ns // 1_000_000
        self._clear_active()
        self._state = RecordingState.READY
        self._detail = (
            f"recording finalized after trimming {trimmed_frame_count} uncovered frames"
            if trimmed_frame_count
            else "recording finalized"
        )

    async def _cancel_countdown_locked(self) -> None:
        task, self._countdown_task = self._countdown_task, None
        if task is not None:
            task.cancel()
        self._accepting_imu = False
        await self._drain_telemetry()
        await self._close_incomplete_writer()
        self._clear_active()
        self._state = RecordingState.READY
        self._detail = "countdown cancelled"

    async def _fail_locked(self, error: BaseException) -> None:
        LOGGER.error("recording failed: %s", error)
        self._accepting_imu = False
        self._metadata_matched_track = None
        recorder, self._recorder = self._recorder, None
        if recorder is not None:
            with suppress(Exception):
                await recorder.stop()
        with suppress(Exception):
            await self._drain_telemetry()
        with suppress(Exception):
            await self._close_incomplete_writer()
        self._state = RecordingState.ERROR
        self._detail = str(error)[:256] or type(error).__name__

    async def _write_telemetry(self, writer: CaptureRecordingWriter) -> None:
        queue = self._telemetry_queue
        assert queue is not None
        while True:
            row = await queue.get()
            try:
                await asyncio.to_thread(writer.append_imu, row)
            except BaseException as error:
                self._telemetry_error = error
            finally:
                queue.task_done()

    async def _drain_telemetry(self) -> None:
        queue = self._telemetry_queue
        task, self._telemetry_task = self._telemetry_task, None
        if queue is not None:
            if task is None and queue.qsize():
                raise CaptureRecordingError("IMU CSV writer stopped before queue drain")
            try:
                await asyncio.wait_for(
                    queue.join(),
                    timeout=TELEMETRY_DRAIN_TIMEOUT_SECONDS,
                )
            except TimeoutError as error:
                raise CaptureRecordingError(
                    f"IMU CSV queue did not drain within "
                    f"{TELEMETRY_DRAIN_TIMEOUT_SECONDS:.0f} seconds"
                ) from error
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._telemetry_queue = None
        if self._telemetry_error is not None:
            raise CaptureRecordingError("IMU CSV writer failed") from self._telemetry_error

    async def _wait_for_imu_tail_coverage(self, camera_timestamp_ns: int) -> None:
        loop = asyncio.get_running_loop()
        timeout_seconds = self._imu_tail_coverage_timeout_seconds
        deadline = loop.time() + timeout_seconds
        while not self._has_imu_tail_coverage(camera_timestamp_ns):
            self._imu_sample_available.clear()
            if self._has_imu_tail_coverage(camera_timestamp_ns):
                break
            remaining = deadline - loop.time()
            if remaining <= 0:
                missing = self._missing_imu_tail_coverage(camera_timestamp_ns)
                raise _ImuTailCoverageTimeout(
                    "IMU tail did not cover the final camera timestamp within "
                    f"{timeout_seconds:.1f} seconds: "
                    + ", ".join(missing)
                )
            try:
                await asyncio.wait_for(
                    self._imu_sample_available.wait(),
                    timeout=remaining,
                )
            except TimeoutError as error:
                missing = self._missing_imu_tail_coverage(camera_timestamp_ns)
                raise _ImuTailCoverageTimeout(
                    "IMU tail did not cover the final camera timestamp within "
                    f"{timeout_seconds:.1f} seconds: "
                    + ", ".join(missing)
                ) from error

    def _recoverable_camera_prefix_length(
        self,
        rows: Sequence[StagedCameraFrame],
    ) -> int:
        if not rows:
            return 0
        first_camera_timestamp_ns = rows[0].row.device_monotonic_ns
        if any(
            self._first_imu_timestamp_ns.get(sensor_type, first_camera_timestamp_ns + 1)
            > first_camera_timestamp_ns
            for sensor_type in RecordingImuSensorType
        ):
            return 0
        latest_timestamps = tuple(
            self._latest_imu_timestamp_ns.get(sensor_type)
            for sensor_type in RecordingImuSensorType
        )
        if any(timestamp is None for timestamp in latest_timestamps):
            return 0
        coverage_end_ns = min(
            timestamp for timestamp in latest_timestamps if timestamp is not None
        )
        return sum(
            row.row.device_monotonic_ns <= coverage_end_ns
            for row in rows
        )

    def _has_imu_tail_coverage(self, camera_timestamp_ns: int) -> bool:
        return not self._missing_imu_tail_coverage(camera_timestamp_ns)

    def _missing_imu_tail_coverage(self, camera_timestamp_ns: int) -> tuple[str, ...]:
        missing: list[str] = []
        for sensor_type in RecordingImuSensorType:
            timestamp_ns = self._latest_imu_timestamp_ns.get(sensor_type)
            if timestamp_ns is None or timestamp_ns < camera_timestamp_ns:
                missing.append(
                    f"{sensor_type.value}={timestamp_ns or 'missing'}<{camera_timestamp_ns}"
                )
        return tuple(missing)

    async def _close_incomplete_writer(self) -> None:
        writer, self._writer = self._writer, None
        if writer is not None:
            await asyncio.to_thread(writer.close_incomplete)

    async def _wait_for_frame_metadata(self, source_frame_pts: int) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + FRAME_METADATA_WAIT_SECONDS
        while not self._has_current_frame_metadata(source_frame_pts):
            self._metadata_match_available.clear()
            if self._has_current_frame_metadata(source_frame_pts):
                break
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            try:
                await asyncio.wait_for(
                    self._metadata_match_available.wait(),
                    timeout=remaining,
                )
            except TimeoutError:
                return False
        return True

    def _has_current_frame_metadata(self, source_frame_pts: int) -> bool:
        match = self._matches_by_pts.get(source_frame_pts)
        return (
            match is not None
            and match.metadata.camera_start_generation == self._camera_start_generation
        )

    def _camera_frames(
        self,
        records: Sequence[RecordedVideoFrame],
    ) -> tuple[StagedCameraFrame, ...]:
        rows: list[StagedCameraFrame] = []
        for record in records:
            match = (
                self._matches_by_pts.get(record.source_frame_pts)
                if record.source_frame_pts is not None
                else None
            )
            if (
                match is not None
                and match.metadata.camera_start_generation
                != self._camera_start_generation
            ):
                match = None
            if match is None:
                raise CaptureRecordingError(
                    f"encoded frame {record.frame_index} has no matching Glass3 metadata"
                )
            metadata = match.metadata
            if (
                (metadata.width, metadata.height)
                != (self._output.width, self._output.height)
                or metadata.rotation_degrees != 0
            ):
                raise CaptureRecordingError(
                    "camera dimensions or orientation changed during recording"
                )
            rows.append(
                StagedCameraFrame(
                    row=CameraFrameRow(
                        frame_idx=record.frame_index,
                        frame_id=metadata.frame_id,
                        rokid_timestamp_ns=metadata.captured_at_rokid_sdk_ms * 1_000_000,
                        device_monotonic_ns=metadata.received_at_elapsed_realtime_ns,
                    ),
                    mp4_pts=record.mp4_pts,
                    mp4_time_base_num=record.mp4_time_base_num,
                    mp4_time_base_den=record.mp4_time_base_den,
                    received_at_client_monotonic_ns=(record.received_at_client_perf_counter_ns),
                )
            )
        return tuple(rows)

    async def _compatible_source(self) -> WebRtcVideoRecordingSource:
        source = await self._source_provider()
        if source is None:
            raise RecordingUnavailableError("Glass3 video is not ready")
        if (source.width, source.height) != (640, 480):
            raise RecordingUnavailableError(
                "Glass3 recording requires the 640x480 capture profile; "
                f"received {source.width}x{source.height}"
            )
        if not _ID_PATTERN.fullmatch(source.connection_session_id):
            raise RecordingUnavailableError("Glass3 connection identifier is invalid")
        if source.camera_start_generation < 1:
            raise RecordingUnavailableError("Glass3 camera generation is invalid")
        self._output = RecordingOutput(
            width=source.width,
            height=source.height,
            fps=float(OUTPUT_FPS),
        )
        return source

    async def _refresh_availability_locked(self) -> None:
        try:
            await self._compatible_source()
        except RecordingUnavailableError as error:
            self._state = RecordingState.UNAVAILABLE
            self._detail = str(error)
        else:
            self._state = RecordingState.READY
            if self._detail != "recording finalized":
                self._detail = ""

    def _status_locked(self) -> RecordingStatus:
        duration_ms = self._recording_duration_ms
        if self._recording_started_at_monotonic_ns is not None and self._state in {
            RecordingState.RECORDING,
            RecordingState.FINALIZING,
        }:
            duration_ms = max(
                0,
                (self._monotonic_clock_ns() - self._recording_started_at_monotonic_ns)
                // 1_000_000,
            )
        return RecordingStatus(
            state=self._state,
            detail=self._detail,
            recording_id=self._recording_id,
            countdown_started_at_unix_ms=(
                None
                if self._countdown_started_at_unix_ns is None
                else self._countdown_started_at_unix_ns // 1_000_000
            ),
            recording_starts_at_unix_ms=self._recording_starts_at_unix_ms,
            recording_started_at_unix_ms=self._recording_started_at_unix_ms,
            recording_duration_ms=duration_ms,
            frame_count=(self._recorder.frames_received if self._recorder is not None else 0),
            imu_sample_count=self._imu_sample_count,
            telemetry_queue_overflow_count=self._telemetry_queue_overflow_count,
            output=self._output,
        )

    def _scan_library(self) -> RecordingLibrary:
        recordings = []
        for directory in self._root.iterdir():
            if not directory.is_dir() or not _ID_PATTERN.fullmatch(directory.name):
                continue
            try:
                recordings.append(CaptureRecordingReader.open(directory).summary())
            except CaptureRecordingError:
                LOGGER.warning("skipping invalid recording %s", directory.name)
        recordings.sort(key=lambda item: item.recorded_at_unix_ns, reverse=True)
        return RecordingLibrary(recordings=recordings)

    def _reader_or_none(self, recording_id: str) -> CaptureRecordingReader | None:
        if not _ID_PATTERN.fullmatch(recording_id):
            return None
        directory = self._root / recording_id
        try:
            return CaptureRecordingReader.open(directory)
        except (CaptureRecordingError, FileNotFoundError, NotADirectoryError):
            return None

    def _require_writer(self) -> CaptureRecordingWriter:
        if self._writer is None:
            raise CaptureRecordingError("recording storage writer is missing")
        return self._writer

    def _clear_active(self) -> None:
        self._writer = None
        self._recording_id = None
        self._connection_session_id = None
        self._camera_start_generation = None
        self._countdown_started_at_unix_ns = None
        self._countdown_started_at_monotonic_ns = None
        self._recording_starts_at_unix_ms = None
        self._recording_started_at_unix_ms = None
        self._recording_started_at_monotonic_ns = None
        self._telemetry_queue = None
        self._telemetry_task = None
        self._matches_by_pts.clear()
        self._metadata_match_available.clear()
        self._metadata_matched_track = None

    def _recover_partial_directories(self) -> None:
        for recording_id in recover_completed_recordings(self._root):
            LOGGER.info("recovered completed recording %s", recording_id)
