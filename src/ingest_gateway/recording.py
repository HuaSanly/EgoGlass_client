from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from .adapters.mp4_recorder import PyAvH264Mp4Recorder, RecordedVideoFrame
from .adapters.webrtc import WebRtcVideoRecordingSource
from .capture_session import (
    CachedConnection,
    CaptureSessionDatabase,
    CaptureSessionWriter,
    read_capture_quality,
)
from .recording_models import (
    CaptureQualityCheck,
    CaptureQualityCounts,
    CaptureQualityIssue,
    CaptureSessionClip,
    CaptureSessionLifecycle,
    CaptureSessionManifest,
    CaptureSessionQuality,
    CaptureSessionQualityReport,
    CaptureSessionState,
    CaptureSessionTimeOrigin,
    CaptureVideoProfile,
    RecordingClip,
    RecordingLibrary,
    RecordingSession,
    RecordingState,
    RecordingStatus,
)
from .webrtc_matcher import FrameMetadataMatch
from .webrtc_models import ImuCapabilities, ImuSample, VideoFrameMetadata

COUNTDOWN_SECONDS = 3.0
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
OUTPUT_FPS = 30
_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_TERMINAL_CONNECTION_STATES = {"closed", "disconnected", "failed", "replaced"}
LOGGER = logging.getLogger(__name__)


class RecordingUnavailableError(RuntimeError):
    """Raised when there is no compatible live Glass3 source."""


class RecordingConflictError(RuntimeError):
    """Raised when a command conflicts with the current recording state."""


class RecordingFailureError(RuntimeError):
    """Raised when a recording or collection session cannot be finalized."""


class RecordingClipNotFoundError(RuntimeError):
    """Raised when a completed recording clip does not exist."""


class RecordingSessionNotFoundError(RuntimeError):
    """Raised when a recording session does not exist."""


class RecordingWriter(Protocol):
    @property
    def frames_received(self) -> int: ...

    @property
    def frame_records(self) -> Sequence[RecordedVideoFrame]: ...

    async def start(self) -> None: ...

    async def wait(self) -> None: ...

    async def stop(self) -> None: ...


RecordingSourceProvider = Callable[[], Awaitable[WebRtcVideoRecordingSource | None]]
RecordingWriterFactory = Callable[[Path, object], RecordingWriter]


class RecordingRuntime:
    """Own dataset-grade collection sessions, telemetry, and MP4 clips."""

    def __init__(
        self,
        root: Path,
        source_provider: RecordingSourceProvider,
        *,
        recorder_factory: RecordingWriterFactory = PyAvH264Mp4Recorder,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        unix_clock_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
        monotonic_clock_ns: Callable[[], int] = time.perf_counter_ns,
        session_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
        telemetry_queue_size: int = 32_768,
    ) -> None:
        self._root = root.expanduser().resolve()
        self._source_provider = source_provider
        self._recorder_factory = recorder_factory
        self._sleep = sleep
        self._unix_clock_ms = unix_clock_ms
        self._monotonic_clock_ns = monotonic_clock_ns
        self._session_id_factory = session_id_factory
        self._telemetry_queue_size = telemetry_queue_size
        self._command_lock = asyncio.Lock()
        self._countdown_task: asyncio.Task[None] | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._recorder: RecordingWriter | None = None
        self._partial_path: Path | None = None
        self._state = RecordingState.UNAVAILABLE
        self._detail = "Glass3 1280x720 video is not ready"
        self._session_id: str | None = None
        self._session_manifest: CaptureSessionManifest | None = None
        self._session_writer: CaptureSessionWriter | None = None
        self._webrtc_session_id: str | None = None
        self._clip_camera_start_generation: int | None = None
        self._clip_id: str | None = None
        self._countdown_started_at_unix_ms: int | None = None
        self._recording_starts_at_unix_ms: int | None = None
        self._recording_started_at_unix_ms: int | None = None
        self._recording_started_at_monotonic_ns: int | None = None
        self._recording_duration_ms = 0
        self._connections: dict[str, CachedConnection] = {}
        self._latest_capabilities: dict[str, tuple[ImuCapabilities, int]] = {}
        self._recover_stale_sessions()

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
            if self._session_id is None:
                self._create_session_locked()
            manifest = self._require_active_manifest()
            now_ms = self._unix_clock_ms()
            clip_id = uuid.uuid4().hex
            manifest = manifest.model_copy(
                update={
                    "clips": [
                        *manifest.clips,
                        CaptureSessionClip(
                            clip_id=clip_id,
                            state="preparing",
                            relative_media_path=f"media/{clip_id}.mp4",
                            video_profile=CaptureVideoProfile(
                                width=OUTPUT_WIDTH,
                                height=OUTPUT_HEIGHT,
                                nominal_fps=OUTPUT_FPS,
                            ),
                        ),
                    ]
                }
            )
            self._write_capture_manifest(manifest)
            self._session_manifest = manifest
            self._webrtc_session_id = source.connection_session_id
            self._clip_camera_start_generation = source.camera_start_generation
            self._clip_id = clip_id
            self._countdown_started_at_unix_ms = now_ms
            self._recording_starts_at_unix_ms = now_ms + round(COUNTDOWN_SECONDS * 1000)
            self._recording_started_at_unix_ms = None
            self._recording_started_at_monotonic_ns = None
            self._recording_duration_ms = 0
            self._state = RecordingState.COUNTDOWN
            self._detail = "recording starts after the server countdown"
            self._record_event(
                "recording_countdown_started",
                clip_id=clip_id,
                details={"countdown_seconds": COUNTDOWN_SECONDS},
            )
            self._countdown_task = asyncio.create_task(
                self._complete_countdown(
                    self._session_id,
                    source.connection_session_id,
                    source.camera_start_generation,
                    clip_id,
                )
            )
            return self._status_locked()

    async def stop(self) -> RecordingStatus:
        async with self._command_lock:
            if self._state is RecordingState.COUNTDOWN:
                self._cancel_countdown_locked()
                return self._status_locked()
            if self._state is not RecordingState.RECORDING:
                raise RecordingConflictError("there is no active recording to stop")
            self._state = RecordingState.FINALIZING
            self._detail = "finalizing MP4"
            try:
                await self._finalize_clip_locked()
            except Exception as error:
                self._fail_locked(error)
                raise RecordingFailureError(self._detail or "recording failed") from error
            return self._status_locked()

    async def session_command(self, action: str) -> RecordingStatus:
        async with self._command_lock:
            if action == "new" and self._state in {
                RecordingState.COUNTDOWN,
                RecordingState.RECORDING,
                RecordingState.FINALIZING,
            }:
                raise RecordingConflictError(
                    "stop the active clip before starting a new collection session"
                )
            reason = "manual_new_session" if action == "new" else "client_shutdown"
            try:
                await self._finish_session_locked(reason, allow_active_clip=action == "finalize")
            except Exception as error:
                self._fail_locked(error)
                raise RecordingFailureError("collection session finalization failed") from error
            await self._refresh_availability_locked()
            return self._status_locked()

    async def library(self) -> RecordingLibrary:
        """Scan the recording library without blocking the media event loop."""

        async with self._command_lock:
            return await self._scan_library_in_worker()

    async def _scan_library_in_worker(self) -> RecordingLibrary:
        return await asyncio.to_thread(self._scan_library)

    def _scan_library(self) -> RecordingLibrary:
        sessions: list[RecordingSession] = []
        if not self._root.exists():
            return RecordingLibrary(sessions=[])
        for path in self._root.glob("*/session.json"):
            if not _ID_PATTERN.fullmatch(path.parent.name):
                continue
            capture = self._read_capture_manifest(path)
            if capture is not None:
                sessions.append(self._capture_library_session(capture))
                continue
            legacy = self._read_legacy_manifest(path)
            if legacy is None:
                continue
            complete_clips = [
                clip
                for clip in legacy.clips
                if self._legacy_completed_path(legacy.session_id, clip.clip_id).is_file()
                and self._legacy_completed_path(legacy.session_id, clip.clip_id).stat().st_size
                == clip.file_size_bytes
            ]
            if complete_clips:
                sessions.append(
                    legacy.model_copy(
                        update={
                            "clips": sorted(
                                complete_clips,
                                key=lambda clip: clip.recorded_at_unix_ms,
                                reverse=True,
                            )
                        }
                    )
                )
        sessions.sort(key=lambda session: session.started_at_unix_ms, reverse=True)
        return RecordingLibrary(sessions=sessions)

    async def media_path(self, session_id: str, clip_id: str) -> Path | None:
        if not _ID_PATTERN.fullmatch(session_id) or not _ID_PATTERN.fullmatch(clip_id):
            return None
        manifest_path = self._session_directory(session_id) / "session.json"
        capture = self._read_capture_manifest(manifest_path)
        if capture is not None:
            clip = next(
                (
                    item
                    for item in capture.clips
                    if item.clip_id == clip_id and item.state in {"complete", "incomplete"}
                ),
                None,
            )
            path = self._completed_path(session_id, clip_id)
            if clip is None or not path.is_file() or clip.sha256 is None:
                return None
            return path if _sha256(path) == clip.sha256 else None
        legacy = self._read_legacy_manifest(manifest_path)
        if legacy is None:
            return None
        clip = next((item for item in legacy.clips if item.clip_id == clip_id), None)
        path = self._legacy_completed_path(session_id, clip_id)
        if clip is None or not path.is_file() or path.stat().st_size != clip.file_size_bytes:
            return None
        return path

    async def delete_clip(self, session_id: str, clip_id: str) -> RecordingLibrary:
        async with self._command_lock:
            if not _ID_PATTERN.fullmatch(session_id) or not _ID_PATTERN.fullmatch(clip_id):
                raise RecordingClipNotFoundError("recording clip not found")
            if session_id == self._session_id:
                raise RecordingConflictError("active collection session cannot be edited")
            session_directory = self._session_directory(session_id)
            manifest_path = session_directory / "session.json"
            capture = self._read_capture_manifest(manifest_path)
            if capture is not None:
                clip = next(
                    (item for item in capture.clips if item.clip_id == clip_id),
                    None,
                )
                completed_path = self._completed_path(session_id, clip_id)
                if clip is None or not completed_path.is_file():
                    raise RecordingClipNotFoundError("recording clip not found")
                updated = capture.model_copy(
                    update={"clips": [item for item in capture.clips if item.clip_id != clip_id]}
                )
                await self._delete_clip_file_and_manifest(
                    completed_path,
                    updated,
                    capture,
                )
                return await self._scan_library_in_worker()

            legacy = self._read_legacy_manifest(manifest_path)
            if legacy is None:
                raise RecordingClipNotFoundError("recording clip not found")
            clip = next((item for item in legacy.clips if item.clip_id == clip_id), None)
            completed_path = self._legacy_completed_path(session_id, clip_id)
            if clip is None or not completed_path.is_file():
                raise RecordingClipNotFoundError("recording clip not found")
            updated_legacy = legacy.model_copy(
                update={"clips": [item for item in legacy.clips if item.clip_id != clip_id]}
            )
            await self._delete_legacy_clip(completed_path, updated_legacy, legacy)
            return await self._scan_library_in_worker()

    async def delete_session(self, session_id: str) -> RecordingLibrary:
        async with self._command_lock:
            if not _ID_PATTERN.fullmatch(session_id):
                raise RecordingSessionNotFoundError("recording session not found")
            if session_id == self._session_id:
                raise RecordingConflictError("active collection session cannot be deleted")
            session_directory = self._session_directory(session_id)
            if not (session_directory / "session.json").is_file():
                raise RecordingSessionNotFoundError("recording session not found")
            deleting = self._root / f"{session_id}.deleting-{uuid.uuid4().hex}"
            try:
                os.replace(session_directory, deleting)
                await asyncio.to_thread(shutil.rmtree, deleting)
            except Exception as error:
                raise RecordingFailureError(
                    "recording session deletion is incomplete; its tombstone is unpublished"
                ) from error
            return await self._scan_library_in_worker()

    async def rename_session(self, session_id: str, display_name: str) -> RecordingLibrary:
        async with self._command_lock:
            if not _ID_PATTERN.fullmatch(session_id):
                raise RecordingSessionNotFoundError("recording session not found")
            manifest_path = self._session_directory(session_id) / "session.json"
            capture = self._read_capture_manifest(manifest_path)
            if capture is not None:
                updated = capture.model_copy(
                    update={
                        "display_name": display_name,
                        "display_name_source": "operator",
                    }
                )
                self._write_capture_manifest(updated)
                if session_id == self._session_id:
                    self._session_manifest = updated
                return await self._scan_library_in_worker()
            legacy = self._read_legacy_manifest(manifest_path)
            if legacy is None:
                raise RecordingSessionNotFoundError("recording session not found")
            self._write_legacy_manifest(legacy.model_copy(update={"display_name": display_name}))
            return await self._scan_library_in_worker()

    async def close(self) -> None:
        async with self._command_lock:
            try:
                await self._finish_session_locked("client_shutdown", allow_active_clip=True)
            except Exception:
                LOGGER.exception("collection session could not be finalized during shutdown")

    async def on_connection_started(
        self,
        connection_session_id: str,
        device_session_id: str,
        observed_at_client_monotonic_ns: int,
    ) -> None:
        connection = CachedConnection(
            connection_session_id=connection_session_id,
            device_session_id=device_session_id,
            state="negotiating",
            observed_at_client_monotonic_ns=observed_at_client_monotonic_ns,
        )
        had_connection = bool(self._connections)
        self._connections[connection_session_id] = connection
        writer = self._session_writer
        if writer is not None:
            writer.enqueue("begin_connection", connection, observed_at_client_monotonic_ns)
            if had_connection:
                self._record_event(
                    "stream_reconnected",
                    elapsed_realtime_ns=None,
                    details={"connection_session_id": connection_session_id},
                )

    async def on_connection_state(
        self,
        connection_session_id: str,
        state: str,
        observed_at_client_monotonic_ns: int,
    ) -> None:
        previous = self._connections.get(connection_session_id)
        if previous is not None:
            self._connections[connection_session_id] = CachedConnection(
                connection_session_id=previous.connection_session_id,
                device_session_id=previous.device_session_id,
                state=state,
                observed_at_client_monotonic_ns=observed_at_client_monotonic_ns,
            )
        writer = self._session_writer
        if writer is not None and state in _TERMINAL_CONNECTION_STATES:
            writer.enqueue(
                "end_connection",
                connection_session_id,
                observed_at_client_monotonic_ns,
                state,
            )
            self._record_event(
                "stream_disconnected",
                elapsed_realtime_ns=None,
                details={
                    "connection_session_id": connection_session_id,
                    "state": state,
                },
            )

    async def on_imu_capabilities(
        self,
        connection_session_id: str,
        capabilities: ImuCapabilities,
        received_at_client_monotonic_ns: int,
    ) -> None:
        self._latest_capabilities[connection_session_id] = (
            capabilities,
            received_at_client_monotonic_ns,
        )
        writer = self._session_writer
        if writer is not None:
            writer.enqueue(
                "record_capabilities",
                connection_session_id,
                capabilities,
                received_at_client_monotonic_ns,
            )

    async def on_imu_sample(
        self,
        connection_session_id: str,
        sample: ImuSample,
        received_at_client_monotonic_ns: int,
    ) -> None:
        writer = self._session_writer
        if writer is not None:
            writer.enqueue(
                "record_imu_sample",
                connection_session_id,
                sample,
                received_at_client_monotonic_ns,
            )

    async def on_frame_metadata_match(
        self,
        connection_session_id: str,
        match: FrameMetadataMatch,
    ) -> None:
        writer = self._session_writer
        if writer is not None:
            writer.enqueue("record_frame_match", connection_session_id, match)

    async def on_video_frame_metadata(
        self,
        connection_session_id: str,
        metadata: VideoFrameMetadata,
        received_at_client_monotonic_ns: int,
        camera_start_generation: int,
        ingest_status: str,
    ) -> None:
        writer = self._session_writer
        if writer is not None:
            writer.enqueue(
                "record_video_frame_metadata",
                connection_session_id,
                metadata,
                received_at_client_monotonic_ns,
                camera_start_generation,
                ingest_status,
            )

    def _create_session_locked(self) -> None:
        session_id = self._session_id_factory()
        if not _ID_PATTERN.fullmatch(session_id):
            raise RuntimeError("capture session identifier is invalid")
        started_at_unix_ns = self._unix_clock_ms() * 1_000_000
        session_directory = self._session_directory(session_id)
        for relative in ("media", "telemetry", "annotations", "derived"):
            (session_directory / relative).mkdir(parents=True, exist_ok=True)
        display_name = datetime.fromtimestamp(started_at_unix_ns / 1_000_000_000).strftime(
            "%Y-%m-%d %H-%M-%S"
        )
        manifest = CaptureSessionManifest(
            session_id=session_id,
            display_name=display_name,
            display_name_source="timestamp_default",
            lifecycle=CaptureSessionLifecycle(
                state=CaptureSessionState.ACTIVE,
                started_at_unix_ns=started_at_unix_ns,
            ),
            session_time_origin=CaptureSessionTimeOrigin(),
            clips=[],
        )
        self._write_capture_manifest(manifest)
        database = CaptureSessionDatabase(session_id, self._telemetry_path(session_id))
        writer = CaptureSessionWriter(
            database,
            max_queue_size=self._telemetry_queue_size,
        )
        self._session_id = session_id
        self._session_manifest = manifest
        self._session_writer = writer
        for connection in self._connections.values():
            if connection.state not in _TERMINAL_CONNECTION_STATES:
                writer.enqueue(
                    "begin_connection",
                    connection,
                    self._monotonic_clock_ns(),
                )
        for connection_id, (capabilities, received_at_ns) in self._latest_capabilities.items():
            writer.enqueue(
                "record_capabilities",
                connection_id,
                capabilities,
                received_at_ns,
            )
        self._record_event(
            "session_started_automatically",
            details={"start_reason": "first_recording_request"},
        )

    async def _complete_countdown(
        self,
        session_id: str,
        webrtc_session_id: str,
        camera_start_generation: int,
        clip_id: str,
    ) -> None:
        try:
            await self._sleep(COUNTDOWN_SECONDS)
            async with self._command_lock:
                if (
                    self._state is not RecordingState.COUNTDOWN
                    or self._session_id != session_id
                    or self._webrtc_session_id != webrtc_session_id
                    or self._clip_id != clip_id
                ):
                    return
                source = await self._compatible_source()
                if (
                    source.connection_session_id != webrtc_session_id
                    or source.camera_start_generation != camera_start_generation
                ):
                    raise RecordingUnavailableError(
                        "Glass3 WebRTC session changed during the countdown"
                    )
                track = source.source.subscribe(buffered=True)
                partial_path = self._partial_file_path(session_id, clip_id)
                recorder = self._recorder_factory(partial_path, track)
                self._partial_path = partial_path
                self._recorder = recorder
                try:
                    await recorder.start()
                except Exception:
                    with suppress(Exception):
                        await recorder.stop()
                    raise
                self._recording_started_at_unix_ms = self._unix_clock_ms()
                self._recording_started_at_monotonic_ns = self._monotonic_clock_ns()
                self._state = RecordingState.RECORDING
                self._detail = ""
                self._countdown_task = None
                self._update_capture_clip(clip_id, state="recording")
                self._record_event("clip_recording_started", clip_id=clip_id)
                self._monitor_task = asyncio.create_task(self._monitor_recorder(recorder))
        except asyncio.CancelledError:
            return
        except Exception as error:
            LOGGER.exception("recording countdown failed")
            async with self._command_lock:
                if self._session_id == session_id and self._clip_id == clip_id:
                    self._countdown_task = None
                    self._update_capture_clip(clip_id, state="incomplete")
                    self._fail_locked(error)

    async def _monitor_recorder(self, recorder: RecordingWriter) -> None:
        failure: Exception | None = None
        try:
            await recorder.wait()
        except asyncio.CancelledError:
            return
        except Exception as error:
            LOGGER.exception("recording writer failed")
            failure = error
        async with self._command_lock:
            if self._recorder is not recorder:
                return
            self._state = RecordingState.FINALIZING
            self._detail = "source ended; finalizing MP4"
            try:
                if failure is not None:
                    await self._discard_recorder_locked()
                    raise failure
                await self._finalize_clip_locked()
            except Exception as error:
                self._fail_locked(error)

    async def _finalize_clip_locked(self) -> None:
        recorder, self._recorder = self._recorder, None
        partial_path, self._partial_path = self._partial_path, None
        monitor, self._monitor_task = self._monitor_task, None
        if monitor is not None and monitor is not asyncio.current_task():
            monitor.cancel()
        if recorder is None or partial_path is None:
            raise RuntimeError("recording finalization has no active recorder")
        try:
            await recorder.stop()
        except Exception:
            partial_path.unlink(missing_ok=True)
            raise
        if recorder.frames_received <= 0:
            partial_path.unlink(missing_ok=True)
            raise RuntimeError("recorder received no video frames")
        frame_records = tuple(recorder.frame_records)
        if len(frame_records) != recorder.frames_received:
            partial_path.unlink(missing_ok=True)
            raise RuntimeError("recorder did not preserve exact MP4 timing for every frame")
        if not partial_path.is_file() or partial_path.stat().st_size <= 0:
            raise RuntimeError("recorder produced no playable MP4 data")
        if self._session_id is None or self._clip_id is None:
            raise RuntimeError("recording identifiers are missing")
        if self._webrtc_session_id is None:
            raise RuntimeError("recording WebRTC connection identifier is missing")
        if self._clip_camera_start_generation is None:
            raise RuntimeError("recording camera start generation is missing")
        if self._recording_started_at_monotonic_ns is None:
            raise RuntimeError("recording start timestamp is missing")
        ended_at_ms = self._unix_clock_ms()
        duration_ms = max(
            0,
            (self._monotonic_clock_ns() - self._recording_started_at_monotonic_ns) // 1_000_000,
        )
        completed_path = self._completed_path(self._session_id, self._clip_id)
        os.replace(partial_path, completed_path)
        writer = self._require_writer()
        writer.enqueue(
            "record_clip_frames",
            self._clip_id,
            self._webrtc_session_id,
            self._clip_camera_start_generation,
            frame_records,
            recorder.frames_received,
        )
        self._record_event(
            "clip_recording_stopped",
            clip_id=self._clip_id,
            details={"ended_at_unix_ms": ended_at_ms},
        )
        try:
            await writer.flush()
            sha256 = await asyncio.to_thread(_sha256, completed_path)
            self._update_capture_clip(
                self._clip_id,
                state="incomplete",
                frame_count=recorder.frames_received,
                sha256=sha256,
            )
        except Exception:
            completed_path.unlink(missing_ok=True)
            self._update_capture_clip(self._clip_id, state="incomplete")
            raise
        self._recording_duration_ms = duration_ms
        self._state = RecordingState.READY
        self._detail = "recording finalized"

    async def _finish_session_locked(self, reason: str, *, allow_active_clip: bool) -> None:
        if self._session_id is None:
            return
        if self._state is RecordingState.COUNTDOWN:
            if not allow_active_clip:
                raise RecordingConflictError("recording countdown is active")
            self._cancel_countdown_locked()
        elif self._state is RecordingState.RECORDING:
            if not allow_active_clip:
                raise RecordingConflictError("recording clip is active")
            self._state = RecordingState.FINALIZING
            self._detail = "finalizing MP4 before collection shutdown"
            await self._finalize_clip_locked()
        manifest = self._require_active_manifest()
        finalizing = manifest.model_copy(
            update={
                "lifecycle": manifest.lifecycle.model_copy(
                    update={"state": CaptureSessionState.FINALIZING}
                )
            }
        )
        self._write_capture_manifest(finalizing)
        self._session_manifest = finalizing
        event_type = "manual_new_session" if reason == "manual_new_session" else "client_shutdown"
        self._record_event(event_type, details={"end_reason": reason})
        writer, self._session_writer = self._require_writer(), None
        result = await writer.finalize()
        clips = [self._finalize_capture_clip(clip) for clip in finalizing.clips]
        complete = result.quality.telemetry_queue_overflow_count == 0 and all(
            clip.state != "incomplete" for clip in clips
        )
        state = CaptureSessionState.COMPLETE if complete else CaptureSessionState.INCOMPLETE
        ended_at_unix_ns = self._unix_clock_ms() * 1_000_000
        completed_manifest = finalizing.model_copy(
            update={
                "lifecycle": finalizing.lifecycle.model_copy(
                    update={
                        "state": state,
                        "ended_at_unix_ns": ended_at_unix_ns,
                        "end_reason": reason,
                    }
                ),
                "session_time_origin": CaptureSessionTimeOrigin(),
                "clips": clips,
            }
        )
        report = self._quality_report(
            completed_manifest,
            result.quality,
            recoverable=state is CaptureSessionState.INCOMPLETE,
            recovered=False,
        )
        self._write_quality_report(completed_manifest.session_id, report)
        self._write_capture_manifest(completed_manifest)
        self._session_id = None
        self._session_manifest = None
        self._webrtc_session_id = None
        self._clip_camera_start_generation = None
        self._clip_id = None
        self._countdown_started_at_unix_ms = None
        self._recording_starts_at_unix_ms = None
        self._recording_started_at_unix_ms = None
        self._recording_started_at_monotonic_ns = None
        self._state = RecordingState.READY
        self._detail = "collection session finalized"

    def _cancel_countdown_locked(self) -> None:
        if self._countdown_task is not None:
            self._countdown_task.cancel()
            self._countdown_task = None
        if self._clip_id is not None:
            self._update_capture_clip(self._clip_id, state="cancelled")
        self._clip_id = None
        self._webrtc_session_id = None
        self._clip_camera_start_generation = None
        self._countdown_started_at_unix_ms = None
        self._recording_starts_at_unix_ms = None
        self._state = RecordingState.READY
        self._detail = "countdown cancelled; session IMU capture remains active"

    async def _discard_recorder_locked(self) -> None:
        recorder, self._recorder = self._recorder, None
        partial_path, self._partial_path = self._partial_path, None
        monitor, self._monitor_task = self._monitor_task, None
        if monitor is not None and monitor is not asyncio.current_task():
            monitor.cancel()
        if recorder is not None:
            with suppress(Exception):
                await recorder.stop()
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)

    async def _compatible_source(self) -> WebRtcVideoRecordingSource:
        source = await self._source_provider()
        if source is None:
            raise RecordingUnavailableError("Glass3 video is not ready")
        if (source.width, source.height) != (OUTPUT_WIDTH, OUTPUT_HEIGHT):
            raise RecordingUnavailableError(
                "Glass3 video must be 1280x720 before recording can start"
            )
        if not _ID_PATTERN.fullmatch(source.connection_session_id):
            raise RecordingUnavailableError("Glass3 session identifier is invalid")
        if source.camera_start_generation < 1:
            raise RecordingUnavailableError("Glass3 camera start generation is invalid")
        return source

    async def _refresh_availability_locked(self) -> None:
        try:
            await self._compatible_source()
        except RecordingUnavailableError as error:
            self._state = RecordingState.UNAVAILABLE
            self._detail = str(error)
            return
        self._state = RecordingState.READY
        if self._session_id is not None:
            self._detail = "collection session active; IMU persistence continues"
        else:
            self._detail = ""

    def _status_locked(self) -> RecordingStatus:
        duration_ms = self._recording_duration_ms
        if self._recording_started_at_monotonic_ns is not None and self._state in {
            RecordingState.RECORDING,
            RecordingState.FINALIZING,
        }:
            duration_ms = max(
                0,
                (self._monotonic_clock_ns() - self._recording_started_at_monotonic_ns) // 1_000_000,
            )
        return RecordingStatus(
            state=self._state,
            detail=self._detail,
            session_id=self._session_id,
            session_state=(
                self._session_manifest.lifecycle.state
                if self._session_manifest is not None
                else None
            ),
            clip_id=self._clip_id,
            countdown_started_at_unix_ms=self._countdown_started_at_unix_ms,
            recording_starts_at_unix_ms=self._recording_starts_at_unix_ms,
            recording_started_at_unix_ms=self._recording_started_at_unix_ms,
            recording_duration_ms=duration_ms,
        )

    def _record_event(
        self,
        event_type: str,
        *,
        clip_id: str | None = None,
        elapsed_realtime_ns: int | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        writer = self._session_writer
        if writer is not None:
            writer.enqueue(
                "record_event",
                uuid.uuid4().hex,
                event_type,
                self._monotonic_clock_ns(),
                elapsed_realtime_ns,
                clip_id,
                details or {},
            )

    def _update_capture_clip(self, clip_id: str, **updates: object) -> None:
        manifest = self._require_active_manifest()
        found = False
        clips: list[CaptureSessionClip] = []
        for clip in manifest.clips:
            if clip.clip_id == clip_id:
                clips.append(clip.model_copy(update=updates))
                found = True
            else:
                clips.append(clip)
        if not found:
            raise RuntimeError("capture clip is missing from session manifest")
        updated = manifest.model_copy(update={"clips": clips})
        self._write_capture_manifest(updated)
        self._session_manifest = updated

    @staticmethod
    def _finalize_capture_clip(
        clip: CaptureSessionClip,
    ) -> CaptureSessionClip:
        return CaptureSessionClip.model_validate(
            {
                **clip.model_dump(mode="python"),
                "state": (
                    "complete"
                    if clip.frame_count is not None
                    and clip.frame_count > 0
                    and clip.sha256 is not None
                    else "incomplete"
                ),
            }
        )

    def _capture_library_session(self, manifest: CaptureSessionManifest) -> RecordingSession:
        clips: list[RecordingClip] = []
        for clip in manifest.clips:
            path = self._completed_path(manifest.session_id, clip.clip_id)
            if (
                clip.state not in {"complete", "incomplete"}
                or clip.sha256 is None
                or not path.is_file()
            ):
                continue
            if _sha256(path) != clip.sha256:
                continue
            started_ns = clip.started_at_session_time_ns or 0
            ended_ns = clip.ended_at_session_time_ns
            duration_ms = (
                round((ended_ns - started_ns) / 1_000_000)
                if ended_ns is not None and ended_ns >= started_ns
                else round((clip.frame_count or 0) * 1000 / OUTPUT_FPS)
            )
            recorded_at_ms = manifest.lifecycle.started_at_unix_ns // 1_000_000
            recorded_at_ms += started_ns // 1_000_000
            clips.append(
                RecordingClip(
                    clip_id=clip.clip_id,
                    recorded_at_unix_ms=recorded_at_ms,
                    ended_at_unix_ms=recorded_at_ms + duration_ms,
                    duration_ms=duration_ms,
                    width=clip.video_profile.width,
                    height=clip.video_profile.height,
                    fps=30,
                    file_size_bytes=path.stat().st_size,
                    frame_count=clip.frame_count or 0,
                    media_url=(f"/api/v1/recordings/media/{manifest.session_id}/{clip.clip_id}"),
                )
            )
        quality = read_capture_quality(self._telemetry_path(manifest.session_id))
        return RecordingSession(
            session_id=manifest.session_id,
            started_at_unix_ms=manifest.lifecycle.started_at_unix_ns // 1_000_000,
            display_name=manifest.display_name,
            state=manifest.lifecycle.state,
            ended_at_unix_ms=(
                None
                if manifest.lifecycle.ended_at_unix_ns is None
                else manifest.lifecycle.ended_at_unix_ns // 1_000_000
            ),
            recoverable=manifest.lifecycle.state is CaptureSessionState.INCOMPLETE,
            telemetry_database="telemetry/telemetry.sqlite",
            quality=quality,
            clips=sorted(clips, key=lambda item: item.recorded_at_unix_ms, reverse=True),
        )

    def _quality_report(
        self,
        manifest: CaptureSessionManifest,
        quality: CaptureSessionQuality,
        *,
        recoverable: bool,
        recovered: bool,
    ) -> CaptureSessionQualityReport:
        complete_clips = sum(clip.state == "complete" for clip in manifest.clips)
        incomplete_clips = sum(clip.state == "incomplete" for clip in manifest.clips)
        missing_imu = quality.accelerometer_sample_count == 0 or quality.gyroscope_sample_count == 0
        provenance = manifest.provenance
        missing_provenance = any(
            value is None
            for value in (
                provenance.device.device_id,
                provenance.device.firmware_version,
                provenance.software.glasses_app_version,
                provenance.source_revisions.superproject_commit,
                provenance.source_revisions.glasses_commit,
                provenance.source_revisions.client_commit,
            )
        )
        failed = (
            manifest.lifecycle.state is CaptureSessionState.INCOMPLETE
            or missing_imu
            or quality.telemetry_queue_overflow_count > 0
            or incomplete_clips > 0
            or missing_provenance
        )
        status = (
            "incomplete"
            if manifest.lifecycle.state is CaptureSessionState.INCOMPLETE
            else ("fail" if failed else "pass")
        )
        issues: list[CaptureQualityIssue] = [
            CaptureQualityIssue(
                issue_id="perception_processing_required",
                severity="info",
                message="Perception processing is required before multimodal training export",
                recoverable=True,
                evidence="capture preserved source clocks and left alignment pending",
            )
        ]
        if missing_imu:
            issues.append(
                CaptureQualityIssue(
                    issue_id="missing_imu_stream",
                    severity="error",
                    message="Both accelerometer and gyroscope samples are required",
                    recoverable=False,
                    evidence=(
                        f"accelerometer={quality.accelerometer_sample_count}, "
                        f"gyroscope={quality.gyroscope_sample_count}"
                    ),
                )
            )
        if quality.telemetry_queue_overflow_count:
            issues.append(
                CaptureQualityIssue(
                    issue_id="telemetry_queue_overflow",
                    severity="error",
                    message="Telemetry records were dropped before SQLite persistence",
                    recoverable=False,
                    evidence=f"dropped={quality.telemetry_queue_overflow_count}",
                )
            )
        if missing_provenance:
            issues.append(
                CaptureQualityIssue(
                    issue_id="capture_provenance_incomplete",
                    severity="error",
                    message="Required device and source revision provenance is missing",
                    recoverable=True,
                    evidence=(
                        "device_id, firmware_version, glasses_app_version, "
                        "glasses_commit, and client_commit are required"
                    ),
                )
            )
        if recovered:
            issues.append(
                CaptureQualityIssue(
                    issue_id="unclean_shutdown_recovered",
                    severity="error",
                    message="The session was active when the previous process stopped",
                    recoverable=True,
                    evidence="startup recovery replayed SQLite WAL and preserved artifacts",
                )
            )
        coverage = quality.metadata_match_coverage
        checks = [
            CaptureQualityCheck(
                check_id="perception_alignment",
                status="not_evaluated",
                metric_value=None,
                threshold=None,
                unit=None,
                evidence="deferred to the versioned perception pipeline",
            ),
            CaptureQualityCheck(
                check_id="capture_provenance",
                status="fail" if missing_provenance else "pass",
                metric_value=0.0 if missing_provenance else 1.0,
                threshold=1.0,
                unit="boolean",
                evidence=("required device, firmware, application, and source revisions"),
            ),
            CaptureQualityCheck(
                check_id="video_metadata_coverage",
                status=(
                    "not_evaluated"
                    if coverage is None
                    else ("pass" if coverage >= 0.99 else "warn")
                ),
                metric_value=coverage,
                threshold=0.99,
                unit="ratio",
                evidence="matched MP4 frames divided by all MP4 frames",
            ),
        ]
        return CaptureSessionQualityReport(
            session_id=manifest.session_id,
            generated_at_unix_ns=self._unix_clock_ms() * 1_000_000,
            status=status,
            training_eligibility="ineligible",
            recoverable=recoverable,
            finalization="unclean" if status == "incomplete" else "clean",
            counts=CaptureQualityCounts(
                clip_count=len(manifest.clips),
                complete_clip_count=complete_clips,
                incomplete_clip_count=incomplete_clips,
                video_metadata_count=quality.video_metadata_count,
                unmatched_video_metadata_count=quality.unmatched_video_metadata_count,
                video_frame_count=quality.recorded_video_frame_count,
                video_frames_with_metadata=(quality.recorded_video_frame_metadata_match_count),
                accelerometer_sample_count=quality.accelerometer_sample_count,
                gyroscope_sample_count=quality.gyroscope_sample_count,
                unaligned_imu_sample_count=quality.unaligned_imu_sample_count,
                imu_sequence_gap_count=quality.imu_sequence_gap_count,
                imu_duplicate_count=quality.imu_duplicate_sample_count,
                imu_out_of_order_count=quality.imu_out_of_order_sample_count,
                clock_mapping_segment_count=quality.timestamp_mapping_segment_count,
                rejected_clock_mapping_segment_count=(quality.rejected_clock_mapping_segment_count),
            ),
            checks=checks,
            issues=issues,
        )

    def _recover_stale_sessions(self) -> None:
        if not self._root.exists():
            return
        for manifest_path in self._root.glob("*/session.json"):
            if not _ID_PATTERN.fullmatch(manifest_path.parent.name):
                continue
            manifest = self._read_capture_manifest(manifest_path)
            if manifest is None or manifest.lifecycle.state not in {
                CaptureSessionState.ACTIVE,
                CaptureSessionState.FINALIZING,
            }:
                continue
            session_id = manifest.session_id
            database_path = self._telemetry_path(session_id)
            quality = CaptureSessionQuality()
            if database_path.is_file():
                try:
                    database = CaptureSessionDatabase(session_id, database_path)
                    result = database.finalize_capture(0)
                    database.checkpoint_and_close()
                    quality = result.quality
                    clips = [self._finalize_capture_clip(clip) for clip in manifest.clips]
                except (OSError, sqlite3.Error, ValueError):
                    LOGGER.exception("capture session recovery failed for %s", session_id)
                    clips = manifest.clips
            else:
                clips = manifest.clips
            clips = [
                clip.model_copy(update={"state": "incomplete"})
                if clip.state in {"preparing", "recording"}
                else clip
                for clip in clips
            ]
            recovered = manifest.model_copy(
                update={
                    "lifecycle": manifest.lifecycle.model_copy(
                        update={
                            "state": CaptureSessionState.INCOMPLETE,
                            "ended_at_unix_ns": self._unix_clock_ms() * 1_000_000,
                            "end_reason": "recovery_finalization",
                        }
                    ),
                    "session_time_origin": CaptureSessionTimeOrigin(),
                    "clips": clips,
                }
            )
            self._write_quality_report(
                session_id,
                self._quality_report(
                    recovered,
                    quality,
                    recoverable=database_path.is_file(),
                    recovered=True,
                ),
            )
            self._write_capture_manifest(recovered)

    async def _delete_clip_file_and_manifest(
        self,
        completed_path: Path,
        updated: CaptureSessionManifest,
        original: CaptureSessionManifest,
    ) -> None:
        deleting_path = completed_path.with_suffix(".deleting")
        quality_path = completed_path.parents[1] / "quality.json"
        original_quality = quality_path.read_bytes() if quality_path.is_file() else None
        manifest_updated = False
        database: CaptureSessionDatabase | None = None
        database_committed = False
        try:
            os.replace(completed_path, deleting_path)
            database = CaptureSessionDatabase(
                original.session_id,
                self._telemetry_path(original.session_id),
            )
            quality = database.begin_clip_delete(completed_path.stem)
            report = self._quality_report(
                updated,
                quality,
                recoverable=updated.lifecycle.state is CaptureSessionState.INCOMPLETE,
                recovered=updated.lifecycle.end_reason == "recovery_finalization",
            )
            self._write_capture_manifest(updated)
            self._write_quality_report(updated.session_id, report)
            manifest_updated = True
            database.commit_clip_delete()
            database_committed = True
            database.checkpoint_and_close()
            database = None
            try:
                deleting_path.unlink()
            except OSError:
                LOGGER.warning("deleted clip tombstone remains at %s", deleting_path)
        except Exception as error:
            if database is not None:
                if not database_committed:
                    with suppress(Exception):
                        database.rollback_clip_delete()
                with suppress(Exception):
                    database.checkpoint_and_close()
            if manifest_updated and not database_committed:
                with suppress(Exception):
                    self._write_capture_manifest(original)
                if original_quality is None:
                    with suppress(OSError):
                        quality_path.unlink(missing_ok=True)
                else:
                    with suppress(OSError):
                        quality_path.write_bytes(original_quality)
            if deleting_path.is_file() and not database_committed:
                with suppress(OSError):
                    os.replace(deleting_path, completed_path)
            raise RecordingFailureError("recording clip could not be deleted") from error

    async def _delete_legacy_clip(
        self,
        completed_path: Path,
        updated: RecordingSession,
        original: RecordingSession,
    ) -> None:
        session_directory = completed_path.parent
        deleting_path = completed_path.with_suffix(".deleting")
        manifest_updated = False
        try:
            os.replace(completed_path, deleting_path)
            self._write_legacy_manifest(updated)
            manifest_updated = True
            deleting_path.unlink()
        except Exception as error:
            if manifest_updated:
                with suppress(Exception):
                    self._write_legacy_manifest(original)
            if deleting_path.is_file():
                with suppress(OSError):
                    os.replace(deleting_path, completed_path)
            raise RecordingFailureError("recording clip could not be deleted") from error
        if not updated.clips:
            with suppress(OSError):
                (session_directory / "session.json").unlink(missing_ok=True)
            with suppress(OSError):
                session_directory.rmdir()

    def _write_capture_manifest(self, manifest: CaptureSessionManifest) -> None:
        self._write_json_atomic(
            self._session_directory(manifest.session_id) / "session.json",
            manifest.model_dump(mode="json"),
        )

    def _write_quality_report(
        self,
        session_id: str,
        report: CaptureSessionQualityReport,
    ) -> None:
        self._write_json_atomic(
            self._session_directory(session_id) / "quality.json",
            report.model_dump(mode="json"),
        )

    def _write_legacy_manifest(self, session: RecordingSession) -> None:
        self._write_json_atomic(
            self._session_directory(session.session_id) / "session.json",
            session.model_dump(mode="json"),
        )

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary_path.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_path, path)
        except Exception:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)
            raise

    def _read_capture_manifest(self, path: Path) -> CaptureSessionManifest | None:
        try:
            resolved = path.resolve()
            resolved.relative_to(self._root)
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            if payload.get("contract_id") != "capture-session-v1":
                return None
            return CaptureSessionManifest.model_validate(payload)
        except (OSError, ValueError, json.JSONDecodeError, ValidationError):
            return None

    def _read_legacy_manifest(self, path: Path) -> RecordingSession | None:
        try:
            resolved = path.resolve()
            resolved.relative_to(self._root)
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            if payload.get("contract_id") == "capture-session-v1":
                return None
            return RecordingSession.model_validate(payload)
        except (OSError, ValueError, json.JSONDecodeError, ValidationError):
            return None

    def _require_active_manifest(self) -> CaptureSessionManifest:
        if self._session_manifest is None or self._session_id is None:
            raise RuntimeError("active collection session is missing")
        return self._session_manifest

    def _require_writer(self) -> CaptureSessionWriter:
        if self._session_writer is None:
            raise RuntimeError("active collection telemetry writer is missing")
        return self._session_writer

    def _session_directory(self, session_id: str) -> Path:
        if not _ID_PATTERN.fullmatch(session_id):
            raise ValueError("invalid recording session identifier")
        path = (self._root / session_id).resolve()
        try:
            path.relative_to(self._root)
        except ValueError as error:
            raise ValueError("recording session path escapes the recording root") from error
        return path

    def _telemetry_path(self, session_id: str) -> Path:
        return self._session_directory(session_id) / "telemetry" / "telemetry.sqlite"

    def _partial_file_path(self, session_id: str, clip_id: str) -> Path:
        return self._session_directory(session_id) / "media" / f"{clip_id}.part.mp4"

    def _completed_path(self, session_id: str, clip_id: str) -> Path:
        if not _ID_PATTERN.fullmatch(clip_id):
            raise ValueError("invalid recording clip identifier")
        return self._session_directory(session_id) / "media" / f"{clip_id}.mp4"

    def _legacy_completed_path(self, session_id: str, clip_id: str) -> Path:
        if not _ID_PATTERN.fullmatch(clip_id):
            raise ValueError("invalid recording clip identifier")
        return self._session_directory(session_id) / f"{clip_id}.mp4"

    def _fail_locked(self, error: Exception) -> None:
        if self._partial_path is not None:
            with suppress(OSError):
                self._partial_path.unlink(missing_ok=True)
        self._partial_path = None
        self._recorder = None
        self._state = RecordingState.ERROR
        self._detail = f"recording failed: {type(error).__name__}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
