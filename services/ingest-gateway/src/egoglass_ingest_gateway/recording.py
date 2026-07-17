from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from .adapters.mp4_recorder import PyAvH264Mp4Recorder
from .adapters.webrtc import WebRtcVideoRecordingSource
from .recording_models import (
    RecordingClip,
    RecordingLibrary,
    RecordingSession,
    RecordingState,
    RecordingStatus,
)

COUNTDOWN_SECONDS = 3.0
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
LOGGER = logging.getLogger(__name__)


class RecordingUnavailableError(RuntimeError):
    """Raised when there is no compatible live Glass3 source."""


class RecordingConflictError(RuntimeError):
    """Raised when a command conflicts with the current recording state."""


class RecordingFailureError(RuntimeError):
    """Raised when a recording could not be finalized safely."""


class RecordingClipNotFoundError(RuntimeError):
    """Raised when a completed recording clip does not exist."""


class RecordingSessionNotFoundError(RuntimeError):
    """Raised when a completed recording session does not exist."""


class RecordingWriter(Protocol):
    @property
    def frames_received(self) -> int: ...

    async def start(self) -> None: ...

    async def wait(self) -> None: ...

    async def stop(self) -> None: ...


RecordingSourceProvider = Callable[
    [], Awaitable[WebRtcVideoRecordingSource | None]
]
RecordingWriterFactory = Callable[[Path, object], RecordingWriter]


class RecordingRuntime:
    """Own the countdown, recorder lifecycle, and completed-clip library."""

    def __init__(
        self,
        root: Path,
        source_provider: RecordingSourceProvider,
        *,
        recorder_factory: RecordingWriterFactory = PyAvH264Mp4Recorder,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        unix_clock_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
        monotonic_clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self._root = root.expanduser().resolve()
        self._source_provider = source_provider
        self._recorder_factory = recorder_factory
        self._sleep = sleep
        self._unix_clock_ms = unix_clock_ms
        self._monotonic_clock_ns = monotonic_clock_ns
        self._command_lock = asyncio.Lock()
        self._countdown_task: asyncio.Task[None] | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._recorder: RecordingWriter | None = None
        self._partial_path: Path | None = None
        self._state = RecordingState.UNAVAILABLE
        self._detail = "Glass3 1920x1080 video is not ready"
        self._session_id: str | None = None
        self._clip_id: str | None = None
        self._countdown_started_at_unix_ms: int | None = None
        self._recording_starts_at_unix_ms: int | None = None
        self._recording_started_at_unix_ms: int | None = None
        self._recording_started_at_monotonic_ns: int | None = None
        self._recording_duration_ms = 0

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
            now_ms = self._unix_clock_ms()
            self._session_id = source.session_id
            self._clip_id = uuid.uuid4().hex
            self._countdown_started_at_unix_ms = now_ms
            self._recording_starts_at_unix_ms = now_ms + int(COUNTDOWN_SECONDS * 1000)
            self._recording_started_at_unix_ms = None
            self._recording_started_at_monotonic_ns = None
            self._recording_duration_ms = 0
            self._state = RecordingState.COUNTDOWN
            self._detail = "recording starts after the server countdown"
            self._session_directory(source.session_id).mkdir(parents=True, exist_ok=True)
            self._countdown_task = asyncio.create_task(
                self._complete_countdown(source.session_id, self._clip_id)
            )
            return self._status_locked()

    async def stop(self) -> RecordingStatus:
        async with self._command_lock:
            if self._state is RecordingState.COUNTDOWN:
                if self._countdown_task is not None:
                    self._countdown_task.cancel()
                    self._countdown_task = None
                self._clip_id = None
                self._countdown_started_at_unix_ms = None
                self._recording_starts_at_unix_ms = None
                self._state = RecordingState.READY
                self._detail = "countdown cancelled; no clip was created"
                if self._session_id is not None:
                    with suppress(OSError):
                        self._session_directory(self._session_id).rmdir()
                return self._status_locked()
            if self._state is not RecordingState.RECORDING:
                raise RecordingConflictError("there is no active recording to stop")
            self._state = RecordingState.FINALIZING
            self._detail = "finalizing MP4"
            try:
                await self._finalize_locked()
            except Exception as error:
                self._fail_locked(error)
                raise RecordingFailureError(self._detail or "recording failed") from error
            return self._status_locked()

    async def library(self) -> RecordingLibrary:
        sessions: list[RecordingSession] = []
        if not self._root.exists():
            return RecordingLibrary(sessions=[])
        for path in self._root.glob("*/session.json"):
            session = self._read_session_manifest(path)
            if session is None:
                continue
            complete_clips = [
                clip
                for clip in session.clips
                if self._completed_path(session.session_id, clip.clip_id).is_file()
                and self._completed_path(session.session_id, clip.clip_id).stat().st_size
                == clip.file_size_bytes
            ]
            if complete_clips:
                sessions.append(
                    session.model_copy(
                        update={
                            "clips": sorted(
                                complete_clips,
                                key=lambda clip: clip.recorded_at_unix_ms,
                                reverse=True,
                            )
                        }
                    )
                )
        sessions.sort(
            key=lambda session: session.started_at_unix_ms,
            reverse=True,
        )
        return RecordingLibrary(sessions=sessions)

    async def media_path(self, session_id: str, clip_id: str) -> Path | None:
        if not _ID_PATTERN.fullmatch(session_id) or not _ID_PATTERN.fullmatch(clip_id):
            return None
        manifest = self._read_session_manifest(
            self._session_directory(session_id) / "session.json"
        )
        if manifest is None:
            return None
        clip = next((item for item in manifest.clips if item.clip_id == clip_id), None)
        if clip is None:
            return None
        path = self._completed_path(session_id, clip_id).resolve()
        try:
            path.relative_to(self._root)
        except ValueError:
            return None
        if not path.is_file() or path.stat().st_size != clip.file_size_bytes:
            return None
        return path

    async def delete_clip(self, session_id: str, clip_id: str) -> RecordingLibrary:
        async with self._command_lock:
            if not _ID_PATTERN.fullmatch(session_id) or not _ID_PATTERN.fullmatch(
                clip_id
            ):
                raise RecordingClipNotFoundError("recording clip not found")
            session_directory = self._session_directory(session_id)
            manifest_path = session_directory / "session.json"
            session = self._read_session_manifest(manifest_path)
            if session is None or session.session_id != session_id:
                raise RecordingClipNotFoundError("recording clip not found")
            clip = next(
                (item for item in session.clips if item.clip_id == clip_id),
                None,
            )
            completed_path = self._completed_path(session_id, clip_id)
            if clip is None or not completed_path.is_file():
                raise RecordingClipNotFoundError("recording clip not found")

            remaining_clips = [
                item for item in session.clips if item.clip_id != clip_id
            ]
            updated_session = session.model_copy(
                update={"clips": remaining_clips}
            )
            deleting_path = session_directory / f"{clip_id}.deleting"
            manifest_updated = False
            try:
                os.replace(completed_path, deleting_path)
                self._write_session_manifest(updated_session)
                manifest_updated = True
                deleting_path.unlink()
            except Exception as error:
                try:
                    if manifest_updated:
                        self._write_session_manifest(session)
                    if deleting_path.is_file():
                        os.replace(deleting_path, completed_path)
                except Exception:
                    LOGGER.exception("recording delete rollback failed")
                raise RecordingFailureError(
                    "recording clip could not be deleted"
                ) from error

            if not remaining_clips:
                with suppress(OSError):
                    manifest_path.unlink(missing_ok=True)
                with suppress(OSError):
                    session_directory.rmdir()
            return await self.library()

    async def rename_session(
        self,
        session_id: str,
        display_name: str,
    ) -> RecordingLibrary:
        async with self._command_lock:
            if not _ID_PATTERN.fullmatch(session_id):
                raise RecordingSessionNotFoundError("recording session not found")
            manifest_path = self._session_directory(session_id) / "session.json"
            session = self._read_session_manifest(manifest_path)
            if session is None or session.session_id != session_id:
                raise RecordingSessionNotFoundError("recording session not found")
            try:
                self._write_session_manifest(
                    session.model_copy(update={"display_name": display_name})
                )
            except Exception as error:
                raise RecordingFailureError(
                    "recording session could not be renamed"
                ) from error
            return await self.library()

    async def close(self) -> None:
        async with self._command_lock:
            if self._countdown_task is not None:
                self._countdown_task.cancel()
                self._countdown_task = None
                if self._session_id is not None:
                    with suppress(OSError):
                        self._session_directory(self._session_id).rmdir()
            if self._recorder is not None:
                self._state = RecordingState.FINALIZING
                self._detail = "finalizing MP4 during shutdown"
                try:
                    await self._finalize_locked()
                except Exception as error:
                    self._fail_locked(error)

    async def _complete_countdown(self, session_id: str, clip_id: str) -> None:
        try:
            await self._sleep(COUNTDOWN_SECONDS)
            async with self._command_lock:
                if (
                    self._state is not RecordingState.COUNTDOWN
                    or self._session_id != session_id
                    or self._clip_id != clip_id
                ):
                    return
                source = await self._compatible_source()
                if source.session_id != session_id:
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
                    try:
                        await recorder.stop()
                    finally:
                        raise
                self._recording_started_at_unix_ms = self._unix_clock_ms()
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
                if self._session_id == session_id and self._clip_id == clip_id:
                    self._countdown_task = None
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
                await self._finalize_locked()
            except Exception as error:
                self._fail_locked(error)

    async def _finalize_locked(self) -> None:
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
        if not partial_path.is_file() or partial_path.stat().st_size <= 0:
            raise RuntimeError("recorder produced no playable MP4 data")
        if self._session_id is None or self._clip_id is None:
            raise RuntimeError("recording identifiers are missing")
        if (
            self._recording_started_at_unix_ms is None
            or self._recording_started_at_monotonic_ns is None
        ):
            raise RuntimeError("recording start timestamps are missing")
        ended_at_ms = self._unix_clock_ms()
        duration_ms = max(
            0,
            (self._monotonic_clock_ns() - self._recording_started_at_monotonic_ns)
            // 1_000_000,
        )
        completed_path = self._completed_path(self._session_id, self._clip_id)
        os.replace(partial_path, completed_path)
        clip = RecordingClip(
            clip_id=self._clip_id,
            recorded_at_unix_ms=self._recording_started_at_unix_ms,
            ended_at_unix_ms=ended_at_ms,
            duration_ms=duration_ms,
            file_size_bytes=completed_path.stat().st_size,
            media_url=(
                f"/api/v1/recordings/media/{self._session_id}/{self._clip_id}"
            ),
        )
        try:
            self._append_manifest(self._session_id, clip)
        except Exception:
            completed_path.unlink(missing_ok=True)
            raise
        self._recording_duration_ms = duration_ms
        self._state = RecordingState.READY
        self._detail = "recording finalized"

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
                "Glass3 video must be 1920x1080 before recording can start"
            )
        if not _ID_PATTERN.fullmatch(source.session_id):
            raise RecordingUnavailableError("Glass3 session identifier is invalid")
        return source

    async def _refresh_availability_locked(self) -> None:
        try:
            source = await self._compatible_source()
        except RecordingUnavailableError as error:
            self._state = RecordingState.UNAVAILABLE
            self._detail = str(error)
            return
        self._state = RecordingState.READY
        self._detail = ""
        self._session_id = source.session_id

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
            session_id=self._session_id,
            clip_id=self._clip_id,
            countdown_started_at_unix_ms=self._countdown_started_at_unix_ms,
            recording_starts_at_unix_ms=self._recording_starts_at_unix_ms,
            recording_started_at_unix_ms=self._recording_started_at_unix_ms,
            recording_duration_ms=duration_ms,
        )

    def _append_manifest(self, session_id: str, clip: RecordingClip) -> None:
        session_directory = self._session_directory(session_id)
        manifest_path = session_directory / "session.json"
        if manifest_path.exists():
            session = self._read_session_manifest(manifest_path)
            if session is None or session.session_id != session_id:
                raise RuntimeError("existing recording session manifest is invalid")
            session = session.model_copy(update={"clips": [*session.clips, clip]})
        else:
            session = RecordingSession(
                session_id=session_id,
                started_at_unix_ms=clip.recorded_at_unix_ms,
                clips=[clip],
            )
        self._write_session_manifest(session)

    def _write_session_manifest(self, session: RecordingSession) -> None:
        session_directory = self._session_directory(session.session_id)
        session_directory.mkdir(parents=True, exist_ok=True)
        temporary_path = session_directory / "session.json.tmp"
        try:
            temporary_path.write_text(
                json.dumps(session.model_dump(mode="json"), indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_path, session_directory / "session.json")
        except Exception:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)
            raise

    def _read_session_manifest(self, path: Path) -> RecordingSession | None:
        try:
            resolved = path.resolve()
            resolved.relative_to(self._root)
            return RecordingSession.model_validate_json(resolved.read_text(encoding="utf-8"))
        except (OSError, ValueError, ValidationError):
            return None

    def _session_directory(self, session_id: str) -> Path:
        if not _ID_PATTERN.fullmatch(session_id):
            raise ValueError("invalid recording session identifier")
        path = (self._root / session_id).resolve()
        try:
            path.relative_to(self._root)
        except ValueError as error:
            raise ValueError("recording session path escapes the recording root") from error
        return path

    def _partial_file_path(self, session_id: str, clip_id: str) -> Path:
        return self._session_directory(session_id) / f"{clip_id}.part.mp4"

    def _completed_path(self, session_id: str, clip_id: str) -> Path:
        if not _ID_PATTERN.fullmatch(clip_id):
            raise ValueError("invalid recording clip identifier")
        return self._session_directory(session_id) / f"{clip_id}.mp4"

    def _fail_locked(self, error: Exception) -> None:
        if self._partial_path is not None:
            with suppress(OSError):
                self._partial_path.unlink(missing_ok=True)
        self._partial_path = None
        self._recorder = None
        if self._session_id is not None:
            with suppress(OSError):
                self._session_directory(self._session_id).rmdir()
        self._state = RecordingState.ERROR
        self._detail = f"recording failed: {type(error).__name__}"
