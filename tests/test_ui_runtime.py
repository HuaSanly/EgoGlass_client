from __future__ import annotations

import asyncio
import threading
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

import ui.application.runtime_host as runtime_module
from schemas.recording import RecordingLibrary, RecordingState, RecordingStatus
from ui.application.runtime_host import RuntimeConfig, UnifiedRuntimeHost
from ui.application.runtime_state import RuntimeSnapshot
from ui.gateway.imu_telemetry import ImuTelemetrySnapshot


class _RecordingStub:
    def __init__(self, library: RecordingLibrary) -> None:
        self._library = library
        self.library_calls = 0

    async def status(self) -> RecordingStatus:
        return RecordingStatus(state=RecordingState.READY)

    async def library(self) -> RecordingLibrary:
        self.library_calls += 1
        return self._library


async def _none() -> None:
    return None


def _host(recording: _RecordingStub) -> UnifiedRuntimeHost:
    host = UnifiedRuntimeHost.__new__(UnifiedRuntimeHost)
    host.webrtc = SimpleNamespace(status=_none, control_status=_none, imu_status=_none)
    host.recording = recording  # type: ignore[assignment]
    host.imu_telemetry = SimpleNamespace(
        snapshot=lambda _now=None: ImuTelemetrySnapshot()
    )
    host.frame_buffer = SimpleNamespace(status=lambda: None)
    host._snapshot_lock = threading.Lock()
    host._snapshot = RuntimeSnapshot()
    host._recent_events = deque(maxlen=50)
    host._library = recording._library
    host._library_refresh_lock = asyncio.Lock()
    return host


def test_status_poll_does_not_rescan_recording_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        recording = _RecordingStub(RecordingLibrary(recordings=[]))
        host = _host(recording)

        async def stop_after_first(_delay: float) -> None:
            raise asyncio.CancelledError

        monkeypatch.setattr(runtime_module.asyncio, "sleep", stop_after_first)
        with pytest.raises(asyncio.CancelledError):
            await host._collect_status()
        assert host.snapshot().revision == 1
        assert recording.library_calls == 0

    asyncio.run(scenario())


def test_library_refresh_is_explicit_and_cached() -> None:
    async def scenario() -> None:
        library = RecordingLibrary(recordings=[])
        recording = _RecordingStub(library)
        host = _host(recording)
        assert await host._refresh_library() is library
        assert host.recording_library() is library
        assert recording.library_calls == 1

    asyncio.run(scenario())


def test_runtime_config_rejects_invalid_ports(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="port"):
        RuntimeConfig(port=0, recordings_root=tmp_path)
