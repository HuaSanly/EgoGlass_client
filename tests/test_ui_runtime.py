from __future__ import annotations

import asyncio
import queue
import threading
from collections import deque
from types import SimpleNamespace

import pytest

import ui.runtime as runtime_module
from ingest_gateway.recording_models import RecordingLibrary
from ui.runtime import UnifiedRuntimeHost
from ui.state import RuntimeSnapshot


class _RecordingStatusStub:
    def __init__(self, library: RecordingLibrary) -> None:
        self.library_calls = 0
        self._library = library

    async def status(self) -> None:
        return None

    async def library(self) -> RecordingLibrary:
        self.library_calls += 1
        return self._library


class _PerceptionEventsStub:
    def __init__(self, payloads: tuple[dict[str, object] | None, ...]) -> None:
        self.payloads = payloads

    async def status_events(self):
        for payload in self.payloads:
            yield payload


def _runtime_stub(recording: _RecordingStatusStub) -> UnifiedRuntimeHost:
    runtime = UnifiedRuntimeHost.__new__(UnifiedRuntimeHost)
    runtime.webrtc = SimpleNamespace(
        status=_return_none,
        control_status=_return_none,
        imu_status=_return_none,
    )
    runtime.recording = recording  # type: ignore[assignment]
    runtime.perception = SimpleNamespace(status=_return_empty_dict)
    runtime.imu_preview = SimpleNamespace(snapshot=lambda: None)
    runtime.frame_buffer = SimpleNamespace(status=lambda: None)
    runtime._snapshot_lock = threading.Lock()
    runtime._snapshot = RuntimeSnapshot()
    runtime._recent_events = deque(maxlen=50)
    runtime._library = None
    runtime._library_refresh_lock = asyncio.Lock()
    return runtime


async def _return_none() -> None:
    return None


async def _return_empty_dict() -> dict[str, object]:
    return {}


def test_status_collection_never_polls_the_recording_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        library = RecordingLibrary(sessions=[])
        recording = _RecordingStatusStub(library)
        runtime = _runtime_stub(recording)
        sleep_calls = 0

        async def stop_after_three_iterations(_delay: float) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls == 3:
                raise asyncio.CancelledError

        monkeypatch.setattr(runtime_module.asyncio, "sleep", stop_after_three_iterations)

        with pytest.raises(asyncio.CancelledError):
            await runtime._collect_status()

        assert runtime.snapshot().revision == 3
        assert recording.library_calls == 0

    asyncio.run(scenario())


def test_initial_library_refresh_scans_once_and_caches_the_result() -> None:
    async def scenario() -> None:
        library = RecordingLibrary(sessions=[])
        recording = _RecordingStatusStub(library)
        runtime = _runtime_stub(recording)

        await runtime._initial_library_refresh()

        assert recording.library_calls == 1
        assert runtime._library is library

    asyncio.run(scenario())


def test_imu_pose_reset_command_resets_preview() -> None:
    async def scenario() -> None:
        calls: list[str] = []
        runtime = UnifiedRuntimeHost.__new__(UnifiedRuntimeHost)
        runtime.imu_preview = SimpleNamespace(reset_orientation=lambda: calls.append("reset"))

        result = await runtime._reset_imu_pose()

        assert calls == ["reset"]
        assert result.detail == "IMU pose reset"

    asyncio.run(scenario())


def test_perception_result_forwarder_keeps_only_the_newest_unique_result() -> None:
    async def scenario() -> None:
        first = {
            "session_id": "session",
            "sequence_id": "connection",
            "frame_index": 10,
            "hands": [],
        }
        latest = {
            "session_id": "session",
            "sequence_id": "connection",
            "frame_index": 12,
            "hands": [{"handedness": "left"}],
        }
        runtime = UnifiedRuntimeHost.__new__(UnifiedRuntimeHost)
        runtime.perception = _PerceptionEventsStub(
            (
                None,
                {"latest_result": {"frame_index": "invalid"}},
                {"latest_result": first},
                {"latest_result": first},
                {"latest_result": latest},
            )
        )
        runtime._perception_results = queue.Queue(maxsize=1)

        await runtime._forward_perception_results()

        assert runtime.take_latest_perception_result() == latest
        assert runtime.take_latest_perception_result() is None

    asyncio.run(scenario())
