from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from schemas.recording import RecordingState, RecordingStatus
from ui.gateway.recording_control import RecordingControlCoordinator
from ui.gateway.webrtc_models import (
    RecordingControlAction,
    RecordingControlCommand,
    RecordingControlState,
    StreamControlState,
    StreamControlStatus,
)


class _Recording:
    def __init__(self, state: RecordingState = RecordingState.READY) -> None:
        self.state = state
        self.start_calls = 0
        self.stop_calls = 0
        self.recording_id: str | None = None

    async def status(self) -> RecordingStatus:
        return RecordingStatus(state=self.state, recording_id=self.recording_id)

    async def start(self) -> RecordingStatus:
        self.start_calls += 1
        self.state = RecordingState.COUNTDOWN
        self.recording_id = "a" * 32
        return await self.status()

    async def stop(self) -> RecordingStatus:
        self.stop_calls += 1
        self.state = RecordingState.READY
        self.recording_id = None
        return await self.status()


class _WebRtc:
    def __init__(self, *, source_ready: bool = True) -> None:
        self.source_ready = source_ready
        self.stream_commands = 0
        self.statuses: list[object] = []
        self.callbacks: dict[str, object] = {}
        self.fail_status_publish = False

    def set_recording_control_callbacks(self, **callbacks: object) -> None:
        self.callbacks = callbacks

    async def recording_source(self) -> object | None:
        if not self.source_ready:
            return None
        return SimpleNamespace(width=640, height=480)

    async def control_status(self) -> StreamControlStatus:
        return StreamControlStatus(state=StreamControlState.READY)

    async def send_control_command(self, _command: object) -> StreamControlStatus:
        self.stream_commands += 1
        self.source_ready = True
        return StreamControlStatus(state=StreamControlState.STREAMING)

    async def publish_recording_control_status(self, status: object) -> bool:
        if self.fail_status_publish:
            return False
        self.statuses.append(status)
        return True


def test_recording_control_contract_is_strict_and_bounded() -> None:
    valid = {
        "schema_version": "1.0",
        "message_type": "recording_control_command",
        "command_id": "a" * 32,
        "action": "start",
        "trigger": "temple_double_tap",
        "requested_at_elapsed_realtime_ns": 1,
    }
    assert RecordingControlCommand.model_validate_json(json.dumps(valid)).action.value == "start"
    with pytest.raises(ValidationError):
        RecordingControlCommand.model_validate({**valid, "unknown": True})
    with pytest.raises(ValidationError):
        RecordingControlCommand.model_validate({**valid, "command_id": "ABC"})
    with pytest.raises(ValidationError):
        RecordingControlCommand.model_validate(
            {**valid, "requested_at_elapsed_realtime_ns": -1}
        )


def test_wearer_start_autostarts_stream_and_duplicate_id_is_idempotent() -> None:
    async def scenario() -> None:
        recording = _Recording()
        webrtc = _WebRtc(source_ready=False)
        coordinator = RecordingControlCoordinator(recording, webrtc)  # type: ignore[arg-type]
        command = RecordingControlCommand(
            command_id="1" * 32,
            action=RecordingControlAction.START,
            trigger="temple_double_tap",
            requested_at_elapsed_realtime_ns=10,
        )
        await coordinator.handle_wearer_command(command)
        await coordinator.handle_wearer_command(command)
        await coordinator.close()

        assert webrtc.stream_commands == 1
        assert recording.start_calls == 1
        assert webrtc.statuses[0].state is RecordingControlState.STARTING_STREAM
        assert webrtc.statuses[-1].command_id == command.command_id

    asyncio.run(scenario())


def test_countdown_stop_cancels_and_redundant_stop_returns_current_state() -> None:
    async def scenario() -> None:
        recording = _Recording(RecordingState.COUNTDOWN)
        recording.recording_id = "b" * 32
        webrtc = _WebRtc()
        coordinator = RecordingControlCoordinator(recording, webrtc)  # type: ignore[arg-type]
        result = await coordinator.execute("stop", command_id="2" * 32)
        assert result.state is RecordingState.READY
        assert recording.stop_calls == 1

        result = await coordinator.execute("stop", command_id="3" * 32)
        await coordinator.close()
        assert result.state is RecordingState.READY
        assert recording.stop_calls == 1

    asyncio.run(scenario())


def test_status_wire_shape_has_only_approved_fields() -> None:
    async def scenario() -> None:
        recording = _Recording()
        webrtc = _WebRtc()
        coordinator = RecordingControlCoordinator(recording, webrtc)  # type: ignore[arg-type]
        await coordinator.channel_ready()
        await coordinator.close()
        payload = json.loads(webrtc.statuses[-1].model_dump_json())
        assert set(payload) == {
            "schema_version",
            "message_type",
            "command_id",
            "state",
            "recording_id",
            "countdown_remaining_ms",
            "recording_duration_ms",
            "frame_count",
            "imu_sample_count",
            "detail",
        }

    asyncio.run(scenario())


def test_open_channel_converges_from_unavailable_to_ready_when_source_arrives() -> None:
    async def scenario() -> None:
        recording = _Recording(RecordingState.UNAVAILABLE)
        webrtc = _WebRtc(source_ready=False)
        heartbeat_advanced = asyncio.Event()
        heartbeat_blocked = asyncio.Event()
        sleep_calls = 0

        async def source_arrives(_delay: float) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls == 1:
                recording.state = RecordingState.READY
                webrtc.source_ready = True
                heartbeat_advanced.set()
                return
            await heartbeat_blocked.wait()

        coordinator = RecordingControlCoordinator(  # type: ignore[arg-type]
            recording,
            webrtc,
            sleep=source_arrives,
        )
        await coordinator.channel_ready()
        await heartbeat_advanced.wait()
        for _ in range(10):
            if webrtc.statuses[-1].state is RecordingControlState.READY:
                break
            await asyncio.sleep(0)

        assert webrtc.statuses[0].state is RecordingControlState.UNAVAILABLE
        assert webrtc.statuses[-1].state is RecordingControlState.READY

        await coordinator.channel_closed()
        await coordinator.close()

    asyncio.run(scenario())


def test_finalizing_keeps_active_id_and_library_refreshes_once() -> None:
    async def scenario() -> None:
        recording = _Recording(RecordingState.RECORDING)
        recording.recording_id = "c" * 32
        webrtc = _WebRtc()
        refreshes = 0

        async def refresh() -> None:
            nonlocal refreshes
            refreshes += 1

        coordinator = RecordingControlCoordinator(  # type: ignore[arg-type]
            recording,
            webrtc,
            on_recording_completed=refresh,
        )
        coordinator._sync_heartbeat(await recording.status())
        await coordinator.execute("stop", command_id="4" * 32)
        await asyncio.sleep(0)
        await coordinator.close()

        finalizing = next(
            status
            for status in webrtc.statuses
            if status.state is RecordingControlState.FINALIZING
        )
        assert finalizing.recording_id == "c" * 32
        assert webrtc.statuses[-1].state is RecordingControlState.READY
        assert webrtc.statuses[-1].recording_id is None
        assert refreshes == 1

    asyncio.run(scenario())


def test_wearer_failure_publishes_error_without_unhandled_callback_exception() -> None:
    async def scenario() -> None:
        recording = _Recording()
        webrtc = _WebRtc(source_ready=False)

        async def never_ready(_delay: float) -> None:
            webrtc.source_ready = False

        coordinator = RecordingControlCoordinator(  # type: ignore[arg-type]
            recording,
            webrtc,
            monotonic=iter((0.0, 6.0)).__next__,
            sleep=never_ready,
        )
        command = RecordingControlCommand(
            command_id="5" * 32,
            action=RecordingControlAction.START,
            trigger="temple_double_tap",
            requested_at_elapsed_realtime_ns=10,
        )
        await coordinator.handle_wearer_command(command)
        await coordinator.close()
        assert webrtc.statuses[-1].state is RecordingControlState.ERROR
        assert recording.start_calls == 0

    asyncio.run(scenario())
