from __future__ import annotations

import asyncio

from schemas.recording import RecordingState, RecordingStatus
from ui.gateway.recording_control import RecordingControlCoordinator
from ui.gateway.webrtc_models import (
    RecordingControlState,
    StreamControlState,
    StreamControlStatus,
)


class _Recording:
    def __init__(self) -> None:
        self.state = RecordingState.READY
        self.transitions: list[RecordingState] = []

    async def status(self) -> RecordingStatus:
        return RecordingStatus(state=self.state)

    async def start(self) -> RecordingStatus:
        self.state = RecordingState.COUNTDOWN
        self.transitions.append(self.state)
        return RecordingStatus(state=self.state, recording_id="a" * 32)

    async def stop(self) -> RecordingStatus:
        self.state = RecordingState.READY
        self.transitions.append(self.state)
        return RecordingStatus(state=self.state)


class _WebRtc:
    def __init__(self) -> None:
        self.source = None
        self.statuses: list[object] = []

    def set_recording_control_callbacks(self, **_callbacks: object) -> None:
        return None

    async def recording_source(self) -> object | None:
        return self.source

    async def control_status(self) -> StreamControlStatus:
        return StreamControlStatus(state=StreamControlState.STOPPED)

    async def send_control_command(self, _command: object) -> StreamControlStatus:
        self.source = type("Source", (), {"width": 640, "height": 480})()
        return StreamControlStatus(state=StreamControlState.STREAMING)

    async def publish_recording_control_status(self, status: object) -> bool:
        self.statuses.append(status)
        return True


def test_ten_wearer_commands_do_not_duplicate_runtime_transitions() -> None:
    async def scenario() -> None:
        recording = _Recording()
        webrtc = _WebRtc()
        coordinator = RecordingControlCoordinator(recording, webrtc)  # type: ignore[arg-type]
        for index in range(10):
            command_id = f"{index:032x}"
            await coordinator.execute("start", command_id=command_id)
            await coordinator.execute("start", command_id=command_id)
            await coordinator.execute("stop", command_id=f"{index + 16:032x}")
        await coordinator.close()
        assert recording.transitions == [
            state
            for _ in range(10)
            for state in (RecordingState.COUNTDOWN, RecordingState.READY)
        ]

    asyncio.run(scenario())


def test_idle_hud_recovers_after_late_recording_source() -> None:
    async def scenario() -> None:
        recording = _Recording()
        recording.state = RecordingState.UNAVAILABLE
        webrtc = _WebRtc()
        source_ready = asyncio.Event()
        heartbeat_blocked = asyncio.Event()
        sleep_calls = 0

        async def connect_source(_delay: float) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls == 1:
                recording.state = RecordingState.READY
                webrtc.source = type("Source", (), {"width": 640, "height": 480})()
                source_ready.set()
                return
            await heartbeat_blocked.wait()

        coordinator = RecordingControlCoordinator(  # type: ignore[arg-type]
            recording,
            webrtc,
            sleep=connect_source,
        )
        await coordinator.channel_ready()
        await source_ready.wait()
        for _ in range(10):
            if webrtc.statuses[-1].state is RecordingControlState.READY:
                break
            await asyncio.sleep(0)

        states = [status.state for status in webrtc.statuses]
        assert states[0] is RecordingControlState.UNAVAILABLE
        assert states[-1] is RecordingControlState.READY
        assert sleep_calls <= 2

        await coordinator.channel_closed()
        await coordinator.close()

    asyncio.run(scenario())
