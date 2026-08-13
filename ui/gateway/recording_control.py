from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable

from schemas.recording import RecordingState, RecordingStatus

from .recording import RecordingRuntime
from .webrtc_models import (
    RecordingControlAction,
    RecordingControlCommand,
    RecordingControlState,
    RecordingControlStatus,
    StreamControlAction,
    StreamControlCommand,
    StreamControlState,
)
from .webrtc_runtime import WebRtcSessionRuntime

STATUS_HEARTBEAT_SECONDS = 0.25
STREAM_START_TIMEOUT_SECONDS = 5.0
STREAM_READY_POLL_SECONDS = 0.05
COMMAND_HISTORY_SIZE = 256
LOGGER = logging.getLogger(__name__)


class RecordingControlCoordinator:
    """Serialize Qt, HTTP, and Glass3 wearer recording commands."""

    def __init__(
        self,
        recording: RecordingRuntime,
        webrtc: WebRtcSessionRuntime,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        stream_start_timeout_seconds: float = STREAM_START_TIMEOUT_SECONDS,
        on_recording_completed: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._recording = recording
        self._webrtc = webrtc
        self._monotonic = monotonic
        self._sleep = sleep
        self._stream_start_timeout_seconds = stream_start_timeout_seconds
        self._on_recording_completed = on_recording_completed
        self._lock = asyncio.Lock()
        self._completed_commands: OrderedDict[str, RecordingControlStatus] = OrderedDict()
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._closed = False
        self._channel_open = False
        self._completion_refresh_pending = False
        webrtc.set_recording_control_callbacks(
            on_ready=self.channel_ready,
            on_closed=self.channel_closed,
            on_command=self.handle_wearer_command,
        )

    async def execute(
        self,
        action: RecordingControlAction | str,
        *,
        command_id: str | None = None,
    ) -> RecordingStatus:
        action = RecordingControlAction(action)
        async with self._lock:
            if command_id is not None and command_id in self._completed_commands:
                await self._webrtc.publish_recording_control_status(
                    self._completed_commands[command_id]
                )
                return await self._recording.status()
            try:
                current = await self._recording.status()
                if action is RecordingControlAction.START:
                    result = await self._start(current, command_id)
                else:
                    result = await self._stop(current, command_id)
                status = await self._publish(result, command_id)
                self._remember(command_id, status)
                self._sync_heartbeat(result)
                return result
            except Exception as error:
                status = RecordingControlStatus(
                    command_id=command_id,
                    state=RecordingControlState.ERROR,
                    detail=(str(error) or type(error).__name__)[:256],
                )
                await self._webrtc.publish_recording_control_status(status)
                self._remember(command_id, status)
                raise

    async def handle_wearer_command(self, command: RecordingControlCommand) -> None:
        try:
            await self.execute(command.action, command_id=command.command_id)
        except Exception:
            LOGGER.exception("Glass3 wearer recording command failed")

    async def channel_ready(self) -> None:
        self._channel_open = True
        LOGGER.info("recording-control-v1 channel is ready")
        status = await self._recording.status()
        await self._publish(status, None)
        self._ensure_heartbeat()

    async def channel_closed(self) -> None:
        self._channel_open = False
        LOGGER.info("recording-control-v1 channel is closed")
        self._cancel_heartbeat()

    async def close(self) -> None:
        self._closed = True
        self._channel_open = False
        task, self._heartbeat_task = self._heartbeat_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _start(
        self,
        current: RecordingStatus,
        command_id: str | None,
    ) -> RecordingStatus:
        if current.state in {
            RecordingState.COUNTDOWN,
            RecordingState.RECORDING,
            RecordingState.FINALIZING,
        }:
            return current
        initial_source = await self._webrtc.recording_source()
        stream_status = await self._webrtc.control_status()
        needs_stream_start = (
            initial_source is None
            or (initial_source.width, initial_source.height) != (640, 480)
            or stream_status.state is not StreamControlState.STREAMING
        )
        if needs_stream_start:
            published = await self._webrtc.publish_recording_control_status(
                RecordingControlStatus(
                    command_id=command_id,
                    state=RecordingControlState.STARTING_STREAM,
                    detail="starting Glass3 camera stream",
                )
            )
            if not published:
                LOGGER.debug("recording-control-v1 unavailable during stream startup")
            baseline_generation = getattr(initial_source, "camera_start_generation", None)
            source_is_compatible = (
                initial_source is not None
                and (initial_source.width, initial_source.height) == (640, 480)
            )
            should_send_stream_start = (
                stream_status.state is not StreamControlState.STARTING
                and (
                    stream_status.state is not StreamControlState.STREAMING
                    or not source_is_compatible
                )
            )
            if should_send_stream_start:
                await self._webrtc.send_control_command(
                    StreamControlCommand(
                        command_id=secrets.token_hex(16),
                        action=StreamControlAction.START,
                    )
                )
            deadline = self._monotonic() + self._stream_start_timeout_seconds
            while self._monotonic() < deadline:
                source = await self._webrtc.recording_source()
                generation = getattr(source, "camera_start_generation", None)
                generation_is_fresh = (
                    baseline_generation is None or generation != baseline_generation
                )
                if (
                    source is not None
                    and (source.width, source.height) == (640, 480)
                    and generation_is_fresh
                ):
                    break
                await self._sleep(STREAM_READY_POLL_SECONDS)
            else:
                raise TimeoutError("Glass3 640x480 recording source did not become ready")
        return await self._recording.start()

    async def _stop(
        self,
        current: RecordingStatus,
        command_id: str | None,
    ) -> RecordingStatus:
        if current.state is RecordingState.COUNTDOWN:
            self._completion_refresh_pending = False
            return await self._recording.stop()
        if current.state is not RecordingState.RECORDING:
            return current
        published = await self._webrtc.publish_recording_control_status(
            self._map_status(
                current.model_copy(update={"state": RecordingState.FINALIZING}),
                command_id,
            )
        )
        if not published:
            LOGGER.warning("recording-control-v1 channel unavailable during finalizing transition")
        self._completion_refresh_pending = False
        result = await self._recording.stop()
        if self._on_recording_completed is not None:
            await self._on_recording_completed()
        return result

    async def _publish(
        self,
        status: RecordingStatus,
        command_id: str | None,
    ) -> RecordingControlStatus:
        control_status = self._map_status(status, command_id)
        published = await self._webrtc.publish_recording_control_status(control_status)
        if not published:
            LOGGER.debug("recording-control-v1 status not delivered because channel is unavailable")
        return control_status

    def _map_status(
        self,
        status: RecordingStatus,
        command_id: str | None,
    ) -> RecordingControlStatus:
        countdown_remaining_ms = None
        if (
            status.state is RecordingState.COUNTDOWN
            and status.recording_starts_at_unix_ms is not None
        ):
            countdown_remaining_ms = max(
                0,
                status.recording_starts_at_unix_ms - time.time_ns() // 1_000_000,
            )
        state = RecordingControlState(status.state.value)
        return RecordingControlStatus(
            command_id=command_id,
            state=state,
            recording_id=status.recording_id,
            countdown_remaining_ms=countdown_remaining_ms,
            recording_duration_ms=status.recording_duration_ms,
            frame_count=status.frame_count,
            imu_sample_count=status.imu_sample_count,
            detail=status.detail or None,
        )

    def _remember(
        self,
        command_id: str | None,
        status: RecordingControlStatus,
    ) -> None:
        if command_id is None:
            return
        self._completed_commands[command_id] = status
        self._completed_commands.move_to_end(command_id)
        while len(self._completed_commands) > COMMAND_HISTORY_SIZE:
            self._completed_commands.popitem(last=False)

    def _sync_heartbeat(self, status: RecordingStatus) -> None:
        active = status.state in {
            RecordingState.COUNTDOWN,
            RecordingState.RECORDING,
            RecordingState.FINALIZING,
        }
        if active:
            self._completion_refresh_pending = True
        if self._channel_open:
            self._ensure_heartbeat()

    def _ensure_heartbeat(self) -> None:
        if self._heartbeat_task is None and not self._closed and self._channel_open:
            self._heartbeat_task = asyncio.create_task(self._heartbeat())

    def _cancel_heartbeat(self) -> None:
        task, self._heartbeat_task = self._heartbeat_task, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    async def _heartbeat(self) -> None:
        try:
            while not self._closed and self._channel_open:
                async with self._lock:
                    status = await self._recording.status()
                    await self._publish(status, None)
                    if (
                        status.state is RecordingState.READY
                        and self._completion_refresh_pending
                    ):
                        if self._on_recording_completed is not None:
                            await self._on_recording_completed()
                        self._completion_refresh_pending = False
                await self._sleep(STATUS_HEARTBEAT_SECONDS)
        except asyncio.CancelledError:
            return
        finally:
            if self._heartbeat_task is asyncio.current_task():
                self._heartbeat_task = None
