from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from ui.gateway.webrtc_models import (
    ImuCapabilities,
    ImuSample,
    ImuSensorType,
    StreamControlAction,
    StreamControlCommand,
)
from ui.gateway.webrtc_runtime import ImuChannelLifecycleSink, WebRtcSessionRuntime

from .writer import ImuCaptureWriter


class CapturePhase(StrEnum):
    WAITING_CONNECTION = "waiting_connection"
    WAITING_CAPABILITIES = "waiting_capabilities"
    STOPPING_VIDEO = "stopping_video"
    WAITING_SAMPLES = "waiting_samples"
    CAPTURING = "capturing"
    FINALIZING = "finalizing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ImuCaptureStatus:
    phase: CapturePhase
    capture_id: str
    rows: int = 0
    accelerometer_rows: int = 0
    gyroscope_rows: int = 0
    sequence_gaps: int = 0
    queue_size: int = 0
    bytes_written: int = 0
    device_span_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    accelerometer_rate_hz: float = 0.0
    gyroscope_rate_hz: float = 0.0
    error: str | None = None


class ImuCalibrationService(ImuChannelLifecycleSink):
    def __init__(
        self, runtime: WebRtcSessionRuntime, output_root: Path, *, capture_id: str | None = None
    ) -> None:
        self.runtime = runtime
        self.writer = ImuCaptureWriter(output_root, capture_id=capture_id)
        self.phase = CapturePhase.WAITING_CONNECTION
        self._stream_stop_sent = False
        self._connection_id: str | None = None
        self._last_sample_at_ns: int | None = None
        self._capture_started_at_ns: int | None = None
        self._error: str | None = None
        self._done = asyncio.Event()
        self._started = asyncio.Event()
        runtime.set_capture_telemetry_sink(self)
        runtime.set_imu_channel_lifecycle_sink(self)

    async def on_connection_started(
        self,
        connection_session_id: str,
        device_session_id: str,
        observed_at_client_monotonic_ns: int,
    ) -> None:
        del device_session_id, observed_at_client_monotonic_ns
        if self.phase is not CapturePhase.WAITING_CONNECTION:
            if self.phase in {
                CapturePhase.FINALIZING,
                CapturePhase.COMPLETE,
                CapturePhase.FAILED,
            }:
                return
            await self.fail("WebRTC connection was replaced")
            return
        self._connection_id = connection_session_id
        self._stream_stop_sent = False
        self.phase = CapturePhase.WAITING_CAPABILITIES

    async def on_connection_state(
        self, connection_session_id: str, state: str, observed_at_client_monotonic_ns: int
    ) -> None:
        del observed_at_client_monotonic_ns
        if connection_session_id != self._connection_id:
            return
        if state in {"disconnected", "closed", "failed", "replaced"} and self.phase not in {
            CapturePhase.WAITING_CONNECTION,
            CapturePhase.FINALIZING,
            CapturePhase.COMPLETE,
            CapturePhase.FAILED,
        }:
            await self.fail(f"WebRTC connection {state}")

    async def on_imu_capabilities(
        self,
        connection_session_id: str,
        capabilities: ImuCapabilities,
        received_at_client_monotonic_ns: int,
    ) -> None:
        del received_at_client_monotonic_ns
        if connection_session_id != self._connection_id or self.phase is CapturePhase.FAILED:
            return
        if capabilities.missing_sensor_types or {
            item.sensor_type for item in capabilities.sensors
        } != set(ImuSensorType):
            await self.fail("accelerometer and gyroscope are both required")
            return
        if self._stream_stop_sent:
            return
        self.phase = CapturePhase.STOPPING_VIDEO
        self._stream_stop_sent = True
        try:
            await self._wait_for_stream_control()
            status = await self.runtime.send_control_command(
                StreamControlCommand(
                    command_id=secrets.token_hex(16),
                    action=StreamControlAction.STOP,
                )
            )
            if status.state.value != "stopped":
                await self.fail(f"video stop was not acknowledged: {status.state.value}")
                return
            if self.phase is CapturePhase.FAILED:
                return
            self.phase = CapturePhase.WAITING_SAMPLES
        except Exception as error:
            await self.fail(f"failed to stop video stream: {error}")

    async def on_imu_sample(
        self, connection_session_id: str, sample: ImuSample, received_at_client_monotonic_ns: int
    ) -> None:
        del received_at_client_monotonic_ns
        if connection_session_id != self._connection_id or self.phase in {
            CapturePhase.FAILED,
            CapturePhase.COMPLETE,
        }:
            return
        if (
            self.phase is not CapturePhase.WAITING_SAMPLES
            and self.phase is not CapturePhase.CAPTURING
        ):
            return
        try:
            if self.phase is CapturePhase.WAITING_SAMPLES:
                self.writer.start()
                self.phase = CapturePhase.CAPTURING
                self._capture_started_at_ns = time.perf_counter_ns()
                self._started.set()
            self.writer.append(sample)
            self._last_sample_at_ns = time.perf_counter_ns()
        except Exception as error:
            await self.fail(str(error))

    async def on_frame_metadata_match(self, connection_session_id: str, match: object) -> None:
        del connection_session_id, match

    async def on_video_frame_metadata(
        self,
        connection_session_id: str,
        metadata: object,
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

    async def on_imu_channel_ready(self, connection_session_id: str) -> None:
        if self._connection_id is None:
            self._connection_id = connection_session_id

    async def on_imu_channel_closed(self, connection_session_id: str) -> None:
        if connection_session_id == self._connection_id and self.phase not in {
            CapturePhase.WAITING_CONNECTION,
            CapturePhase.FINALIZING,
            CapturePhase.COMPLETE,
            CapturePhase.FAILED,
        }:
            await self.fail("IMU data channel closed")

    async def on_imu_message_rejected(self, connection_session_id: str) -> None:
        if connection_session_id == self._connection_id and self.phase not in {
            CapturePhase.FINALIZING,
            CapturePhase.COMPLETE,
            CapturePhase.FAILED,
        }:
            await self.fail("malformed IMU telemetry message rejected")

    async def fail(self, message: str) -> None:
        if self.phase in {CapturePhase.FAILED, CapturePhase.COMPLETE}:
            return
        self._error = message
        self.phase = CapturePhase.FAILED
        try:
            await asyncio.to_thread(self.writer.discard)
        except Exception as error:
            self._error = f"{message}; partial cleanup failed: {error}"
        self._started.set()
        self._done.set()

    async def finish(self) -> Path | None:
        if self.phase is CapturePhase.FAILED:
            return None
        if self.phase is not CapturePhase.CAPTURING:
            await self.fail("capture did not receive both IMU streams")
            return None
        self.phase = CapturePhase.FINALIZING
        try:
            path = await asyncio.to_thread(self.writer.finish, publish=True)
        except Exception as error:
            await self.fail(str(error))
            return None
        self.phase = CapturePhase.COMPLETE
        self._done.set()
        return path

    async def wait(self, timeout_seconds: float | None = None) -> None:
        if timeout_seconds is None:
            await self._done.wait()
        else:
            await asyncio.wait_for(self._done.wait(), timeout_seconds)

    async def wait_until_started_or_done(self) -> None:
        await self._started.wait()

    async def watchdog(self, *, timeout_seconds: float = 5.0) -> None:
        while self.phase not in {CapturePhase.COMPLETE, CapturePhase.FAILED}:
            await asyncio.sleep(min(1.0, timeout_seconds / 2))
            if (
                self.phase is CapturePhase.CAPTURING
                and self._last_sample_at_ns is not None
                and time.perf_counter_ns() - self._last_sample_at_ns
                > timeout_seconds * 1_000_000_000
            ):
                await self.fail(f"no IMU sample received for {timeout_seconds:g} seconds")
                return

    async def _wait_for_stream_control(self, timeout_seconds: float = 5.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            status = await self.runtime.control_status()
            if status.state.value in {"ready", "streaming", "stopped"}:
                return
            if status.state.value == "error":
                raise RuntimeError(status.detail or "stream control is in error state")
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("stream control channel did not become ready")
            await asyncio.sleep(0.05)

    def status(self) -> ImuCaptureStatus:
        stats = self.writer.stats
        device_span_seconds = (
            0.0
            if stats.first_timestamp_ns is None or stats.last_timestamp_ns is None
            else (stats.last_timestamp_ns - stats.first_timestamp_ns) / 1_000_000_000
        )
        elapsed_seconds = (
            0.0
            if self._capture_started_at_ns is None
            else (time.perf_counter_ns() - self._capture_started_at_ns) / 1_000_000_000
        )
        return ImuCaptureStatus(
            self.phase,
            self.writer.capture_id,
            stats.rows,
            stats.accelerometer_rows,
            stats.gyroscope_rows,
            stats.sequence_gaps,
            stats.queue_size,
            stats.bytes_written,
            device_span_seconds,
            elapsed_seconds,
            stats.accelerometer_rate_hz,
            stats.gyroscope_rate_hz,
            self._error,
        )
