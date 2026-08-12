from __future__ import annotations

from dataclasses import dataclass

from schemas.recording import RecordingLibrary, RecordingStatus
from ui.gateway.imu_telemetry import ImuTelemetrySnapshot
from ui.gateway.live_frames import LiveFrameStatus
from ui.gateway.webrtc_models import ImuTelemetryStatus, StreamControlStatus, WebRtcStatus


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    """One coherent read-only view of the recording client."""

    revision: int = 0
    captured_at_client_monotonic_ns: int = 0
    server_ready: bool = False
    webrtc: WebRtcStatus | None = None
    stream_control: StreamControlStatus | None = None
    imu: ImuTelemetryStatus | None = None
    imu_telemetry: ImuTelemetrySnapshot | None = None
    recording: RecordingStatus | None = None
    library: RecordingLibrary | None = None
    display: LiveFrameStatus | None = None
    last_error: str | None = None
    recent_events: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CommandResult:
    name: str
    succeeded: bool
    detail: str
