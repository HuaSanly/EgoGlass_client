from __future__ import annotations

from dataclasses import dataclass, field

from ui.gateway.imu_preview import ImuPoseSnapshot
from ui.gateway.live_frames import LiveFrameStatus
from ui.gateway.recording_models import RecordingLibrary, RecordingStatus
from ui.gateway.webrtc_models import (
    ImuTelemetryStatus,
    StreamControlStatus,
    WebRtcStatus,
)
from ui.processing import ProcessingServiceSnapshot


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    """One coherent, read-only UI view of the running client modules."""

    revision: int = 0
    captured_at_client_monotonic_ns: int = 0
    server_ready: bool = False
    webrtc: WebRtcStatus | None = None
    stream_control: StreamControlStatus | None = None
    imu: ImuTelemetryStatus | None = None
    imu_pose: ImuPoseSnapshot | None = None
    recording: RecordingStatus | None = None
    library: RecordingLibrary | None = None
    perception: dict[str, object] = field(default_factory=dict)
    processing: ProcessingServiceSnapshot | None = None
    display: LiveFrameStatus | None = None
    last_error: str | None = None
    recent_events: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Completed command notification consumed by the Qt main thread."""

    name: str
    succeeded: bool
    detail: str
