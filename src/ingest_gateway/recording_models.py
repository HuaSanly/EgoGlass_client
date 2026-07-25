from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _normalize_display_name(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("display_name must contain visible characters")
    return normalized


class RecordingState(StrEnum):
    UNAVAILABLE = "unavailable"
    READY = "ready"
    COUNTDOWN = "countdown"
    RECORDING = "recording"
    FINALIZING = "finalizing"
    ERROR = "error"


class CaptureSessionState(StrEnum):
    ACTIVE = "active"
    FINALIZING = "finalizing"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class TimestampAlignmentState(StrEnum):
    UNVERIFIED = "unverified"


class RecordingOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    width: Literal[1280] = 1280
    height: Literal[720] = 720
    fps: Literal[30] = 30
    container: Literal["mp4"] = "mp4"
    video_codec: Literal["h264"] = "h264"


class RecordingCommandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["start", "stop"]


class RecordingSessionCommandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["new", "finalize"]


class RecordingSessionRenameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    display_name: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[^\x00-\x1f\x7f]+$",
    )

    @field_validator("display_name")
    @classmethod
    def normalize_display_name(cls, value: str) -> str:
        return _normalize_display_name(value)


class RecordingStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    state: RecordingState
    detail: str = Field(default="", max_length=256)
    session_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    session_state: CaptureSessionState | None = None
    clip_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    countdown_started_at_unix_ms: int | None = Field(default=None, ge=0)
    recording_starts_at_unix_ms: int | None = Field(default=None, ge=0)
    recording_started_at_unix_ms: int | None = Field(default=None, ge=0)
    recording_duration_ms: int = Field(default=0, ge=0)
    output: RecordingOutput = Field(default_factory=RecordingOutput)


class RecordingClip(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    recorded_at_unix_ms: int = Field(ge=0)
    ended_at_unix_ms: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    width: int = Field(default=1280, gt=0, le=8192)
    height: int = Field(default=720, gt=0, le=8192)
    fps: Literal[30] = 30
    file_size_bytes: int = Field(gt=0)
    frame_count: int = Field(default=0, ge=0)
    media_url: str = Field(pattern=r"^/api/v1/recordings/media/[0-9a-f]{32}/[0-9a-f]{32}$")


class CaptureSessionQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    imu_sample_count: int = Field(default=0, ge=0)
    accelerometer_sample_count: int = Field(default=0, ge=0)
    gyroscope_sample_count: int = Field(default=0, ge=0)
    imu_sequence_gap_count: int = Field(default=0, ge=0)
    imu_duplicate_sample_count: int = Field(default=0, ge=0)
    imu_out_of_order_sample_count: int = Field(default=0, ge=0)
    telemetry_queue_overflow_count: int = Field(default=0, ge=0)
    connection_segment_count: int = Field(default=0, ge=0)
    matched_video_frame_count: int = Field(default=0, ge=0)
    video_metadata_count: int = Field(default=0, ge=0)
    unmatched_video_metadata_count: int = Field(default=0, ge=0)
    recorded_video_frame_count: int = Field(default=0, ge=0)
    recorded_video_frame_metadata_match_count: int = Field(default=0, ge=0)
    metadata_match_coverage: float | None = Field(default=None, ge=0, le=1)
    timestamp_mapping_segment_count: int = Field(default=0, ge=0)
    rejected_clock_mapping_segment_count: int = Field(default=0, ge=0)
    timestamp_max_uncertainty_ns: int | None = Field(default=None, ge=0)
    unaligned_imu_sample_count: int = Field(default=0, ge=0)
    timestamp_alignment_state: TimestampAlignmentState = TimestampAlignmentState.UNVERIFIED


class RecordingSession(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    started_at_unix_ms: int = Field(ge=0)
    display_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[^\x00-\x1f\x7f]+$",
    )
    state: CaptureSessionState = CaptureSessionState.COMPLETE
    ended_at_unix_ms: int | None = Field(default=None, ge=0)
    recoverable: bool = False
    telemetry_database: Literal["telemetry/telemetry.sqlite"] | None = None
    quality: CaptureSessionQuality = Field(default_factory=CaptureSessionQuality)
    clips: list[RecordingClip]

    @field_validator("display_name")
    @classmethod
    def normalize_display_name(cls, value: str | None) -> str | None:
        return None if value is None else _normalize_display_name(value)


class RecordingLibrary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    sessions: list[RecordingSession]


class CaptureSessionLifecycle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: CaptureSessionState
    start_reason: Literal["first_recording_request"] = "first_recording_request"
    started_at_unix_ns: int = Field(ge=0)
    ended_at_unix_ns: int | None = Field(default=None, ge=0)
    end_reason: (
        Literal[
            "manual_new_session",
            "client_shutdown",
            "device_changed",
            "recovery_finalization",
        ]
        | None
    ) = None


class CaptureSessionTimeOrigin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["pending", "established"] = "pending"
    clock_id: Literal["glasses_elapsed_realtime_ns"] = "glasses_elapsed_realtime_ns"
    origin_elapsed_realtime_ns: int | None = Field(default=None, ge=0)
    origin_event: (
        Literal[
            "first_imu_sample",
            "first_video_frame",
            "glasses_clock_handshake",
        ]
        | None
    ) = None


class CaptureSessionStorage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    telemetry_database_path: Literal["telemetry/telemetry.sqlite"] = "telemetry/telemetry.sqlite"
    quality_report_path: Literal["quality.json"] = "quality.json"
    media_directory: Literal["media"] = "media"
    annotations_directory: Literal["annotations"] = "annotations"
    derived_directory: Literal["derived"] = "derived"


class CaptureDeviceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manufacturer: Literal["Rokid"] = "Rokid"
    model: Literal["Glass3 Enterprise"] = "Glass3 Enterprise"
    device_id: str | None = Field(default=None, min_length=1, max_length=128)
    firmware_version: str | None = Field(default=None, min_length=1, max_length=128)


class CaptureSoftwareProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    glasses_sdk: Literal["com.rokid.security:glass3.open.sdk:2.2.0-E"] = (
        "com.rokid.security:glass3.open.sdk:2.2.0-E"
    )
    glasses_app_version: str | None = Field(default=None, min_length=1, max_length=128)
    client_app_version: str | None = Field(default="0.1.0", min_length=1, max_length=128)


class CaptureSourceRevisions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    superproject_commit: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")
    glasses_commit: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")
    client_commit: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")


class CaptureConfigProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capture_config_id: str | None = Field(default="720p30", min_length=1, max_length=128)
    source: Literal["glasses_negotiated", "client_default", "unknown"] = "client_default"
    width: int | None = Field(default=1280, ge=1)
    height: int | None = Field(default=720, ge=1)
    nominal_fps: float | None = Field(default=30.0, gt=0, le=240)
    target_bitrate_bps: int | None = Field(default=8_000_000, ge=1)


class CaptureSessionProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: CaptureDeviceProvenance = Field(default_factory=CaptureDeviceProvenance)
    software: CaptureSoftwareProvenance = Field(default_factory=CaptureSoftwareProvenance)
    source_revisions: CaptureSourceRevisions = Field(default_factory=CaptureSourceRevisions)
    capture_config: CaptureConfigProvenance = Field(default_factory=CaptureConfigProvenance)


class CaptureVideoProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    container: Literal["mp4"] = "mp4"
    codec: Literal["h264"] = "h264"
    width: int = Field(ge=1, le=8192)
    height: int = Field(ge=1, le=8192)
    nominal_fps: float = Field(gt=0, le=240)


class CaptureSessionClip(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    state: Literal["preparing", "recording", "complete", "incomplete", "cancelled"]
    relative_media_path: str = Field(pattern=r"^media/[A-Za-z0-9][A-Za-z0-9_.-]*[.]mp4$")
    requested_at_session_time_ns: int | None = Field(default=None, ge=0)
    started_at_session_time_ns: int | None = Field(default=None, ge=0)
    ended_at_session_time_ns: int | None = Field(default=None, ge=0)
    video_profile: CaptureVideoProfile
    frame_count: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_complete_clip(self) -> CaptureSessionClip:
        if self.state == "complete" and (
            self.frame_count is None or self.frame_count < 1 or self.sha256 is None
        ):
            raise ValueError("complete clip requires frame_count and sha256")
        if (
            self.started_at_session_time_ns is not None
            and self.ended_at_session_time_ns is not None
            and self.ended_at_session_time_ns < self.started_at_session_time_ns
        ):
            raise ValueError("clip end session time cannot precede its start")
        return self


class CaptureSessionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    contract_id: Literal["capture-session-v1"] = "capture-session-v1"
    session_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    display_name: str = Field(min_length=1, max_length=128)
    display_name_source: Literal["timestamp_default", "operator"]
    lifecycle: CaptureSessionLifecycle
    session_time_origin: CaptureSessionTimeOrigin
    imu_capture_policy: Literal["continuous_while_session_active"] = (
        "continuous_while_session_active"
    )
    provenance: CaptureSessionProvenance = Field(default_factory=CaptureSessionProvenance)
    storage: CaptureSessionStorage = Field(default_factory=CaptureSessionStorage)
    clips: list[CaptureSessionClip]


class CaptureQualityCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_count: int = Field(ge=0)
    complete_clip_count: int = Field(ge=0)
    incomplete_clip_count: int = Field(ge=0)
    video_metadata_count: int = Field(ge=0)
    unmatched_video_metadata_count: int = Field(ge=0)
    video_frame_count: int = Field(ge=0)
    video_frames_with_metadata: int = Field(ge=0)
    accelerometer_sample_count: int = Field(ge=0)
    gyroscope_sample_count: int = Field(ge=0)
    unaligned_imu_sample_count: int = Field(ge=0)
    imu_sequence_gap_count: int = Field(ge=0)
    imu_duplicate_count: int = Field(ge=0)
    imu_out_of_order_count: int = Field(ge=0)
    clock_mapping_segment_count: int = Field(ge=0)
    rejected_clock_mapping_segment_count: int = Field(ge=0)


class CaptureQualityCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    status: Literal["pass", "warn", "fail", "not_evaluated"]
    metric_value: float | None
    threshold: float | None
    unit: str | None
    evidence: str


class CaptureQualityIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_id: str
    severity: Literal["info", "warning", "error"]
    message: str
    recoverable: bool
    evidence: str


class CaptureSessionQualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    contract_id: Literal["capture-session-quality-v1"] = "capture-session-quality-v1"
    session_id: str
    generated_at_unix_ns: int = Field(ge=0)
    status: Literal["pass", "warn", "fail", "incomplete"]
    training_eligibility: Literal["eligible", "review_required", "ineligible"]
    recoverable: bool
    finalization: Literal["clean", "unclean", "recovered"]
    counts: CaptureQualityCounts
    checks: list[CaptureQualityCheck]
    issues: list[CaptureQualityIssue]
