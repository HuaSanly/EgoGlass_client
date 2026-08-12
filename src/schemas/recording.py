"""Typed boundary contracts for one independent capture recording."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RECORDING_ID_PATTERN = r"^[0-9a-f]{32}$"
SHA256_PATTERN = r"^[a-f0-9]{64}$"
CONNECTION_SESSION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"


class RecordingState(StrEnum):
    UNAVAILABLE = "unavailable"
    READY = "ready"
    COUNTDOWN = "countdown"
    RECORDING = "recording"
    FINALIZING = "finalizing"
    ERROR = "error"


class CaptureRecordingState(StrEnum):
    RECORDING = "recording"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class RecordingOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    width: int = Field(default=640, gt=0, le=8192)
    height: int = Field(default=480, gt=0, le=8192)
    fps: float = Field(default=30.0, gt=0, le=240)
    container: Literal["mp4"] = "mp4"
    video_codec: Literal["h264"] = "h264"


class RecordingCommandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["start", "stop"]


class RecordingStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = "2.0"
    state: RecordingState
    detail: str = Field(default="", max_length=256)
    recording_id: str | None = Field(default=None, pattern=RECORDING_ID_PATTERN)
    countdown_started_at_unix_ms: int | None = Field(default=None, ge=0)
    recording_starts_at_unix_ms: int | None = Field(default=None, ge=0)
    recording_started_at_unix_ms: int | None = Field(default=None, ge=0)
    recording_duration_ms: int = Field(default=0, ge=0)
    frame_count: int = Field(default=0, ge=0)
    imu_sample_count: int = Field(default=0, ge=0)
    telemetry_queue_overflow_count: int = Field(default=0, ge=0)
    output: RecordingOutput = Field(default_factory=RecordingOutput)


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
    client_app_version: str = Field(default="0.1.0", min_length=1, max_length=128)


class CaptureSourceRevisions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    superproject_commit: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")
    glasses_commit: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")
    client_commit: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")


class CaptureRecordingProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: CaptureDeviceProvenance = Field(default_factory=CaptureDeviceProvenance)
    software: CaptureSoftwareProvenance = Field(default_factory=CaptureSoftwareProvenance)
    source_revisions: CaptureSourceRevisions = Field(default_factory=CaptureSourceRevisions)


class RecordingStorage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    video_path: Literal["video.mp4"] = "video.mp4"
    imu_path: Literal["imu.csv"] = "imu.csv"
    frames_path: Literal["frames.csv"] = "frames.csv"
    quality_path: Literal["quality.json"] = "quality.json"
    annotations_directory: Literal["annotations"] = "annotations"
    derived_directory: Literal["derived"] = "derived"


class RecordingTimeOrigin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clock_id: Literal["client_monotonic_ns"] = "client_monotonic_ns"
    recording_time_zero_event: Literal["countdown_started"] = "countdown_started"
    countdown_started_at_unix_ns: int = Field(ge=0)
    countdown_started_at_client_monotonic_ns: int = Field(ge=0)
    first_video_frame_recording_time_ns: int | None = Field(default=None, ge=0)
    last_video_frame_recording_time_ns: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_video_span(self) -> RecordingTimeOrigin:
        if (
            self.first_video_frame_recording_time_ns is not None
            and self.last_video_frame_recording_time_ns is not None
            and self.last_video_frame_recording_time_ns < self.first_video_frame_recording_time_ns
        ):
            raise ValueError("last video frame cannot precede first video frame")
        return self


class RecordingFileInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Literal["video.mp4", "imu.csv", "frames.csv", "quality.json"]
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_hash_pair(self) -> RecordingFileInfo:
        if (self.size_bytes is None) != (self.sha256 is None):
            raise ValueError("file size and SHA256 must be populated together")
        return self


class RecordingArtifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video: RecordingFileInfo = Field(default_factory=lambda: RecordingFileInfo(path="video.mp4"))
    imu: RecordingFileInfo = Field(default_factory=lambda: RecordingFileInfo(path="imu.csv"))
    frames: RecordingFileInfo = Field(default_factory=lambda: RecordingFileInfo(path="frames.csv"))
    quality: RecordingFileInfo = Field(
        default_factory=lambda: RecordingFileInfo(path="quality.json")
    )


class CaptureRecordingManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    contract_id: Literal["capture-recording-v1"] = "capture-recording-v1"
    recording_id: str = Field(pattern=RECORDING_ID_PATTERN)
    state: CaptureRecordingState
    started_at_unix_ns: int = Field(ge=0)
    ended_at_unix_ns: int | None = Field(default=None, ge=0)
    duration_ns: int | None = Field(default=None, ge=0)
    video_profile: RecordingOutput = Field(default_factory=RecordingOutput)
    frame_count: int = Field(default=0, ge=0)
    imu_sample_count: int = Field(default=0, ge=0)
    time_origin: RecordingTimeOrigin
    provenance: CaptureRecordingProvenance = Field(default_factory=CaptureRecordingProvenance)
    storage: RecordingStorage = Field(default_factory=RecordingStorage)
    artifacts: RecordingArtifacts = Field(default_factory=RecordingArtifacts)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> CaptureRecordingManifest:
        if self.ended_at_unix_ns is not None and self.ended_at_unix_ns < self.started_at_unix_ns:
            raise ValueError("recording end cannot precede start")
        if self.state is CaptureRecordingState.COMPLETE:
            if self.ended_at_unix_ns is None or self.duration_ns is None:
                raise ValueError("complete recording requires end time and duration")
            if self.frame_count < 1:
                raise ValueError("complete recording requires at least one video frame")
            if any(
                item.sha256 is None
                for item in (
                    self.artifacts.video,
                    self.artifacts.imu,
                    self.artifacts.frames,
                    self.artifacts.quality,
                )
            ):
                raise ValueError("complete recording requires hashes for every artifact")
        return self


class FrameMetadataMatchStatus(StrEnum):
    EXACT = "exact"
    WITHIN_TOLERANCE = "within_tolerance"
    UNMATCHED = "unmatched"


class RecordingFrameRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    frame_index: int = Field(ge=0)
    recording_time_ns: int = Field(ge=0)
    mp4_pts: int = Field(ge=0)
    mp4_time_base_num: int = Field(gt=0)
    mp4_time_base_den: int = Field(gt=0)
    connection_session_id: str = Field(pattern=CONNECTION_SESSION_ID_PATTERN)
    frame_id: int | None = Field(default=None, ge=0)
    camera_start_generation: int | None = Field(default=None, ge=1)
    captured_at_rokid_sdk_ms: int | None = Field(default=None, ge=0)
    received_at_elapsed_realtime_ns: int | None = Field(default=None, ge=0)
    video_at_monotonic_ns: int | None = Field(default=None, ge=0)
    rtp_timestamp_90khz: int | None = Field(default=None, ge=0, le=0xFFFFFFFF)
    received_at_client_monotonic_ns: int = Field(ge=0)
    metadata_match_status: FrameMetadataMatchStatus
    timestamp_match_error_90khz: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_metadata_match(self) -> RecordingFrameRow:
        metadata_values = (
            self.frame_id,
            self.camera_start_generation,
            self.captured_at_rokid_sdk_ms,
            self.received_at_elapsed_realtime_ns,
            self.video_at_monotonic_ns,
            self.rtp_timestamp_90khz,
        )
        if self.metadata_match_status is FrameMetadataMatchStatus.UNMATCHED:
            if any(value is not None for value in metadata_values):
                raise ValueError("unmatched frame cannot contain matched metadata")
            if self.timestamp_match_error_90khz is not None:
                raise ValueError("unmatched frame cannot contain a match error")
        elif any(value is None for value in metadata_values):
            raise ValueError("matched frame requires all Rokid and RTP metadata")
        elif self.timestamp_match_error_90khz is None:
            raise ValueError("matched frame requires a timestamp match error")
        elif (
            self.metadata_match_status is FrameMetadataMatchStatus.EXACT
            and self.timestamp_match_error_90khz != 0
        ):
            raise ValueError("exact frame match must have zero timestamp error")
        elif (
            self.metadata_match_status is FrameMetadataMatchStatus.WITHIN_TOLERANCE
            and self.timestamp_match_error_90khz == 0
        ):
            raise ValueError("within-tolerance match must have nonzero timestamp error")
        return self


class ImuSensorType(StrEnum):
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"


class RecordingImuRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    sample_index: int = Field(ge=0)
    recording_time_ns: int = Field(ge=0)
    connection_session_id: str = Field(pattern=CONNECTION_SESSION_ID_PATTERN)
    sensor_type: ImuSensorType
    sequence_number: int = Field(ge=0)
    sensor_event_monotonic_ns: int = Field(ge=0)
    received_at_elapsed_realtime_ns: int = Field(ge=0)
    received_at_client_monotonic_ns: int = Field(ge=0)
    accuracy: int = Field(ge=-1, le=3)
    x: float
    y: float
    z: float
    inside_video_span: bool


class RecordingQualityCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_frame_count: int = Field(ge=0)
    matched_video_frame_count: int = Field(ge=0)
    imu_sample_count: int = Field(ge=0)
    imu_inside_video_span_count: int = Field(ge=0)
    accelerometer_sample_count: int = Field(ge=0)
    gyroscope_sample_count: int = Field(ge=0)
    imu_sequence_gap_count: int = Field(ge=0)
    imu_duplicate_sample_count: int = Field(ge=0)
    imu_out_of_order_sample_count: int = Field(ge=0)
    telemetry_queue_overflow_count: int = Field(ge=0)


class RecordingQualityCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str = Field(min_length=1, max_length=128)
    status: Literal["pass", "warn", "fail", "not_evaluated"]
    metric_value: float | None = None
    threshold: float | None = None
    unit: str | None = None
    evidence: str = Field(max_length=512)


class CaptureRecordingQualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    contract_id: Literal["capture-recording-quality-v1"] = "capture-recording-quality-v1"
    recording_id: str = Field(pattern=RECORDING_ID_PATTERN)
    generated_at_unix_ns: int = Field(ge=0)
    status: Literal["pass", "warn", "fail", "incomplete"]
    counts: RecordingQualityCounts
    frame_metadata_coverage: float | None = Field(default=None, ge=0, le=1)
    timestamp_mapping_residual_ns: int | None = Field(default=None, ge=0)
    checks: list[RecordingQualityCheck] = Field(default_factory=list)


class RecordingSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recording_id: str = Field(pattern=RECORDING_ID_PATTERN)
    recorded_at_unix_ns: int = Field(ge=0)
    ended_at_unix_ns: int = Field(ge=0)
    duration_ns: int = Field(ge=0)
    width: int = Field(gt=0, le=8192)
    height: int = Field(gt=0, le=8192)
    fps: float = Field(gt=0, le=240)
    file_size_bytes: int = Field(gt=0)
    frame_count: int = Field(ge=1)
    imu_sample_count: int = Field(ge=0)
    hashes_verified: bool


class RecordingLibrary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = "2.0"
    recordings: list[RecordingSummary]
