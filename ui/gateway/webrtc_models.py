from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class WebRtcPhase(StrEnum):
    IDLE = "idle"
    NEGOTIATING = "negotiating"
    CONNECTED = "connected"
    STREAMING = "streaming"
    DISCONNECTED = "disconnected"
    FAILED = "failed"


class StreamControlAction(StrEnum):
    START = "start"
    STOP = "stop"


class StreamControlState(StrEnum):
    UNAVAILABLE = "unavailable"
    READY = "ready"
    STARTING = "starting"
    STREAMING = "streaming"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class StreamControlCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["1.0"] = "1.0"
    message_type: Literal["stream_control_command"] = "stream_control_command"
    command_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    action: StreamControlAction


class StreamControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["start", "stop"]


class StreamControlStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["1.0"] = "1.0"
    message_type: Literal["stream_control_status"] = "stream_control_status"
    command_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    state: StreamControlState
    detail: str | None = Field(default=None, max_length=256)


class RecordingControlAction(StrEnum):
    START = "start"
    STOP = "stop"


class RecordingControlState(StrEnum):
    UNAVAILABLE = "unavailable"
    READY = "ready"
    STARTING_STREAM = "starting_stream"
    COUNTDOWN = "countdown"
    RECORDING = "recording"
    FINALIZING = "finalizing"
    ERROR = "error"


class RecordingControlCommand(BaseModel):
    """Strict command sent by the Glass3 wearer over recording-control-v1."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["1.0"] = "1.0"
    message_type: Literal["recording_control_command"] = "recording_control_command"
    command_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    action: RecordingControlAction
    trigger: Literal["temple_double_tap"]
    requested_at_elapsed_realtime_ns: int = Field(ge=0)


class RecordingControlStatus(BaseModel):
    """Authoritative recording state sent to the Glass3 HUD."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["1.0"] = "1.0"
    message_type: Literal["recording_control_status"] = "recording_control_status"
    command_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    state: RecordingControlState
    recording_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    countdown_remaining_ms: int | None = Field(default=None, ge=0)
    recording_duration_ms: int = Field(default=0, ge=0)
    frame_count: int = Field(default=0, ge=0)
    imu_sample_count: int = Field(default=0, ge=0)
    detail: str | None = Field(default=None, max_length=256)


class WebRtcOffer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    device_session_id: str = Field(
        min_length=16,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    type: Literal["offer"] = "offer"
    sdp: str = Field(min_length=16, max_length=1_048_576)


class WebRtcAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    session_id: str = Field(
        min_length=16,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    type: Literal["answer"] = "answer"
    sdp: str = Field(min_length=16, max_length=1_048_576)


class VideoFrameMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    message_type: Literal["video_frame"] = "video_frame"
    stream_id: Literal["camera"] = "camera"
    frame_id: int = Field(ge=0)
    camera_start_generation: int = Field(ge=1)
    captured_at_rokid_sdk_ms: int = Field(ge=0)
    received_at_elapsed_realtime_ns: int = Field(ge=0)
    video_at_monotonic_ns: int = Field(ge=0)
    rtp_timestamp_90khz: int = Field(ge=0, le=0xFFFFFFFF)
    width: int = Field(ge=1, le=8192)
    height: int = Field(ge=1, le=8192)
    rotation_degrees: Literal[0, 90, 180, 270] = 0
    capture_config_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )


class ImuSensorType(StrEnum):
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"


class ImuChannelState(StrEnum):
    UNAVAILABLE = "unavailable"
    READY = "ready"
    RECEIVING = "receiving"


class ImuSensorDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    sensor_type: ImuSensorType
    android_sensor_type: Literal[1, 4]
    name: str = Field(min_length=1, max_length=128)
    vendor: str = Field(max_length=128)
    version: int = Field(ge=0)
    unit: Literal["m_s2", "rad_s"]
    resolution: float = Field(ge=0)
    max_range: float = Field(gt=0)
    min_delay_us: int = Field(ge=0)
    max_delay_us: int = Field(ge=0)
    is_wake_up: bool

    @model_validator(mode="after")
    def validate_sensor_mapping(self) -> ImuSensorDescriptor:
        expected = {
            ImuSensorType.ACCELEROMETER: (1, "m_s2"),
            ImuSensorType.GYROSCOPE: (4, "rad_s"),
        }[self.sensor_type]
        if (self.android_sensor_type, self.unit) != expected:
            raise ValueError("sensor type, Android type, and unit do not match")
        return self


class ImuCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    schema_version: Literal["0.1"] = "0.1"
    message_type: Literal["imu_capabilities"] = "imu_capabilities"
    source: Literal["android_sensor_manager"] = "android_sensor_manager"
    requested_sampling_period_us: int = Field(ge=5_000, le=1_000_000)
    sensors: list[ImuSensorDescriptor] = Field(max_length=2)
    missing_sensor_types: list[ImuSensorType] = Field(max_length=2)

    @model_validator(mode="after")
    def validate_requested_sensor_coverage(self) -> ImuCapabilities:
        available = [descriptor.sensor_type for descriptor in self.sensors]
        missing = self.missing_sensor_types
        if len(set(available)) != len(available) or len(set(missing)) != len(missing):
            raise ValueError("IMU sensor types must be unique")
        if set(available).intersection(missing):
            raise ValueError("IMU sensor cannot be both available and missing")
        if set(available).union(missing) != set(ImuSensorType):
            raise ValueError("capabilities must cover both requested IMU sensor types")
        return self


class ImuSample(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    schema_version: Literal["0.1"] = "0.1"
    message_type: Literal["imu_sample"] = "imu_sample"
    sensor_type: ImuSensorType
    android_sensor_type: Literal[1, 4]
    sequence_number: int = Field(ge=0)
    sensor_event_monotonic_ns: int = Field(ge=0)
    received_at_elapsed_realtime_ns: int = Field(ge=0)
    accuracy: int = Field(ge=-1, le=3)
    values: tuple[float, float, float]

    @model_validator(mode="after")
    def validate_android_sensor_type(self) -> ImuSample:
        expected = {
            ImuSensorType.ACCELEROMETER: 1,
            ImuSensorType.GYROSCOPE: 4,
        }[self.sensor_type]
        if self.android_sensor_type != expected:
            raise ValueError("sensor type and Android type do not match")
        return self


ImuTelemetryMessage = Annotated[
    ImuCapabilities | ImuSample,
    Field(discriminator="message_type"),
]
IMU_TELEMETRY_ADAPTER = TypeAdapter(ImuTelemetryMessage)


class ImuSensorStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_count: int = Field(default=0, ge=0)
    observed_rate_hz: float | None = Field(default=None, ge=0)
    first_received_at_perf_counter_ns: int | None = Field(default=None, ge=0)
    last_received_at_perf_counter_ns: int | None = Field(default=None, ge=0)
    latest_sequence_number: int | None = Field(default=None, ge=0)
    sequence_gaps: int = Field(default=0, ge=0)
    out_of_order_samples: int = Field(default=0, ge=0)
    last_event_to_callback_delta_ns: int | None = None
    min_event_to_callback_delta_ns: int | None = None
    max_event_to_callback_delta_ns: int | None = None
    last_sample: ImuSample | None = None


class ImuTelemetryStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1"] = "0.1"
    connection_session_id: str | None = None
    device_session_id: str | None = None
    channel_state: ImuChannelState = ImuChannelState.UNAVAILABLE
    messages_received: int = Field(default=0, ge=0)
    capabilities_received: int = Field(default=0, ge=0)
    samples_received: int = Field(default=0, ge=0)
    malformed_messages: int = Field(default=0, ge=0)
    capabilities: ImuCapabilities | None = None
    sensors: dict[ImuSensorType, ImuSensorStatus]


class WebRtcStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    phase: WebRtcPhase
    connection_session_id: str | None = None
    device_session_id: str | None = None
    connection_state: str | None = None
    frames_received: int = Field(default=0, ge=0)
    metadata_received: int = Field(default=0, ge=0)
    metadata_matched: int = Field(default=0, ge=0)
    metadata_anchor_matches: int = Field(default=0, ge=0)
    metadata_ordered_gap_matches: int = Field(default=0, ge=0)
    malformed_metadata: int = Field(default=0, ge=0)
    duplicate_metadata: int = Field(default=0, ge=0)
    unmatched_entries_dropped: int = Field(default=0, ge=0)
    sdk_clock_discontinuities: int = Field(default=0, ge=0)
    pending_frames: int = Field(default=0, ge=0)
    pending_metadata: int = Field(default=0, ge=0)
    max_timestamp_match_error_90khz: int = Field(default=0, ge=0, le=6_000)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    video_codec: str | None = Field(default=None, pattern=r"^[A-Z0-9.-]+$")
    average_fps: float | None = Field(default=None, ge=0)
    rtp_packets_received: int = Field(default=0, ge=0)
    rtp_packets_lost: int = Field(default=0, ge=0)
    rtp_packet_loss_percent: float = Field(default=0.0, ge=0, le=100)
    rtp_jitter_ms: float = Field(default=0.0, ge=0)
    corrupt_frames_dropped: int = Field(default=0, ge=0)
    first_frame_latency_ms: float | None = Field(default=None, ge=0)
    last_frame_pts: int | None = None
    last_frame_time_base_num: int | None = None
    last_frame_time_base_den: int | None = None
    metadata_rtp_origin_90khz: int | None = Field(default=None, ge=0, le=0xFFFFFFFF)
    metadata_calibrated: bool = False
    metadata_calibration_support: int = Field(default=0, ge=0)
    last_frame_received_at_perf_counter_ns: int | None = Field(default=None, ge=0)
    last_error: str | None = None
