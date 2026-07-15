from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class WebRtcPhase(StrEnum):
    IDLE = "idle"
    NEGOTIATING = "negotiating"
    CONNECTED = "connected"
    STREAMING = "streaming"
    DISCONNECTED = "disconnected"
    FAILED = "failed"


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


class WebRtcStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    phase: WebRtcPhase
    session_id: str | None = None
    device_session_id: str | None = None
    connection_state: str | None = None
    frames_received: int = Field(default=0, ge=0)
    metadata_received: int = Field(default=0, ge=0)
    metadata_matched: int = Field(default=0, ge=0)
    malformed_metadata: int = Field(default=0, ge=0)
    duplicate_metadata: int = Field(default=0, ge=0)
    unmatched_entries_dropped: int = Field(default=0, ge=0)
    sdk_clock_discontinuities: int = Field(default=0, ge=0)
    pending_frames: int = Field(default=0, ge=0)
    pending_metadata: int = Field(default=0, ge=0)
    max_timestamp_match_error_90khz: int = Field(default=0, ge=0, le=90)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    video_codec: str | None = Field(default=None, pattern=r"^[A-Z0-9.-]+$")
    average_fps: float | None = Field(default=None, ge=0)
    first_frame_latency_ms: float | None = Field(default=None, ge=0)
    last_frame_pts: int | None = None
    last_frame_time_base_num: int | None = None
    last_frame_time_base_den: int | None = None
    metadata_rtp_origin_90khz: int | None = Field(default=None, ge=0, le=0xFFFFFFFF)
    last_frame_received_at_perf_counter_ns: int | None = Field(default=None, ge=0)
    last_error: str | None = None
