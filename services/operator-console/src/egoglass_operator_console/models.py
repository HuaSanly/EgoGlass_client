from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SessionPhase(StrEnum):
    IDLE = "idle"
    CONNECTING = "connecting"
    LIVE = "live"
    DEGRADED = "degraded"
    DISCONNECTED = "disconnected"


class CalibrationState(StrEnum):
    MISSING = "missing"
    SIMULATED = "simulated"
    VERIFIED = "verified"
    INVALID = "invalid"


class HandSide(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_width: int = Field(default=1280, ge=320, le=3840)
    video_height: int = Field(default=720, ge=240, le=2160)
    capture_fps: int = Field(default=20, ge=5, le=30)
    target_bitrate_kbps: int = Field(default=2500, ge=500, le=10_000)
    inference_fps: int = Field(default=10, ge=1, le=30)
    history_frames: int = Field(default=8, ge=1, le=32)
    prediction_steps: int = Field(default=10, ge=1, le=30)
    prediction_interval_ms: int = Field(default=100, ge=50, le=1000)
    max_feedback_age_ms: int = Field(default=500, ge=100, le=3000)
    recording_segment_seconds: int = Field(default=60, ge=10, le=600)
    session_limit_minutes: int = Field(default=30, ge=0, le=240)
    min_free_disk_gb: int = Field(default=20, ge=1, le=1000)
    retain_recordings: bool = True

    @model_validator(mode="after")
    def validate_rates(self) -> RuntimeSettings:
        if self.inference_fps > self.capture_fps:
            raise ValueError("inference_fps must not exceed capture_fps")
        return self


class CalibrationSummary(BaseModel):
    profile_id: str
    state: CalibrationState
    coordinate_frame: str = "camera_optical"
    units: str = "meters"
    reprojection_error_px: float | None = Field(default=None, ge=0)
    verified_at_unix_ns: int | None = Field(default=None, ge=0)


class Waypoint3D(BaseModel):
    t_offset_ms: int = Field(ge=0)
    x_m: float
    y_m: float
    z_m: float = Field(gt=0)
    confidence: float = Field(ge=0, le=1)


class HandTrajectory(BaseModel):
    side: HandSide
    present: bool
    confidence: float = Field(ge=0, le=1)
    waypoints: list[Waypoint3D]


class RuntimeMetrics(BaseModel):
    capture_fps: float = Field(ge=0)
    inference_fps: float = Field(ge=0)
    media_latency_ms: float = Field(ge=0)
    inference_latency_ms: float = Field(ge=0)
    feedback_latency_ms: float = Field(ge=0)
    dropped_frames: int = Field(ge=0)
    queue_depth: int = Field(ge=0)
    gpu_memory_gb: float = Field(ge=0)


class TelemetrySnapshot(BaseModel):
    schema_version: str = "1.0"
    session_id: str
    source: str = "synthetic"
    session_phase: SessionPhase
    recording: bool
    frame_seq: int = Field(ge=0)
    captured_at_sdk_ms: int = Field(ge=0)
    received_at_perf_counter_ns: int = Field(ge=0)
    generated_at_unix_ns: int = Field(ge=0)
    max_feedback_age_ms: int = Field(ge=0)
    calibration: CalibrationSummary
    metrics: RuntimeMetrics
    hands: list[HandTrajectory]


class ConsoleState(BaseModel):
    service_version: str
    mode: str = "simulation"
    session_id: str
    session_phase: SessionPhase
    recording: bool
    settings_revision: int = Field(ge=1)
    settings: RuntimeSettings
    calibration: CalibrationSummary
