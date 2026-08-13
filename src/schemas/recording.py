"""Typed contracts for one calibration-ready capture recording."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RECORDING_ID_PATTERN = r"^[0-9a-f]{32}$"


class RecordingState(StrEnum):
    UNAVAILABLE = "unavailable"
    READY = "ready"
    COUNTDOWN = "countdown"
    RECORDING = "recording"
    FINALIZING = "finalizing"
    ERROR = "error"


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

    schema_version: Literal["3.0"] = "3.0"
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


class CameraFrameRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    frame_idx: int = Field(ge=0)
    frame_id: int = Field(ge=0)
    rokid_timestamp_ns: int = Field(ge=0)
    device_monotonic_ns: int = Field(ge=0)


class ImuSensorType(StrEnum):
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"


class RecordingImuRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    sensor_type: ImuSensorType
    sequence: int = Field(ge=0)
    timestamp_ns: int = Field(ge=0)
    x: float
    y: float
    z: float


class CameraCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    model: Literal["pinhole"] = "pinhole"
    resolution: tuple[int, int]
    intrinsics: tuple[float, float, float, float] = (1.0, 1.0, 0.0, 0.0)
    distortion_model: Literal["radtan"] = "radtan"
    distortion_coeffs: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @model_validator(mode="after")
    def validate_resolution(self) -> CameraCalibration:
        if any(value <= 0 or value > 8192 for value in self.resolution):
            raise ValueError("calibration resolution must be between 1 and 8192")
        return self


class ImuCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    gyro_noise_density: float | None = Field(default=None, ge=0)
    gyro_random_walk: float | None = Field(default=None, ge=0)
    accel_noise_density: float | None = Field(default=None, ge=0)
    accel_random_walk: float | None = Field(default=None, ge=0)


class CalibrationSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    camera: CameraCalibration
    T_cam_imu: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    imu: ImuCalibration = Field(default_factory=ImuCalibration)

    @classmethod
    def placeholder(cls, width: int, height: int) -> CalibrationSnapshot:
        return cls(
            camera=CameraCalibration(resolution=(width, height)),
            T_cam_imu=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )

    @model_validator(mode="after")
    def validate_transform(self) -> CalibrationSnapshot:
        if any(not math.isfinite(value) for row in self.T_cam_imu for value in row):
            raise ValueError("T_cam_imu must contain finite values")
        if self.T_cam_imu[3] != (0.0, 0.0, 0.0, 1.0):
            raise ValueError("T_cam_imu must be a homogeneous transform")
        return self


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
    imu_sample_count: int = Field(ge=1)
    camera_frame_gap_count: int = Field(default=0, ge=0)
    imu_sequence_gap_count: int = Field(default=0, ge=0)
    protocol_validated: bool = True


class RecordingLibrary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["3.0"] = "3.0"
    recordings: list[RecordingSummary]
