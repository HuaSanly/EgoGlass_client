"""Shared IMU orientation fusion for live preview and recorded replay."""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from ahrs.common.orientation import acc2q
from ahrs.filters import Madgwick
from pydantic import BaseModel, ConfigDict, Field

from .models import ImuSensorType

_IDENTITY_QUATERNION = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
_IDENTITY_AXES = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


class ImuOrientationConfig(BaseModel):
    """Tunable fusion values; measured sensor quantities remain in calibration JSON."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    madgwick_beta: float = Field(default=0.05, ge=0.0, le=1.0)
    initial_static_samples: int = Field(default=20, ge=3, le=2_000)
    static_gyro_threshold_rad_s: float = Field(default=0.05, gt=0.0)
    gravity_magnitude_m_s2: float = Field(default=9.80665, gt=0.0)
    gravity_tolerance_m_s2: float = Field(default=1.25, gt=0.0)
    quaternion_smoothing_time_constant_s: float = Field(default=0.025, ge=0.0)
    minimum_sample_interval_s: float = Field(default=0.0001, gt=0.0)
    maximum_sample_interval_s: float = Field(default=0.2, gt=0.0)


class ImuOrientationFilter:
    """Fuse calibrated accelerometer and gyro samples into a relative quaternion.

    Raw samples are first mapped into the configured body axes. Accelerometer
    correction is disabled while its norm is inconsistent with gravity, so
    translational motion cannot directly pull the orientation estimate.
    """

    def __init__(
        self,
        config: ImuOrientationConfig | None = None,
        *,
        raw_imu_to_body_axes: tuple[tuple[float, float, float], ...] = _IDENTITY_AXES,
        gyroscope_bias_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
        accelerometer_bias_m_s2: tuple[float, float, float] = (0.0, 0.0, 0.0),
        accelerometer_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        self.config = config or ImuOrientationConfig()
        self._axes = _validated_rotation(raw_imu_to_body_axes)
        self._fixed_gyro_bias = _validated_vector(gyroscope_bias_rad_s, "gyroscope bias")
        self._accelerometer_bias = _validated_vector(
            accelerometer_bias_m_s2, "accelerometer bias"
        )
        self._accelerometer_scale = _validated_vector(
            accelerometer_scale, "accelerometer scale"
        )
        if np.any(self._accelerometer_scale <= 0.0):
            raise ValueError("accelerometer scale values must be positive")
        self._madgwick = Madgwick(frequency=100.0, beta=self.config.madgwick_beta)
        self.reset()

    def reset(self) -> None:
        """Clear fusion, bias-estimation, timestamp, and relative-origin state."""

        self._quaternion = _IDENTITY_QUATERNION.copy()
        self._smoothed_quaternion = _IDENTITY_QUATERNION.copy()
        self._reference_quaternion: np.ndarray | None = None
        self._initialized = False
        self._accelerometer: np.ndarray | None = None
        self._gyroscope: np.ndarray | None = None
        self._accelerometer_window: deque[np.ndarray] = deque(
            maxlen=self.config.initial_static_samples
        )
        self._static_gyro_window: deque[np.ndarray] = deque(
            maxlen=self.config.initial_static_samples
        )
        self._estimated_gyro_bias = np.zeros(3, dtype=np.float64)
        self._last_gyro_timestamp_ns: int | None = None

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def last_gyro_timestamp_ns(self) -> int | None:
        return self._last_gyro_timestamp_ns

    @property
    def accelerometer_mps2(self) -> tuple[float, float, float] | None:
        return _optional_tuple(self._accelerometer)

    @property
    def gyroscope_radps(self) -> tuple[float, float, float] | None:
        return _optional_tuple(self._gyroscope)

    @property
    def estimated_gyroscope_bias_rad_s(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self._estimated_gyro_bias)

    @property
    def quaternion_wxyz(self) -> tuple[float, float, float, float]:
        return tuple(float(value) for value in self._smoothed_quaternion)

    @property
    def relative_quaternion_wxyz(self) -> tuple[float, float, float, float]:
        return tuple(
            float(value)
            for value in relative_quaternion(
                self._reference_quaternion,
                self._smoothed_quaternion,
            )
        )

    def reset_reference(self) -> None:
        """Make the current fused orientation the new display origin."""

        self._reference_quaternion = (
            normalized_quaternion(self._smoothed_quaternion)
            if self._initialized
            else None
        )

    def process(
        self,
        sensor_type: ImuSensorType,
        values: tuple[float, float, float],
        timestamp_ns: int,
    ) -> bool:
        """Consume one sample and return whether it was accepted."""

        if timestamp_ns < 0:
            return False
        raw = np.asarray(values, dtype=np.float64)
        if raw.shape != (3,) or not np.isfinite(raw).all():
            return False
        if sensor_type is ImuSensorType.ACCELEROMETER:
            calibrated = self._axes @ ((raw - self._accelerometer_bias) * self._accelerometer_scale)
            self._accelerometer = calibrated
            if self._gravity_is_reliable(calibrated):
                self._accelerometer_window.append(calibrated.copy())
            else:
                self._accelerometer_window.clear()
            return True
        if sensor_type is not ImuSensorType.GYROSCOPE:
            return False
        if (
            self._last_gyro_timestamp_ns is not None
            and timestamp_ns <= self._last_gyro_timestamp_ns
        ):
            return False

        previous_ns = self._last_gyro_timestamp_ns
        self._last_gyro_timestamp_ns = timestamp_ns
        calibrated_gyro = self._axes @ (raw - self._fixed_gyro_bias)
        self._update_static_bias(calibrated_gyro)
        self._gyroscope = calibrated_gyro - self._estimated_gyro_bias

        if not self._initialized:
            if len(self._accelerometer_window) >= self.config.initial_static_samples:
                initial = acc2q(np.mean(tuple(self._accelerometer_window), axis=0))
                if initial is not None and np.isfinite(initial).all():
                    self._quaternion = normalized_quaternion(initial)
                    self._smoothed_quaternion = self._quaternion.copy()
                    self._reference_quaternion = self._quaternion.copy()
                    self._initialized = True
            return True
        if previous_ns is None:
            return True

        delta_seconds = (timestamp_ns - previous_ns) / 1_000_000_000
        if not (
            self.config.minimum_sample_interval_s
            <= delta_seconds
            <= self.config.maximum_sample_interval_s
        ):
            return False

        updated = self._integrate(delta_seconds)
        if updated is None or not np.isfinite(updated).all():
            return False
        self._quaternion = normalized_quaternion(updated)
        self._smoothed_quaternion = slerp_quaternion(
            self._smoothed_quaternion,
            self._quaternion,
            _smoothing_alpha(
                delta_seconds,
                self.config.quaternion_smoothing_time_constant_s,
            ),
        )
        return True

    def _integrate(self, delta_seconds: float) -> np.ndarray | None:
        assert self._gyroscope is not None
        if self._accelerometer is not None and self._gravity_is_reliable(
            self._accelerometer
        ):
            return self._madgwick.updateIMU(
                self._quaternion,
                gyr=self._gyroscope,
                acc=self._accelerometer,
                dt=delta_seconds,
            )
        derivative = 0.5 * quaternion_product(
            self._quaternion,
            np.asarray((0.0, *self._gyroscope), dtype=np.float64),
        )
        return self._quaternion + derivative * delta_seconds

    def _update_static_bias(self, calibrated_gyro: np.ndarray) -> None:
        if (
            self._accelerometer is None
            or not self._gravity_is_reliable(self._accelerometer)
            or float(np.linalg.norm(calibrated_gyro))
            > self.config.static_gyro_threshold_rad_s
        ):
            self._static_gyro_window.clear()
            return
        self._static_gyro_window.append(calibrated_gyro.copy())
        if len(self._static_gyro_window) >= self.config.initial_static_samples:
            self._estimated_gyro_bias = np.mean(tuple(self._static_gyro_window), axis=0)

    def _gravity_is_reliable(self, accelerometer: np.ndarray) -> bool:
        return (
            abs(float(np.linalg.norm(accelerometer)) - self.config.gravity_magnitude_m_s2)
            <= self.config.gravity_tolerance_m_s2
        )


def relative_quaternion(reference: np.ndarray | None, current: np.ndarray) -> np.ndarray:
    if reference is None:
        return _IDENTITY_QUATERNION.copy()
    inverse = normalized_quaternion(reference).copy()
    inverse[1:] *= -1.0
    return normalized_quaternion(
        quaternion_product(inverse, normalized_quaternion(current))
    )


def normalized_quaternion(quaternion: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if values.shape != (4,) or not math.isfinite(norm) or norm <= 1e-12:
        return _IDENTITY_QUATERNION.copy()
    return values / norm


def quaternion_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def slerp_quaternion(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    start = normalized_quaternion(left)
    end = normalized_quaternion(right)
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return normalized_quaternion(start + fraction * (end - start))
    angle = math.acos(dot)
    sine = math.sin(angle)
    return normalized_quaternion(
        math.sin((1.0 - fraction) * angle) / sine * start
        + math.sin(fraction * angle) / sine * end
    )


def quaternion_to_euler_degrees(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch_term = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(pitch_term)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def _smoothing_alpha(delta_seconds: float, time_constant_seconds: float) -> float:
    if time_constant_seconds <= 0.0:
        return 1.0
    return 1.0 - math.exp(-delta_seconds / time_constant_seconds)


def _validated_vector(values: tuple[float, float, float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite values")
    return vector


def _validated_rotation(
    values: tuple[tuple[float, float, float], ...],
) -> np.ndarray:
    rotation = np.asarray(values, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("raw IMU axis mapping must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("raw IMU axis mapping must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("raw IMU axis mapping determinant must be +1")
    return rotation


def _optional_tuple(values: np.ndarray | None) -> tuple[float, float, float] | None:
    return None if values is None else tuple(float(value) for value in values)
