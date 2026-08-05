from __future__ import annotations

import math
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass

import numpy as np
from ahrs.common.orientation import acc2q
from ahrs.filters import Madgwick

from .clock_mapping import SegmentedClockMapper, imu_sensor_event_observation
from .models import ImuSensorType, RawImuSample

_INITIAL_ACCELEROMETER_SAMPLES = 20
_RATE_WINDOW_NS = 2_000_000_000


@dataclass(frozen=True, slots=True)
class RecordedImuPose:
    """One fused glasses orientation addressed on the capture-session clock."""

    session_id: str
    session_time_ns: int
    quaternion_wxyz: tuple[float, float, float, float]
    roll_degrees: float
    pitch_degrees: float
    yaw_degrees: float
    accelerometer_mps2: tuple[float, float, float] | None
    gyroscope_radps: tuple[float, float, float] | None
    samples_received: int
    samples_processed: int
    queue_overflow_count: int = 0
    recent_rate_hz: float = 0.0
    latest_sample_age_ms: float | None = None


class RecordedImuPoseTimeline:
    """Random-access IMU orientation timeline built once when a session is opened."""

    def __init__(self, poses: tuple[RecordedImuPose, ...]) -> None:
        if any(
            current.session_time_ns <= previous.session_time_ns
            for previous, current in zip(poses, poses[1:], strict=False)
        ):
            raise ValueError("recorded IMU poses must use strictly increasing session time")
        self.poses = poses
        self._times_ns = tuple(pose.session_time_ns for pose in poses)

    def pose_at(self, session_time_ns: int) -> RecordedImuPose | None:
        index = bisect_right(self._times_ns, session_time_ns) - 1
        return None if index < 0 else self.poses[index]

    def __len__(self) -> int:
        return len(self.poses)


def build_recorded_imu_pose_timeline(
    session_id: str,
    samples: tuple[RawImuSample, ...],
    mapper: SegmentedClockMapper | None,
    *,
    beta: float = 0.05,
) -> RecordedImuPoseTimeline:
    """Fuse mapped raw IMU into poses that can be queried by video session time."""

    mapped_samples = sorted(
        (
            (_sample_session_time_ns(sample, mapper), sample)
            for sample in samples
        ),
        key=lambda item: (item[0], item[1].sample_id),
    )
    filter_ = Madgwick(frequency=100.0, beta=beta)
    quaternion = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    reference: np.ndarray | None = None
    initialized = False
    accelerometer: tuple[float, float, float] | None = None
    gyroscope: tuple[float, float, float] | None = None
    accelerometer_window: deque[np.ndarray] = deque(
        maxlen=_INITIAL_ACCELEROMETER_SAMPLES
    )
    gyro_times_ns: deque[int] = deque(maxlen=400)
    last_gyro_timestamp_ns: int | None = None
    poses: list[RecordedImuPose] = []

    for processed, (session_time_ns, sample) in enumerate(mapped_samples, start=1):
        if sample.sensor_type is ImuSensorType.ACCELEROMETER:
            accelerometer = sample.values
            accelerometer_window.append(np.asarray(sample.values, dtype=np.float64))
            continue

        current_gyro_ns = sample.sensor_event_monotonic_ns
        if last_gyro_timestamp_ns is not None and current_gyro_ns <= last_gyro_timestamp_ns:
            continue
        previous_gyro_ns = last_gyro_timestamp_ns
        last_gyro_timestamp_ns = current_gyro_ns
        gyroscope = sample.values
        gyro_times_ns.append(current_gyro_ns)

        if accelerometer is not None and previous_gyro_ns is not None:
            if not initialized:
                if len(accelerometer_window) >= _INITIAL_ACCELEROMETER_SAMPLES:
                    initial = acc2q(np.mean(tuple(accelerometer_window), axis=0))
                    if np.isfinite(initial).all():
                        quaternion = _normalized_quaternion(initial)
                        reference = quaternion.copy()
                        initialized = True
            else:
                delta_seconds = (current_gyro_ns - previous_gyro_ns) / 1_000_000_000
                if 0.0001 <= delta_seconds <= 0.2:
                    updated = filter_.updateIMU(
                        quaternion,
                        gyr=np.asarray(gyroscope, dtype=np.float64),
                        acc=np.asarray(accelerometer, dtype=np.float64),
                        dt=delta_seconds,
                    )
                    if updated is not None and np.isfinite(updated).all():
                        quaternion = _normalized_quaternion(updated)

        relative = _relative_quaternion(reference, quaternion)
        quaternion_wxyz = tuple(float(value) for value in relative)
        roll, pitch, yaw = _quaternion_to_euler_degrees(quaternion_wxyz)
        pose = RecordedImuPose(
            session_id=session_id,
            session_time_ns=session_time_ns,
            quaternion_wxyz=quaternion_wxyz,
            roll_degrees=roll,
            pitch_degrees=pitch,
            yaw_degrees=yaw,
            accelerometer_mps2=accelerometer,
            gyroscope_radps=gyroscope,
            samples_received=processed,
            samples_processed=processed,
            recent_rate_hz=round(_recent_rate_hz(gyro_times_ns), 3),
        )
        if poses and poses[-1].session_time_ns == session_time_ns:
            poses[-1] = pose
        else:
            poses.append(pose)
    return RecordedImuPoseTimeline(tuple(poses))


def _sample_session_time_ns(
    sample: RawImuSample,
    mapper: SegmentedClockMapper | None,
) -> int:
    session_time_ns = sample.stored_alignment.session_time_ns
    if session_time_ns is not None:
        return session_time_ns
    if mapper is None:
        raise ValueError("recorded IMU sample has no session-time mapping")
    estimate = mapper.map(imu_sensor_event_observation(sample))
    if estimate.session_time_ns is None:
        raise ValueError("recorded IMU sample has no session-time mapping")
    return estimate.session_time_ns


def _recent_rate_hz(gyro_times_ns: deque[int]) -> float:
    if len(gyro_times_ns) < 2:
        return 0.0
    cutoff_ns = gyro_times_ns[-1] - _RATE_WINDOW_NS
    recent = tuple(value for value in gyro_times_ns if value >= cutoff_ns)
    if len(recent) < 2 or recent[-1] <= recent[0]:
        return 0.0
    return (len(recent) - 1) * 1_000_000_000 / (recent[-1] - recent[0])


def _relative_quaternion(reference: np.ndarray | None, current: np.ndarray) -> np.ndarray:
    if reference is None:
        return np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    reference_inverse = _normalized_quaternion(reference).copy()
    reference_inverse[1:] *= -1.0
    return _normalized_quaternion(
        _quaternion_product(reference_inverse, _normalized_quaternion(current))
    )


def _normalized_quaternion(quaternion: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= 0.0:
        return np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    return values / norm


def _quaternion_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
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


def _quaternion_to_euler_degrees(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch_term = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(pitch_term)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))
