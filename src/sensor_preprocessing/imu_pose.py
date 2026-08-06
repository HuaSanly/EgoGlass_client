from __future__ import annotations

from bisect import bisect_right
from collections import deque
from dataclasses import dataclass

from .clock_mapping import SegmentedClockMapper, imu_sensor_event_observation
from .imu_orientation import (
    ImuOrientationConfig,
    ImuOrientationFilter,
    quaternion_to_euler_degrees,
)
from .models import ImuSensorType, RawImuSample

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
    orientation_config: ImuOrientationConfig | None = None,
    beta: float | None = None,
    raw_imu_to_body_axes: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    gyroscope_bias_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
    accelerometer_bias_m_s2: tuple[float, float, float] = (0.0, 0.0, 0.0),
    accelerometer_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> RecordedImuPoseTimeline:
    """Fuse mapped raw IMU using the same implementation as live preview."""

    mapped_samples = sorted(
        ((_sample_session_time_ns(sample, mapper), sample) for sample in samples),
        key=lambda item: (item[0], item[1].sample_id),
    )
    effective_config = orientation_config or ImuOrientationConfig()
    if beta is not None:
        effective_config = effective_config.model_copy(update={"madgwick_beta": beta})
    orientation = ImuOrientationFilter(
        effective_config,
        raw_imu_to_body_axes=raw_imu_to_body_axes,
        gyroscope_bias_rad_s=gyroscope_bias_rad_s,
        accelerometer_bias_m_s2=accelerometer_bias_m_s2,
        accelerometer_scale=accelerometer_scale,
    )
    gyro_times_ns: deque[int] = deque(maxlen=400)
    poses: list[RecordedImuPose] = []

    for processed, (session_time_ns, sample) in enumerate(mapped_samples, start=1):
        accepted = orientation.process(
            sample.sensor_type,
            sample.values,
            sample.sensor_event_monotonic_ns,
        )
        if sample.sensor_type is not ImuSensorType.GYROSCOPE or not accepted:
            continue
        gyro_times_ns.append(sample.sensor_event_monotonic_ns)
        quaternion = orientation.relative_quaternion_wxyz
        roll, pitch, yaw = quaternion_to_euler_degrees(quaternion)
        pose = RecordedImuPose(
            session_id=session_id,
            session_time_ns=session_time_ns,
            quaternion_wxyz=quaternion,
            roll_degrees=roll,
            pitch_degrees=pitch,
            yaw_degrees=yaw,
            accelerometer_mps2=orientation.accelerometer_mps2,
            gyroscope_radps=orientation.gyroscope_radps,
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
