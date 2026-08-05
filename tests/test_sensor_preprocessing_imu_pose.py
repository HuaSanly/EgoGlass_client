from sensor_preprocessing import (
    AlignmentStatus,
    ImuSensorType,
    RawImuSample,
    StoredAlignment,
    build_recorded_imu_pose_timeline,
)


def _sample(
    sample_id: int,
    sensor_type: ImuSensorType,
    timestamp_ns: int,
    values: tuple[float, float, float],
) -> RawImuSample:
    return RawImuSample(
        sample_id=sample_id,
        session_id="session",
        connection_session_id="connection",
        sensor_type=sensor_type,
        android_sensor_type=(1 if sensor_type is ImuSensorType.ACCELEROMETER else 4),
        sequence_number=sample_id,
        sensor_event_monotonic_ns=1_000_000_000 + timestamp_ns,
        received_at_elapsed_realtime_ns=2_000_000_000 + timestamp_ns,
        received_at_client_perf_counter_ns=3_000_000_000 + timestamp_ns,
        accuracy=3,
        values=values,
        unit=("m_s2" if sensor_type is ImuSensorType.ACCELEROMETER else "rad_s"),
        stored_alignment=StoredAlignment(
            AlignmentStatus.MAPPED,
            timestamp_ns,
            100_000,
            "mapping",
        ),
    )


def test_recorded_imu_pose_timeline_is_random_access_on_session_time() -> None:
    samples: list[RawImuSample] = []
    sample_id = 1
    for index in range(80):
        timestamp_ns = 100_000_000 + index * 10_000_000
        samples.append(
            _sample(
                sample_id,
                ImuSensorType.ACCELEROMETER,
                timestamp_ns,
                (0.0, 0.0, 9.81),
            )
        )
        sample_id += 1
        samples.append(
            _sample(
                sample_id,
                ImuSensorType.GYROSCOPE,
                timestamp_ns + 1,
                (0.0, 0.0, 0.4),
            )
        )
        sample_id += 1

    timeline = build_recorded_imu_pose_timeline("session", tuple(samples), None)
    first = timeline.poses[0]
    final = timeline.poses[-1]

    assert timeline.pose_at(first.session_time_ns - 1) is None
    assert timeline.pose_at(first.session_time_ns) is first
    assert timeline.pose_at(final.session_time_ns + 1_000_000) is final
    assert final.samples_received == len(samples)
    assert final.recent_rate_hz == 100.0
    assert abs(final.yaw_degrees) > 5.0


def test_recorded_imu_pose_uses_latest_sample_not_future_sample() -> None:
    samples = (
        _sample(1, ImuSensorType.ACCELEROMETER, 10, (0.0, 0.0, 9.81)),
        _sample(2, ImuSensorType.GYROSCOPE, 20, (0.0, 0.0, 0.0)),
        _sample(3, ImuSensorType.GYROSCOPE, 40, (0.0, 0.0, 0.0)),
    )
    timeline = build_recorded_imu_pose_timeline("session", samples, None)

    assert timeline.pose_at(19) is None
    assert timeline.pose_at(39) is timeline.poses[0]
    assert timeline.pose_at(40) is timeline.poses[1]
