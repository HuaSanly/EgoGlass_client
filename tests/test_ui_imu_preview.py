from __future__ import annotations

import asyncio
import time

import numpy as np

from ingest_gateway.imu_preview import ImuPreviewRuntime
from ingest_gateway.webrtc_models import ImuSample, ImuSensorType


def _sample(
    sensor: ImuSensorType,
    sequence: int,
    timestamp_ns: int,
    *,
    values: tuple[float, float, float] | None = None,
) -> ImuSample:
    return ImuSample(
        sensor_type=sensor,
        android_sensor_type=1 if sensor is ImuSensorType.ACCELEROMETER else 4,
        sequence_number=sequence,
        sensor_event_monotonic_ns=timestamp_ns,
        received_at_elapsed_realtime_ns=timestamp_ns + 1_000_000,
        accuracy=3,
        values=values
        or (
            (0.0, 0.0, 9.81)
            if sensor is ImuSensorType.ACCELEROMETER
            else (0.1, 0.0, 0.0)
        ),
    )


async def _wait_until_processed(runtime: ImuPreviewRuntime, count: int) -> None:
    for _ in range(2_000):
        if runtime.snapshot().samples_processed == count:
            return
        await asyncio.sleep(0.001)
    raise AssertionError(f"IMU preview did not process {count} samples")


def test_imu_preview_consumes_ordered_samples_and_produces_finite_pose() -> None:
    async def run() -> None:
        runtime = ImuPreviewRuntime()
        try:
            received_at_ns = time.perf_counter_ns()
            await runtime.submit_imu_sample(
                session_id="session",
                sample=_sample(ImuSensorType.ACCELEROMETER, 0, 1_000_000_000),
                received_at_client_monotonic_ns=received_at_ns,
            )
            for index in range(5):
                await runtime.submit_imu_sample(
                    session_id="session",
                    sample=_sample(
                        ImuSensorType.GYROSCOPE,
                        index,
                        1_000_000_000 + index * 10_000_000,
                    ),
                    received_at_client_monotonic_ns=received_at_ns + index,
                )
            for _ in range(100):
                snapshot = runtime.snapshot()
                if snapshot.samples_processed == 6:
                    break
                await asyncio.sleep(0.001)
            assert snapshot.samples_received == 6
            assert snapshot.samples_processed == 6
            assert snapshot.queue_overflow_count == 0
            assert np.isfinite(snapshot.quaternion_wxyz).all()
            assert snapshot.session_id == "session"
        finally:
            await runtime.close()

    asyncio.run(run())


def test_imu_preview_reset_orientation_returns_to_identity_pose() -> None:
    async def run() -> None:
        runtime = ImuPreviewRuntime()
        try:
            with runtime._lock:
                runtime._quaternion = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
                runtime._orientation_initialized = True
                runtime._last_gyro_timestamp_ns = 123

            runtime.reset_orientation()
            snapshot = runtime.snapshot()

            assert snapshot.quaternion_wxyz == (1.0, 0.0, 0.0, 0.0)
            assert snapshot.roll_degrees == 0.0
            assert snapshot.pitch_degrees == 0.0
            assert snapshot.yaw_degrees == 0.0
            with runtime._lock:
                assert np.allclose(runtime._quaternion, (0.5, 0.5, 0.5, 0.5))
                assert np.allclose(
                    runtime._reference_quaternion,
                    (0.5, 0.5, 0.5, 0.5),
                )
                assert runtime._last_gyro_timestamp_ns == 123
        finally:
            await runtime.close()

    asyncio.run(run())


def test_imu_preview_reports_gyroscope_rate_instead_of_combined_sensor_rate() -> None:
    async def run() -> None:
        runtime = ImuPreviewRuntime()
        try:
            base_sensor_ns = 1_000_000_000
            received_at_ns = time.perf_counter_ns()
            for index in range(50):
                timestamp_ns = base_sensor_ns + index * 10_000_000
                for sensor in (
                    ImuSensorType.ACCELEROMETER,
                    ImuSensorType.GYROSCOPE,
                ):
                    await runtime.submit_imu_sample(
                        session_id="session",
                        sample=_sample(sensor, index, timestamp_ns),
                        received_at_client_monotonic_ns=received_at_ns + index,
                    )
            await _wait_until_processed(runtime, 100)

            assert runtime.snapshot().recent_rate_hz == 100.0
        finally:
            await runtime.close()

    asyncio.run(run())


def test_imu_preview_bootstraps_tilt_without_rendering_filter_convergence() -> None:
    async def run() -> None:
        runtime = ImuPreviewRuntime(queue_size=1_024)
        try:
            runtime.reset_orientation()
            base_sensor_ns = 1_000_000_000
            received_at_ns = time.perf_counter_ns()
            tilted_gravity = (-4.903325, 0.0, 8.492809)
            for index in range(200):
                timestamp_ns = base_sensor_ns + index * 10_000_000
                await runtime.submit_imu_sample(
                    session_id="session",
                    sample=_sample(
                        ImuSensorType.ACCELEROMETER,
                        index,
                        timestamp_ns,
                        values=tilted_gravity,
                    ),
                    received_at_client_monotonic_ns=received_at_ns + index,
                )
                await runtime.submit_imu_sample(
                    session_id="session",
                    sample=_sample(
                        ImuSensorType.GYROSCOPE,
                        index,
                        timestamp_ns,
                        values=(0.0, 0.008, 0.0),
                    ),
                    received_at_client_monotonic_ns=received_at_ns + index,
                )
            await _wait_until_processed(runtime, 400)
            snapshot = runtime.snapshot()

            assert abs(snapshot.pitch_degrees) < 0.5
            assert snapshot.recent_rate_hz == 100.0
        finally:
            await runtime.close()

    asyncio.run(run())


def test_imu_preview_does_not_move_clock_back_on_out_of_order_gyroscope() -> None:
    runtime = ImuPreviewRuntime()
    try:
        with runtime._lock:
            runtime._last_gyro_timestamp_ns = 1_000_000_000
            runtime._accelerometer = (0.0, 0.0, 9.81)
            runtime._update_orientation(
                _sample(ImuSensorType.GYROSCOPE, 1, 900_000_000)
            )

            assert runtime._last_gyro_timestamp_ns == 1_000_000_000
    finally:
        asyncio.run(runtime.close())
