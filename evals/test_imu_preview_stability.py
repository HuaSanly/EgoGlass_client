from __future__ import annotations

import asyncio
import time

from ingest_gateway.imu_preview import ImuPreviewRuntime
from ingest_gateway.webrtc_models import ImuSample, ImuSensorType


def _sample(
    sensor: ImuSensorType,
    sequence: int,
    timestamp_ns: int,
) -> ImuSample:
    gravity_at_thirty_degree_pitch = (-4.903325, 0.0, 8.492809)
    return ImuSample(
        sensor_type=sensor,
        android_sensor_type=1 if sensor is ImuSensorType.ACCELEROMETER else 4,
        sequence_number=sequence,
        sensor_event_monotonic_ns=timestamp_ns,
        received_at_elapsed_realtime_ns=timestamp_ns + 1_000_000,
        accuracy=3,
        values=(
            gravity_at_thirty_degree_pitch
            if sensor is ImuSensorType.ACCELEROMETER
            else (0.0, 0.008, 0.0)
        ),
    )


async def _wait_until_processed(runtime: ImuPreviewRuntime, count: int) -> None:
    for _ in range(5_000):
        if runtime.snapshot().samples_processed == count:
            return
        await asyncio.sleep(0.001)
    raise AssertionError(f"IMU preview did not process {count} samples")


def test_thirty_second_stationary_tilt_does_not_render_pitch_convergence() -> None:
    async def run() -> None:
        runtime = ImuPreviewRuntime(queue_size=512)
        pitch_samples: list[float] = []
        try:
            base_sensor_ns = 1_000_000_000
            received_at_ns = time.perf_counter_ns()
            sample_count = 3_000
            chunk_size = 50
            for chunk_start in range(0, sample_count, chunk_size):
                chunk_end = min(sample_count, chunk_start + chunk_size)
                for index in range(chunk_start, chunk_end):
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
                await _wait_until_processed(runtime, chunk_end * 2)
                if chunk_end == sample_count // 2:
                    runtime.reset_orientation()
                pitch_samples.append(runtime.snapshot().pitch_degrees)

            snapshot = runtime.snapshot()
            assert snapshot.recent_rate_hz == 100.0
            assert max(abs(value) for value in pitch_samples) < 1.0
            assert abs(snapshot.pitch_degrees) < 1.0
        finally:
            await runtime.close()

    asyncio.run(run())
