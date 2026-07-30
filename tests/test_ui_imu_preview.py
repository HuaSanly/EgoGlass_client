from __future__ import annotations

import asyncio
import time

import numpy as np

from ingest_gateway.imu_preview import ImuPreviewRuntime
from ingest_gateway.webrtc_models import ImuSample, ImuSensorType


def _sample(sensor: ImuSensorType, sequence: int, timestamp_ns: int) -> ImuSample:
    return ImuSample(
        sensor_type=sensor,
        android_sensor_type=1 if sensor is ImuSensorType.ACCELEROMETER else 4,
        sequence_number=sequence,
        sensor_event_monotonic_ns=timestamp_ns,
        received_at_elapsed_realtime_ns=timestamp_ns + 1_000_000,
        accuracy=3,
        values=(0.0, 0.0, 9.81) if sensor is ImuSensorType.ACCELEROMETER else (0.1, 0.0, 0.0),
    )


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
