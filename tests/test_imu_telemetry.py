from __future__ import annotations

import asyncio

import pytest

from ui.gateway.imu_telemetry import ImuTelemetryRuntime
from ui.gateway.webrtc_models import ImuSample, ImuSensorType


def _sample(sequence: int, event_ns: int) -> ImuSample:
    return ImuSample(
        sensor_type=ImuSensorType.ACCELEROMETER,
        android_sensor_type=1,
        sequence_number=sequence,
        sensor_event_monotonic_ns=event_ns,
        received_at_elapsed_realtime_ns=event_ns + 2_000_000,
        accuracy=3,
        values=(1.0, 2.0, 3.0),
    )


def test_raw_imu_monitor_reports_rate_latency_and_sequence_quality() -> None:
    async def scenario() -> None:
        runtime = ImuTelemetryRuntime(maximum_points_per_sensor=3)
        samples = (
            (0, 1_000_000_000),
            (2, 1_010_000_000),
            (2, 1_020_000_000),
            (1, 1_030_000_000),
        )
        for sequence, event_ns in samples:
            await runtime.submit_imu_sample(
                connection_session_id="connection",
                sample=_sample(sequence, event_ns),
                received_at_client_monotonic_ns=event_ns + 5_000_000,
            )

        snapshot = runtime.snapshot(now_ns=1_040_000_000)
        assert snapshot.accelerometer_rate_hz == pytest.approx(100.0)
        assert snapshot.latest_callback_latency_ms == 2.0
        assert snapshot.sequence_gap_count == 1
        assert snapshot.duplicate_count == 1
        assert snapshot.out_of_order_count == 1
        assert snapshot.buffer_eviction_count == 1
        assert len(snapshot.accelerometer) == 3
        assert [item.sequence for item in runtime.samples_after(2)] == [3, 4]

    asyncio.run(scenario())


def test_raw_imu_monitor_resets_sequence_quality_for_a_new_connection() -> None:
    async def scenario() -> None:
        runtime = ImuTelemetryRuntime(maximum_points_per_sensor=3)
        await runtime.submit_imu_sample(
            connection_session_id="first",
            sample=_sample(50, 1_000_000_000),
            received_at_client_monotonic_ns=1_001_000_000,
        )
        await runtime.submit_imu_sample(
            connection_session_id="second",
            sample=_sample(0, 2_000_000_000),
            received_at_client_monotonic_ns=2_001_000_000,
        )

        snapshot = runtime.snapshot(now_ns=2_002_000_000)
        assert snapshot.connection_session_id == "second"
        assert snapshot.out_of_order_count == 0
        assert snapshot.sequence_gap_count == 0
        assert [point.sequence for point in snapshot.accelerometer] == [2]

    asyncio.run(scenario())
