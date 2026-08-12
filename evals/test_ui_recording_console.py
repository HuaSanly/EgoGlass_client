from __future__ import annotations

import asyncio
import time

from PyQt6.QtWidgets import QApplication

from ui.gateway.imu_telemetry import ImuTelemetryRuntime
from ui.gateway.webrtc_models import ImuSample, ImuSensorType
from ui.widgets.imu_monitor import ImuChartSample, ImuMonitorWidget


def test_raw_imu_chart_stays_responsive_at_capture_window_capacity(
    qt_application: QApplication,
) -> None:
    monitor = ImuMonitorWidget(maximum_samples=600)
    monitor.resize(420, 620)
    monitor.show()
    samples = tuple(
        ImuChartSample(
            sensor_type="accelerometer" if index % 2 == 0 else "gyroscope",
            sequence_number=index,
            recording_time_ns=index * 5_000_000,
            received_at_client_monotonic_ns=index * 5_000_000 + 1_000_000,
            values=(index / 100, index / 200, index / 400),
        )
        for index in range(1200)
    )

    started = time.perf_counter()
    assert monitor.append_samples(samples) == 1200
    qt_application.processEvents()
    render_seconds = time.perf_counter() - started

    assert monitor.sample_count("accelerometer") == 600
    assert monitor.sample_count("gyroscope") == 600
    assert render_seconds < 0.5
    assert not monitor.grab().isNull()
    monitor.close()


def test_display_buffer_eviction_is_not_reported_as_capture_queue_overflow() -> None:
    async def scenario() -> None:
        runtime = ImuTelemetryRuntime(maximum_points_per_sensor=2)
        for sequence in range(3):
            await runtime.submit_imu_sample(
                connection_session_id="eval-connection",
                sample=ImuSample(
                    sensor_type=ImuSensorType.ACCELEROMETER,
                    android_sensor_type=1,
                    sequence_number=sequence,
                    sensor_event_monotonic_ns=sequence * 10_000_000,
                    received_at_elapsed_realtime_ns=sequence * 10_000_000 + 1_000_000,
                    accuracy=3,
                    values=(0.0, 0.0, 9.81),
                ),
                received_at_client_monotonic_ns=sequence * 10_000_000 + 2_000_000,
            )

        snapshot = runtime.snapshot(now_ns=30_000_000)
        assert snapshot.buffer_eviction_count == 1
        assert not hasattr(snapshot, "queue_overflow_count")

    asyncio.run(scenario())
