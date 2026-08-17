from __future__ import annotations

import time
from pathlib import Path

from ui.gateway.webrtc_models import ImuSample, ImuSensorType
from ui.imu_calibration.writer import ImuCaptureWriter

CAPTURE_ID = "c" * 32
SAMPLES_PER_SENSOR = 1_080_000


def test_three_hour_imu_capture_writes_all_2_16_million_rows(tmp_path: Path) -> None:
    writer = ImuCaptureWriter(
        tmp_path,
        capture_id=CAPTURE_ID,
        queue_size=8192,
        batch_size=1000,
    )
    writer.start()
    maximum_queue_size = 0
    for sequence in range(SAMPLES_PER_SENSOR):
        timestamp_ns = 1_000_000_000 + sequence * 10_000_000
        for sensor_type, android_type in (
            (ImuSensorType.ACCELEROMETER, 1),
            (ImuSensorType.GYROSCOPE, 4),
        ):
            while writer.stats.queue_size >= 8000:
                time.sleep(0.001)
            writer.append(
                ImuSample.model_construct(
                    schema_version="0.1",
                    message_type="imu_sample",
                    sensor_type=sensor_type,
                    android_sensor_type=android_type,
                    sequence_number=sequence,
                    sensor_event_monotonic_ns=timestamp_ns,
                    received_at_elapsed_realtime_ns=timestamp_ns + 1_000,
                    accuracy=3,
                    values=(0.01, -0.02, 0.03),
                )
            )
            maximum_queue_size = max(maximum_queue_size, writer.stats.queue_size)

    output = writer.finish(publish=True)

    assert output is not None
    with output.open("r", encoding="utf-8") as stream:
        assert sum(1 for _line in stream) == SAMPLES_PER_SENSOR * 2 + 1
    assert writer.stats.accelerometer_rows == SAMPLES_PER_SENSOR
    assert writer.stats.gyroscope_rows == SAMPLES_PER_SENSOR
    assert writer.stats.sequence_gaps == 0
    assert writer.stats.accelerometer_rate_hz == 100.0
    assert writer.stats.gyroscope_rate_hz == 100.0
    assert maximum_queue_size <= 8192
    assert {path.name for path in output.parent.iterdir()} == {"imu.csv"}
