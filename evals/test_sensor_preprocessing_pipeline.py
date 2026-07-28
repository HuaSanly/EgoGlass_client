from pathlib import Path

import numpy as np
from av import VideoFrame

from perception.sensor_preprocessing import (
    ClockId,
    ClockMappingSegment,
    ImuSensorType,
    LiveFrameInput,
    LiveImuInput,
    SegmentedClockMapper,
    SensorPreprocessingPipeline,
    TimeObservation,
    TimestampSemantic,
    TimeStatus,
    glasses_elapsed_source_instance_id,
)


def test_placeholder_profile_exercises_the_real_720p30_live_boundary() -> None:
    session_id = "pipeline-eval-session"
    connection_id = "pipeline-eval-connection"
    instance_id = glasses_elapsed_source_instance_id(session_id, connection_id)
    segment = ClockMappingSegment(
        session_id=session_id,
        source_clock_id=ClockId.GLASSES_ELAPSED_REALTIME_NS,
        source_instance_id=instance_id,
        segment_index=0,
        source_from=0,
        source_to=1_000_000_000,
        source_anchor=0,
        target_anchor_ns=0,
        scale_numerator_ns=1,
        scale_denominator_source_units=1,
        uncertainty_ns=1_000_000,
        status=TimeStatus.ESTIMATED,
        fit_method="eval_identity",
        provenance_id="pipeline-eval",
        uncertainty_basis="fixed_eval_bound",
    )
    calibration_path = (
        Path(__file__).parents[1] / "config" / "sensor-calibration.sample.json"
    )
    pipeline = SensorPreprocessingPipeline.from_calibration_file(
        calibration_path,
        SegmentedClockMapper(session_id, (segment,)),
        allow_placeholder_calibration=True,
    )
    frame_observation = TimeObservation(
        session_id=session_id,
        source_clock_id=ClockId.GLASSES_ELAPSED_REALTIME_NS,
        source_instance_id=instance_id,
        source_timestamp=20_000_000,
        timestamp_semantic=TimestampSemantic.CAMERA_CALLBACK,
    )
    imu_observation = TimeObservation(
        session_id=session_id,
        source_clock_id=ClockId.GLASSES_ELAPSED_REALTIME_NS,
        source_instance_id=instance_id,
        source_timestamp=10_000_000,
        timestamp_semantic=TimestampSemantic.SENSOR_EVENT,
    )
    decoded_frame = VideoFrame.from_ndarray(
        np.zeros((720, 1280, 3), dtype=np.uint8),
        format="bgr24",
    )

    bundle = pipeline.process_live_frame(
        decoded_frame,
        LiveFrameInput(
            session_id=session_id,
            stream_id="live-eval",
            frame_index=0,
            time_observation=frame_observation,
            rotation_degrees=90,
            capture_config_id="720p30",
        ),
        (
            LiveImuInput(
                sample_id=0,
                sequence_number=0,
                time_observation=imu_observation,
                sensor_type=ImuSensorType.ACCELEROMETER,
                values=(0.0, 0.0, 9.81),
                unit="m_s2",
                accuracy=3,
            ),
        ),
    )

    assert bundle.image_bgr.shape == (1280, 720, 3)
    assert bundle.session_time_ns == 20_000_000
    assert bundle.imu_samples[0].session_time_ns == 10_000_000
    assert bundle.calibration.placeholder is True
    assert bundle.timestamp_semantic is TimestampSemantic.CAMERA_CALLBACK
