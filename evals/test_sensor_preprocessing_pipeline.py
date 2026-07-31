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


def test_repository_profile_exercises_the_real_four_by_three_live_boundary() -> None:
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
    config_path = (
        Path(__file__).parents[1] / "config" / "sensor-preprocessing.yaml"
    )
    pipeline = SensorPreprocessingPipeline.from_config_file(
        config_path,
        SegmentedClockMapper(session_id, (segment,)),
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
    source_image = np.zeros((480, 640, 3), dtype=np.uint8)
    source_image[0, 0] = (11, 12, 13)
    source_image[0, -1] = (21, 22, 23)
    source_image[-1, 0] = (31, 32, 33)
    source_image[-1, -1] = (41, 42, 43)
    decoded_frame = VideoFrame.from_ndarray(source_image, format="bgr24")

    bundle = pipeline.process_live_frame(
        decoded_frame,
        LiveFrameInput(
            session_id=session_id,
            stream_id="live-eval",
            frame_index=0,
            time_observation=frame_observation,
            rotation_degrees=0,
            capture_config_id="640x480p30",
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

    assert bundle.image_bgr.shape == (480, 640, 3)
    np.testing.assert_array_equal(bundle.image_bgr, source_image)
    assert bundle.session_time_ns == 20_000_000
    assert bundle.imu_samples[0].session_time_ns == 10_000_000
    assert bundle.calibration.profile_name == "rokid-glass3-640x480p30-sample"
    assert bundle.timestamp_semantic is TimestampSemantic.CAMERA_CALLBACK
