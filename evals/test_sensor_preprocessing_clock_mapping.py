from sensor_preprocessing import (
    ClockId,
    ClockMappingSegment,
    SegmentedClockMapper,
    TimeObservation,
    TimestampSemantic,
    TimeStatus,
    rokid_sdk_source_instance_id,
    rtp_match_error_to_uncertainty_ns,
)


def test_camera_clock_mapping_preserves_semantics_and_measured_uncertainty() -> None:
    segment = ClockMappingSegment(
        session_id="session-1",
        source_clock_id=ClockId.ROKID_SDK_MS,
        source_instance_id=rokid_sdk_source_instance_id(
            "session-1",
            "connection-1",
            1,
        ),
        segment_index=0,
        source_from=10_000,
        source_to=20_000,
        source_anchor=10_000,
        target_anchor_ns=2_000_000_000,
        scale_numerator_ns=1_000_000,
        scale_denominator_source_units=1,
        uncertainty_ns=500_000,
        status=TimeStatus.ESTIMATED,
        fit_method="external_camera_clock_fit_v1",
        provenance_id="calibration-profile-1",
        uncertainty_basis="fit_residual_max_plus_sdk_quantization",
    )
    observation = TimeObservation(
        session_id="session-1",
        source_clock_id=ClockId.ROKID_SDK_MS,
        source_instance_id=segment.source_instance_id,
        source_timestamp=10_033,
        timestamp_semantic=TimestampSemantic.CAMERA_SDK_TIMESTAMP,
    )
    match_uncertainty_ns = rtp_match_error_to_uncertainty_ns(622)
    mapper = SegmentedClockMapper("session-1", (segment,))

    first = mapper.map(
        observation,
        additional_uncertainty_ns=match_uncertainty_ns,
    )
    replay = mapper.map(
        observation,
        additional_uncertainty_ns=match_uncertainty_ns,
    )

    assert first == replay
    assert first.session_time_ns == 2_033_000_000
    assert first.uncertainty_ns == 500_000 + match_uncertainty_ns
    assert first.status is TimeStatus.ESTIMATED
    assert first.timestamp_semantic is TimestampSemantic.CAMERA_SDK_TIMESTAMP
    assert first.timestamp_semantic is not TimestampSemantic.CAMERA_EXPOSURE


def test_clock_reset_cannot_reuse_another_source_instances_mapping() -> None:
    segment = ClockMappingSegment(
        session_id="session-1",
        source_clock_id=ClockId.ROKID_SDK_MS,
        source_instance_id=rokid_sdk_source_instance_id(
            "session-1",
            "connection-1",
            1,
        ),
        segment_index=0,
        source_from=0,
        source_to=1_000,
        source_anchor=0,
        target_anchor_ns=0,
        scale_numerator_ns=1_000_000,
        scale_denominator_source_units=1,
        uncertainty_ns=1_000_000,
        status=TimeStatus.ESTIMATED,
        fit_method="external_camera_clock_fit_v1",
        provenance_id="calibration-profile-1",
        uncertainty_basis="fit_residual_max_plus_sdk_quantization",
    )
    restarted_camera_timestamp = TimeObservation(
        session_id="session-1",
        source_clock_id=ClockId.ROKID_SDK_MS,
        source_instance_id=rokid_sdk_source_instance_id(
            "session-1",
            "connection-1",
            2,
        ),
        source_timestamp=500,
        timestamp_semantic=TimestampSemantic.CAMERA_SDK_TIMESTAMP,
    )

    result = SegmentedClockMapper("session-1", (segment,)).map(
        restarted_camera_timestamp
    )

    assert result.status is TimeStatus.UNAVAILABLE
    assert result.session_time_ns is None
    assert result.clock_mapping_id is None
