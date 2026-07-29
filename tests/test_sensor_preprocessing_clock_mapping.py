import json
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from pathlib import Path

import pytest

from perception.sensor_preprocessing import (
    AlignmentStatus,
    ClockId,
    ClockMappingSegment,
    ImuSensorType,
    MetadataMatchStatus,
    Mp4Timestamp,
    RawFrameRef,
    RawImuSample,
    RecordedAlignmentError,
    SegmentedClockMapper,
    StoredAlignment,
    TimeObservation,
    TimestampOutOfRangeError,
    TimestampSemantic,
    TimeStatus,
    client_perf_source_instance_id,
    derive_recorded_clock_mapping,
    frame_callback_observation,
    frame_presentation_observation,
    frame_sdk_observation,
    glasses_elapsed_source_instance_id,
    imu_sensor_event_observation,
    mp4_source_instance_id,
    persist_recorded_clock_mapping,
    rokid_sdk_source_instance_id,
    rtp_match_error_to_uncertainty_ns,
)

SESSION_ID = "session-1"
CONNECTION_ID = "connection-1"


def _segment(**overrides: object) -> ClockMappingSegment:
    values: dict[str, object] = {
        "session_id": SESSION_ID,
        "source_clock_id": ClockId.GLASSES_ELAPSED_REALTIME_NS,
        "source_instance_id": glasses_elapsed_source_instance_id(
            SESSION_ID,
            CONNECTION_ID,
        ),
        "segment_index": 0,
        "source_from": 1_000_000_000,
        "source_to": 2_000_000_000,
        "source_anchor": 1_000_000_000,
        "target_anchor_ns": 0,
        "scale_numerator_ns": 1,
        "scale_denominator_source_units": 1,
        "uncertainty_ns": 200,
        "status": TimeStatus.ESTIMATED,
        "fit_method": "android_elapsed_realtime_identity",
        "provenance_id": "device-clock-contract-v1",
        "uncertainty_basis": "device_test_residual_max",
        "evidence_id": "device-test-2026-07",
    }
    values.update(overrides)
    return ClockMappingSegment(**values)  # type: ignore[arg-type]


def _imu_observation(
    timestamp: int,
    *,
    instance: str | None = None,
    session_id: str = SESSION_ID,
) -> TimeObservation:
    return TimeObservation(
        session_id=session_id,
        source_clock_id=ClockId.GLASSES_ELAPSED_REALTIME_NS,
        source_instance_id=(
            glasses_elapsed_source_instance_id(session_id, CONNECTION_ID)
            if instance is None
            else instance
        ),
        source_timestamp=timestamp,
        timestamp_semantic=TimestampSemantic.SENSOR_EVENT,
    )


def _pending_alignment() -> StoredAlignment:
    return StoredAlignment(
        status=AlignmentStatus.PENDING,
        session_time_ns=None,
        uncertainty_ns=None,
        clock_mapping_segment_id=None,
    )


def _raw_imu_sample() -> RawImuSample:
    return RawImuSample(
        sample_id=1,
        session_id=SESSION_ID,
        connection_session_id=CONNECTION_ID,
        sensor_type=ImuSensorType.GYROSCOPE,
        android_sensor_type=4,
        sequence_number=0,
        sensor_event_monotonic_ns=1_100_000_000,
        received_at_elapsed_realtime_ns=1_100_100_000,
        received_at_client_perf_counter_ns=5_000_000_000,
        accuracy=3,
        values=(0.1, 0.2, 0.3),
        unit="rad_s",
        stored_alignment=_pending_alignment(),
    )


def _raw_frame(tmp_path: Path, *, pts: int = 0) -> RawFrameRef:
    return RawFrameRef(
        video_frame_row_id=1,
        session_id=SESSION_ID,
        clip_id="clip-1",
        frame_index=0,
        media_path=(tmp_path / "clip.mp4").resolve(),
        mp4_timestamp=Mp4Timestamp(pts, 1, 30),
        metadata_match_status=MetadataMatchStatus.EXACT,
        video_frame_metadata_id="metadata-1",
        frame_metadata_match_id=1,
        timestamp_match_error_90khz=0,
        connection_session_id=CONNECTION_ID,
        camera_start_generation=1,
        frame_id=0,
        captured_at_rokid_sdk_ms=10_000,
        received_at_elapsed_realtime_ns=1_100_000_000,
        video_at_monotonic_ns=1_100_000_000,
        rtp_timestamp_90khz=90,
        received_at_client_perf_counter_ns=5_000_000_000,
        metadata_received_at_client_perf_counter_ns=5_000_100_000,
        width=1280,
        height=720,
        rotation_degrees=0,
        capture_config_id="720p30",
        source_frame_timestamp=Mp4Timestamp(pts, 1, 90_000),
        stored_alignment=_pending_alignment(),
    )


def test_recorded_evidence_derives_device_and_mp4_segments(tmp_path: Path) -> None:
    frames = tuple(
        replace(
            _raw_frame(tmp_path, pts=index),
            video_frame_row_id=index + 1,
            frame_index=index,
            frame_id=index,
            video_at_monotonic_ns=1_100_000_000 + index * 33_333_333,
            received_at_elapsed_realtime_ns=(
                1_100_100_000 + index * 33_333_333
            ),
            received_at_client_perf_counter_ns=(
                5_000_000_000 + index * 33_333_333
            ),
            metadata_received_at_client_perf_counter_ns=(
                5_000_100_000 + index * 33_333_333
            ),
            source_frame_timestamp=Mp4Timestamp(index, 1, 30),
        )
        for index in range(3)
    )
    first_imu = replace(
        _raw_imu_sample(),
        sensor_event_monotonic_ns=1_050_000_000,
        received_at_client_perf_counter_ns=4_950_200_000,
    )
    second_imu = replace(
        _raw_imu_sample(),
        sample_id=2,
        sequence_number=1,
        sensor_event_monotonic_ns=1_200_000_000,
        received_at_elapsed_realtime_ns=1_200_100_000,
        received_at_client_perf_counter_ns=5_100_300_000,
    )

    first = derive_recorded_clock_mapping(
        SESSION_ID,
        frames,
        (first_imu, second_imu),
    )
    replay = derive_recorded_clock_mapping(
        SESSION_ID,
        frames,
        (first_imu, second_imu),
    )

    assert first == replay
    assert len(first.mapper.segments) == 2
    assert {segment.source_clock_id for segment in first.mapper.segments} == {
        ClockId.GLASSES_ELAPSED_REALTIME_NS,
        ClockId.MP4_PRESENTATION_TICKS,
    }
    callback_time = first.mapper.map(frame_callback_observation(frames[1]))
    presentation_time = first.mapper.map(frame_presentation_observation(frames[1]))
    imu_time = first.mapper.map(imu_sensor_event_observation(second_imu))
    assert callback_time.status is TimeStatus.ESTIMATED
    assert presentation_time.status is TimeStatus.ESTIMATED
    assert abs(callback_time.session_time_ns - presentation_time.session_time_ns) <= 1
    assert imu_time.session_time_ns > callback_time.session_time_ns

    session_directory = tmp_path / SESSION_ID
    session_directory.mkdir()
    artifact_path = persist_recorded_clock_mapping(first, session_directory)
    first_bytes = artifact_path.read_bytes()
    assert persist_recorded_clock_mapping(first, session_directory) == artifact_path
    assert artifact_path.read_bytes() == first_bytes
    payload = json.loads(first_bytes)
    assert payload["contract_id"] == "sensor-clock-mapping-v1"
    assert payload["evidence_sha256"] == first.evidence_sha256
    assert len(payload["segments"]) == 2
    assert all(segment["status"] == "estimated" for segment in payload["segments"])


def test_recorded_alignment_rejects_missing_strict_evidence(tmp_path: Path) -> None:
    frame = _raw_frame(tmp_path)

    with pytest.raises(RecordedAlignmentError, match="requires IMU samples"):
        derive_recorded_clock_mapping(SESSION_ID, (frame,), ())
    with pytest.raises(RecordedAlignmentError, match="fewer than two"):
        derive_recorded_clock_mapping(
            SESSION_ID,
            (frame,),
            (
                replace(
                    _raw_imu_sample(),
                    sensor_event_monotonic_ns=1_050_000_000,
                ),
            ),
        )


def test_identity_mapping_subtracts_session_origin_without_float_math() -> None:
    segment = _segment(status=TimeStatus.VERIFIED)

    result = segment.map(_imu_observation(1_123_456_789))

    assert segment.scale == Fraction(1, 1)
    assert result.status is TimeStatus.VERIFIED
    assert result.session_time_ns == 123_456_789
    assert result.uncertainty_ns == 200
    assert result.clock_mapping_id == segment.clock_mapping_id
    assert result.clock_mapping_id.startswith("clock-map-v1-")
    assert result.timestamp_semantic is TimestampSemantic.SENSOR_EVENT


def test_millisecond_source_uses_exact_integer_scale() -> None:
    segment = _segment(
        source_clock_id=ClockId.ROKID_SDK_MS,
        source_instance_id=rokid_sdk_source_instance_id(
            SESSION_ID,
            CONNECTION_ID,
            1,
        ),
        source_from=10_000,
        source_to=20_000,
        source_anchor=10_000,
        scale_numerator_ns=1_000_000,
        fit_method="device_verified_sdk_milliseconds",
    )
    observation = TimeObservation(
        session_id=SESSION_ID,
        source_clock_id=ClockId.ROKID_SDK_MS,
        source_instance_id=rokid_sdk_source_instance_id(
            SESSION_ID,
            CONNECTION_ID,
            1,
        ),
        source_timestamp=10_033,
        timestamp_semantic=TimestampSemantic.CAMERA_SDK_TIMESTAMP,
    )

    result = segment.map(observation)

    assert result.session_time_ns == 33_000_000
    assert result.uncertainty_ns == 200


def test_fractional_target_rounding_adds_one_nanosecond_uncertainty() -> None:
    segment = _segment(
        source_from=0,
        source_to=10,
        source_anchor=0,
        scale_numerator_ns=1,
        scale_denominator_source_units=2,
        uncertainty_ns=5,
    )

    result = segment.map(_imu_observation(1), additional_uncertainty_ns=7)

    assert result.session_time_ns == 1
    assert result.uncertainty_ns == 13


def test_segment_rejects_timestamp_from_wrong_instance_or_range() -> None:
    segment = _segment()

    with pytest.raises(TimestampOutOfRangeError, match="does not belong"):
        segment.map(
            _imu_observation(
                1_500_000_000,
                instance=glasses_elapsed_source_instance_id(
                    SESSION_ID,
                    "connection-2",
                ),
            )
        )
    with pytest.raises(TimestampOutOfRangeError, match="does not belong"):
        segment.map(_imu_observation(2_000_000_001))


def test_segment_rejects_non_observation_input() -> None:
    with pytest.raises(TypeError, match="TimeObservation"):
        _segment().map("not-an-observation")  # type: ignore[arg-type]


def test_mapper_returns_unavailable_without_discarding_source_evidence() -> None:
    observation = _imu_observation(999_999_999)

    result = SegmentedClockMapper(SESSION_ID, (_segment(),)).map(observation)

    assert result.status is TimeStatus.UNAVAILABLE
    assert result.source_clock_id is observation.source_clock_id
    assert result.source_instance_id == observation.source_instance_id
    assert result.source_timestamp == observation.source_timestamp
    assert result.timestamp_semantic is TimestampSemantic.SENSOR_EVENT
    assert result.clock_mapping_id is None


def test_mapper_selects_segment_by_clock_instance_and_range() -> None:
    first_boot = _segment()
    second_boot = _segment(
        source_instance_id=glasses_elapsed_source_instance_id(
            SESSION_ID,
            "connection-2",
        ),
        target_anchor_ns=5_000_000_000,
    )
    mapper = SegmentedClockMapper(SESSION_ID, (second_boot, first_boot))

    first_result = mapper.map(_imu_observation(1_000_000_100))
    second_result = mapper.map(
        _imu_observation(
            1_000_000_100,
            instance=glasses_elapsed_source_instance_id(
                SESSION_ID,
                "connection-2",
            ),
        )
    )

    assert first_result.session_time_ns == 100
    assert second_result.session_time_ns == 5_000_000_100


def test_mapper_rejects_overlapping_segments_for_same_clock_instance() -> None:
    overlapping = _segment(
        segment_index=1,
        source_from=1_500_000_000,
        source_anchor=1_500_000_000,
        target_anchor_ns=500_000_000,
    )

    with pytest.raises(ValueError, match="cannot overlap"):
        SegmentedClockMapper(SESSION_ID, (_segment(), overlapping))


def test_mapper_rejects_reversed_indices_and_target_time_regression() -> None:
    first = _segment(
        segment_index=0,
        source_from=0,
        source_to=9,
        source_anchor=0,
        target_anchor_ns=100,
    )
    reversed_index = _segment(
        segment_index=0,
        source_from=10,
        source_to=19,
        source_anchor=10,
        target_anchor_ns=110,
    )
    regressing_target = _segment(
        segment_index=1,
        source_from=10,
        source_to=19,
        source_anchor=10,
        target_anchor_ns=50,
    )

    with pytest.raises(ValueError, match="indices must be contiguous"):
        SegmentedClockMapper(SESSION_ID, (first, reversed_index))
    with pytest.raises(ValueError, match="strictly increase"):
        SegmentedClockMapper(SESSION_ID, (first, regressing_target))


def test_adjacent_segments_include_each_boundary_without_ambiguity() -> None:
    first = _segment(
        segment_index=0,
        source_from=0,
        source_to=9,
        source_anchor=0,
        target_anchor_ns=0,
    )
    second = _segment(
        segment_index=1,
        source_from=10,
        source_to=19,
        source_anchor=10,
        target_anchor_ns=10,
    )
    mapper = SegmentedClockMapper(SESSION_ID, (second, first))

    assert mapper.map(_imu_observation(9)).session_time_ns == 9
    assert mapper.map(_imu_observation(10)).session_time_ns == 10


def test_verified_mapping_is_weakened_by_estimated_or_unavailable_association() -> None:
    segment = _segment(status=TimeStatus.VERIFIED)
    observation = _imu_observation(1_100_000_000)

    estimated = segment.map(observation, additional_status=TimeStatus.ESTIMATED)
    unavailable = segment.map(observation, additional_status=TimeStatus.UNAVAILABLE)

    assert estimated.status is TimeStatus.ESTIMATED
    assert unavailable.status is TimeStatus.UNAVAILABLE
    assert unavailable.session_time_ns is None


def test_mapping_id_and_result_are_content_deterministic() -> None:
    first = _segment()
    replay = _segment()
    observation = _imu_observation(1_100_000_000)

    assert first.clock_mapping_id == replay.clock_mapping_id
    assert first.map(observation) == replay.map(observation)


def test_mapping_id_cannot_collide_when_text_fields_contain_separators() -> None:
    first = _segment(fit_method="a\x1fb", provenance_id="c")
    second = _segment(fit_method="a", provenance_id="b\x1fc")

    assert first.clock_mapping_id != second.clock_mapping_id


def test_source_instance_helpers_encode_each_reset_boundary() -> None:
    elapsed = glasses_elapsed_source_instance_id(SESSION_ID, CONNECTION_ID)

    assert elapsed != glasses_elapsed_source_instance_id(SESSION_ID, "connection-2")
    assert elapsed != glasses_elapsed_source_instance_id("session-2", CONNECTION_ID)
    assert rokid_sdk_source_instance_id(
        SESSION_ID, CONNECTION_ID, 1
    ) != rokid_sdk_source_instance_id(SESSION_ID, CONNECTION_ID, 2)
    assert client_perf_source_instance_id(
        SESSION_ID, CONNECTION_ID
    ) != client_perf_source_instance_id(SESSION_ID, "connection-2")
    assert mp4_source_instance_id(
        SESSION_ID, "clip-1", 1, 90_000
    ) != mp4_source_instance_id(SESSION_ID, "clip-2", 1, 90_000)


@pytest.mark.parametrize(
    "source_instance_id, message",
    [
        ("not-json", "canonical constructor"),
        (
            '{"kind":"glasses-elapsed","components":["session-1","connection-1"]}',
            "canonical JSON serialization",
        ),
        (
            glasses_elapsed_source_instance_id("session-2", CONNECTION_ID),
            "session does not match",
        ),
        (
            rokid_sdk_source_instance_id(SESSION_ID, CONNECTION_ID, 1),
            "does not match source clock",
        ),
    ],
)
def test_mapping_rejects_invalid_source_instance_scope(
    source_instance_id: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _segment(source_instance_id=source_instance_id)


def test_mapper_rejects_segment_from_another_session() -> None:
    other_session_segment = _segment(
        session_id="session-2",
        source_instance_id=glasses_elapsed_source_instance_id(
            "session-2",
            CONNECTION_ID,
        ),
    )

    with pytest.raises(ValueError, match="session does not match mapper"):
        SegmentedClockMapper(SESSION_ID, (other_session_segment,))


def test_verified_mapping_requires_evidence() -> None:
    with pytest.raises(ValueError, match="requires evidence_id"):
        _segment(status=TimeStatus.VERIFIED, evidence_id=None)


@pytest.mark.parametrize("additional_uncertainty_ns", [-1, 1.5, True])
def test_mapping_rejects_invalid_additional_uncertainty(
    additional_uncertainty_ns: object,
) -> None:
    expected_exception = ValueError if additional_uncertainty_ns == -1 else TypeError

    with pytest.raises(expected_exception, match="additional uncertainty"):
        _segment().map(
            _imu_observation(1_100_000_000),
            additional_uncertainty_ns=additional_uncertainty_ns,  # type: ignore[arg-type]
        )


def test_mp4_mapping_accepts_negative_pts_and_keeps_long_timing_exact() -> None:
    source_instance_id = mp4_source_instance_id(SESSION_ID, "clip-1", 1, 90_000)
    segment = _segment(
        source_clock_id=ClockId.MP4_PRESENTATION_TICKS,
        source_instance_id=source_instance_id,
        source_from=-1,
        source_to=10_000,
        source_anchor=-1,
        target_anchor_ns=0,
        scale_numerator_ns=1_000_000_000,
        scale_denominator_source_units=90_000,
        uncertainty_ns=0,
        fit_method="mp4_time_base",
    )
    negative_pts = TimeObservation(
        session_id=SESSION_ID,
        source_clock_id=ClockId.MP4_PRESENTATION_TICKS,
        source_instance_id=source_instance_id,
        source_timestamp=-1,
        timestamp_semantic=TimestampSemantic.MEDIA_PRESENTATION,
    )
    long_pts = TimeObservation(
        session_id=SESSION_ID,
        source_clock_id=ClockId.MP4_PRESENTATION_TICKS,
        source_instance_id=source_instance_id,
        source_timestamp=3_002,
        timestamp_semantic=TimestampSemantic.MEDIA_PRESENTATION,
    )

    assert segment.map(negative_pts).session_time_ns == 0
    mapped = segment.map(long_pts)
    assert mapped.session_time_ns == 33_366_667
    assert mapped.uncertainty_ns == 1


def test_raw_records_create_session_scoped_observations(tmp_path: Path) -> None:
    sample_observation = imu_sensor_event_observation(_raw_imu_sample())
    frame = _raw_frame(tmp_path, pts=-1)
    callback_observation = frame_callback_observation(frame)
    sdk_observation = frame_sdk_observation(frame)
    presentation_observation = frame_presentation_observation(frame)

    assert sample_observation.source_instance_id == glasses_elapsed_source_instance_id(
        SESSION_ID,
        CONNECTION_ID,
    )
    assert sample_observation.timestamp_semantic is TimestampSemantic.SENSOR_EVENT
    assert callback_observation is not None
    assert callback_observation.timestamp_semantic is TimestampSemantic.CAMERA_CALLBACK
    assert sdk_observation is not None
    assert sdk_observation.source_instance_id == rokid_sdk_source_instance_id(
        SESSION_ID,
        CONNECTION_ID,
        1,
    )
    assert sdk_observation.timestamp_semantic is TimestampSemantic.CAMERA_SDK_TIMESTAMP
    assert presentation_observation.source_timestamp == -1
    assert presentation_observation.source_instance_id == mp4_source_instance_id(
        SESSION_ID,
        "clip-1",
        1,
        30,
    )


@pytest.mark.parametrize(
    "clock_id",
    [ClockId.GLASSES_ELAPSED_REALTIME_NS, ClockId.ROKID_SDK_MS],
)
def test_raw_camera_clocks_cannot_claim_exposure_semantics(clock_id: ClockId) -> None:
    instance = (
        glasses_elapsed_source_instance_id(SESSION_ID, CONNECTION_ID)
        if clock_id is ClockId.GLASSES_ELAPSED_REALTIME_NS
        else rokid_sdk_source_instance_id(SESSION_ID, CONNECTION_ID, 1)
    )

    with pytest.raises(ValueError, match="incompatible"):
        TimeObservation(
            session_id=SESSION_ID,
            source_clock_id=clock_id,
            source_instance_id=instance,
            source_timestamp=1_000,
            timestamp_semantic=TimestampSemantic.CAMERA_EXPOSURE,
        )


def test_empty_or_wrong_session_mapper_returns_unavailable() -> None:
    empty_mapper = SegmentedClockMapper(SESSION_ID, ())
    wrong_session_observation = _imu_observation(
        1_100_000_000,
        session_id="session-2",
    )

    assert empty_mapper.map(_imu_observation(1_100_000_000)).status is TimeStatus.UNAVAILABLE
    assert (
        SegmentedClockMapper(SESSION_ID, (_segment(),))
        .map(wrong_session_observation)
        .status
        is TimeStatus.UNAVAILABLE
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"source_to": 999_999_999}, "source range"),
        ({"source_anchor": 999_999_999}, "source_anchor"),
        ({"scale_numerator_ns": 0}, "scale"),
        ({"uncertainty_ns": -1}, "uncertainty"),
        ({"status": TimeStatus.UNAVAILABLE}, "verified or estimated"),
        ({"target_clock_id": ClockId.CLIENT_PERF_COUNTER_NS}, "target"),
    ],
)
def test_mapping_segment_rejects_invalid_contract_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _segment(**overrides)


def test_time_observation_rejects_clock_semantic_mismatch() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        TimeObservation(
            session_id=SESSION_ID,
            source_clock_id=ClockId.CLIENT_PERF_COUNTER_NS,
            source_instance_id="client-process-1",
            source_timestamp=1_000,
            timestamp_semantic=TimestampSemantic.CAMERA_EXPOSURE,
        )


@pytest.mark.parametrize(
    "ticks, expected_ns",
    [(0, 0), (1, 11_112), (90, 1_000_000), (1_000, 11_111_112)],
)
def test_rtp_match_error_conversion_is_a_conservative_integer_bound(
    ticks: int,
    expected_ns: int,
) -> None:
    assert rtp_match_error_to_uncertainty_ns(ticks) == expected_ns


def test_rtp_match_error_rejects_negative_ticks() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        rtp_match_error_to_uncertainty_ns(-1)


def test_clock_mapping_records_are_immutable() -> None:
    segment = _segment()

    with pytest.raises(FrozenInstanceError):
        segment.uncertainty_ns = 0  # type: ignore[misc]
