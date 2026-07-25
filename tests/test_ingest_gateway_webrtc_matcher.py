import pytest

from ingest_gateway.webrtc_matcher import (
    DuplicateMetadataError,
    FrameMetadataMatcher,
)
from ingest_gateway.webrtc_models import VideoFrameMetadata


def metadata(
    frame_id: int,
    rtp_timestamp: int,
    sdk_timestamp_ms: int = 1000,
    *,
    camera_start_generation: int = 1,
) -> VideoFrameMetadata:
    return VideoFrameMetadata(
        frame_id=frame_id,
        camera_start_generation=camera_start_generation,
        captured_at_rokid_sdk_ms=sdk_timestamp_ms,
        received_at_elapsed_realtime_ns=2_000_000_000,
        video_at_monotonic_ns=2_000_000_000,
        rtp_timestamp_90khz=rtp_timestamp,
        width=1280,
        height=720,
        rotation_degrees=0,
        capture_config_id="720p30",
    )


def test_matches_metadata_before_or_after_decoded_frame() -> None:
    matcher = FrameMetadataMatcher()

    assert not matcher.add_metadata(metadata(1, 90_000))
    first_match = matcher.add_frame(
        0,
        frame_index=0,
        time_base_num=1,
        time_base_den=90_000,
        received_at_client_monotonic_ns=123,
    )
    assert first_match is not None
    assert first_match.metadata.frame_id == 1
    assert first_match.decoded_frame_index == 0
    assert first_match.decoded_frame_time_base_den == 90_000
    assert first_match.decoded_frame_received_at_client_monotonic_ns == 123
    assert not matcher.add_frame(90_000)
    second_match = matcher.add_metadata(metadata(2, 180_000, 1050))
    assert second_match is not None
    assert second_match.decoded_frame_pts == 90_000

    assert matcher.matched == 2
    assert matcher.metadata_origin_90khz == 90_000
    assert matcher.pending_frames == 0
    assert matcher.pending_metadata == 0


def test_camera_restart_generation_clears_pending_and_allows_frame_id_reuse() -> None:
    matcher = FrameMetadataMatcher()
    matcher.add_metadata(metadata(1, 90_000))
    restarted = metadata(1, 180_000, 10).model_copy(
        update={"camera_start_generation": 2}
    )

    assert matcher.add_metadata(restarted) is None
    match = matcher.add_frame(0)

    assert match is not None
    assert match.metadata.camera_start_generation == 2
    assert matcher.dropped == 1


def test_normalized_rtp_timestamp_matching_survives_32_bit_wrap() -> None:
    matcher = FrameMetadataMatcher()

    assert not matcher.add_metadata(metadata(1, 0xFFFF_FF00))
    assert matcher.add_frame(0)
    assert not matcher.add_frame(0x200)
    assert matcher.add_metadata(metadata(2, 0x100, 1050))

    assert matcher.matched == 2


def test_matches_sub_millisecond_clock_quantization_but_not_adjacent_frames() -> None:
    matcher = FrameMetadataMatcher()

    matcher.add_metadata(metadata(1, 1_000_000))
    assert matcher.add_frame(45)
    matcher.add_metadata(metadata(2, 1_004_500, 1050))
    assert matcher.add_frame(3_500)
    matcher.add_metadata(metadata(3, 1_009_000, 1100))
    assert not matcher.add_frame(7_999)

    assert matcher.matched == 2
    assert matcher.max_timestamp_match_error_90khz == 1_000


def test_calibrates_dominant_receipt_time_offset_after_encoder_drops_startup() -> None:
    matcher = FrameMetadataMatcher()
    wrong_startup_origin = 322_555_136
    correct_offset = 322_619_600
    frame_period_ns = 33_333_333
    base_receipt_ns = 2_000_000_000
    jitters = (-622, -300, -90, 0, 110, 300, 622)
    matches = []

    for index in range(21):
        result = matcher.add_metadata(
            metadata(index, wrong_startup_origin + index * 3_000),
            received_at_client_monotonic_ns=1_000_000_000 + index * frame_period_ns,
        )
        if result is not None:
            matches.append(result)

    metadata_receipts = []
    for index in range(91):
        receipt_ns = base_receipt_ns + index * frame_period_ns
        metadata_receipts.append(receipt_ns)
        result = matcher.add_metadata(
            metadata(
                100 + index,
                (correct_offset + index * 3_000 + jitters[index % len(jitters)])
                & 0xFFFFFFFF,
                2_000 + index * 33,
            ),
            received_at_client_monotonic_ns=receipt_ns,
        )
        if result is not None:
            matches.append(result)

    for index in range(91):
        # Reproduce the device run's minority adjacent-frame arrival cluster.
        receipt_index = index + 1 if index < 26 else index
        result = matcher.add_frame(
            index * 3_000,
            frame_index=index,
            time_base_num=1,
            time_base_den=90_000,
            received_at_client_monotonic_ns=metadata_receipts[receipt_index] + 1_000_000,
        )
        if result is not None:
            matches.append(result)
        matches.extend(matcher.drain_matches())

    assert matcher.calibrated
    assert matcher.metadata_origin_90khz == correct_offset
    assert matcher.metadata_origin_90khz != wrong_startup_origin
    assert matcher.calibration_support >= 39
    assert matcher.matched == 91
    assert {match.metadata.frame_id for match in matches} == set(range(100, 191))
    assert max(match.timestamp_match_error_90khz for match in matches) == 622


def test_camera_generation_resets_and_recalibrates_timed_offset() -> None:
    matcher = FrameMetadataMatcher()

    def calibrate(generation: int, offset: int, receipt_origin_ns: int) -> None:
        for index in range(60):
            matcher.add_metadata(
                metadata(
                    index,
                    offset + index * 3_000,
                    camera_start_generation=generation,
                ),
                received_at_client_monotonic_ns=(
                    receipt_origin_ns + index * 33_333_333
                ),
            )
        for index in range(60):
            matcher.add_frame(
                index * 3_000,
                received_at_client_monotonic_ns=(
                    receipt_origin_ns + index * 33_333_333 + 1_000_000
                ),
            )
            matcher.drain_matches()

    calibrate(1, 400_000, 1_000_000_000)
    assert matcher.metadata_origin_90khz == 400_000

    matcher.add_metadata(
        metadata(0, 900_000, camera_start_generation=2),
        received_at_client_monotonic_ns=4_000_000_000,
    )
    assert not matcher.calibrated
    assert matcher.calibration_support == 0

    for index in range(1, 60):
        matcher.add_metadata(
            metadata(
                index,
                900_000 + index * 3_000,
                camera_start_generation=2,
            ),
            received_at_client_monotonic_ns=4_000_000_000 + index * 33_333_333,
        )
    for index in range(60):
        matcher.add_frame(
            index * 3_000,
            received_at_client_monotonic_ns=(
                4_000_000_000 + index * 33_333_333 + 1_000_000
            ),
        )
        matcher.drain_matches()

    assert matcher.metadata_origin_90khz == 900_000
    assert matcher.matched == 120


def test_late_prior_generation_metadata_does_not_reset_current_generation() -> None:
    matcher = FrameMetadataMatcher()
    matcher.add_metadata(metadata(1, 90_000, camera_start_generation=1))
    assert matcher.add_frame(0)
    matcher.add_metadata(metadata(2, 180_000, camera_start_generation=2))
    assert matcher.add_frame(0)

    assert (
        matcher.add_metadata(metadata(3, 183_000, camera_start_generation=1))
        is None
    )
    assert matcher.dropped == 1

    matcher.add_frame(3_000)
    match = matcher.add_metadata(metadata(4, 183_000, camera_start_generation=2))
    assert match is not None
    assert match.metadata.camera_start_generation == 2


def test_ordered_gap_fill_recovers_jitter_without_crossing_dropped_inputs() -> None:
    matcher = FrameMetadataMatcher()
    offset = 700_000
    frame_period_ns = 33_333_333
    base_receipt_ns = 5_000_000_000
    residuals = (0, 1_450, -1_300, 180, -1_800, 420, 0)
    dropped_input_indices = {25, 75}
    expected_frame_ids = set(range(120)) - dropped_input_indices
    matches = []

    for source_index in range(120):
        result = matcher.add_metadata(
            metadata(
                source_index,
                offset + source_index * 3_000 + residuals[source_index % 7],
                4_000 + source_index * 33,
            ),
            received_at_client_monotonic_ns=(
                base_receipt_ns + source_index * frame_period_ns
            ),
        )
        if result is not None:
            matches.append(result)

    for source_index in sorted(expected_frame_ids):
        result = matcher.add_frame(
            source_index * 3_000,
            frame_index=source_index,
            time_base_num=1,
            time_base_den=90_000,
            received_at_client_monotonic_ns=(
                base_receipt_ns + source_index * frame_period_ns + 3_000_000
            ),
        )
        if result is not None:
            matches.append(result)
        matches.extend(matcher.drain_matches())

    assert matcher.matched == len(expected_frame_ids)
    assert matcher.anchor_matches > 0
    assert matcher.ordered_gap_matches > 0
    assert matcher.pending_frames == 0
    assert matcher.pending_metadata == 0
    assert matcher.dropped == len(dropped_input_indices)
    assert {match.metadata.frame_id for match in matches} == expected_frame_ids
    assert all(
        match.metadata.frame_id == match.decoded_frame_index for match in matches
    )
    assert max(match.timestamp_match_error_90khz for match in matches) <= 2_000
    assert {match.match_method for match in matches} == {
        "timestamp_anchor",
        "ordered_gap_fill",
    }


def test_unordered_metadata_is_sorted_by_frame_id_before_gap_fill() -> None:
    matcher = FrameMetadataMatcher()
    offset = 900_000
    arrival_order = tuple(
        item
        for group_start in range(0, 120, 3)
        for item in (group_start, group_start + 2, group_start + 1)
    )
    matches = []

    for frame_id in arrival_order:
        matcher.add_metadata(
            metadata(frame_id, offset + frame_id * 3_000, 10_000 + frame_id * 33),
            received_at_client_monotonic_ns=20_000_000_000 + frame_id * 33_333_333,
        )
    for frame_id in range(120):
        first = matcher.add_frame(
            frame_id * 3_000,
            frame_index=frame_id,
            received_at_client_monotonic_ns=(
                20_000_000_000 + frame_id * 33_333_333 + 2_000_000
            ),
        )
        if first is not None:
            matches.append(first)
        matches.extend(matcher.drain_matches())

    assert matcher.matched == 120
    assert matcher.sdk_clock_discontinuities == 0
    assert [match.metadata.frame_id for match in matches] == list(range(120))
    assert all(
        match.metadata.frame_id == match.decoded_frame_index for match in matches
    )


def test_late_metadata_after_committed_anchor_is_dropped() -> None:
    matcher = FrameMetadataMatcher()
    matcher.add_metadata(metadata(1, 90_000))
    assert matcher.add_frame(0)

    assert matcher.add_metadata(metadata(0, 87_000, 900)) is None
    assert matcher.dropped == 1
    assert matcher.pending_metadata == 0


def test_rejects_duplicate_and_records_sdk_clock_discontinuity() -> None:
    matcher = FrameMetadataMatcher()
    matcher.add_metadata(metadata(1, 90_000, 1000))

    with pytest.raises(DuplicateMetadataError):
        matcher.add_metadata(metadata(1, 90_001, 1001))

    matcher.add_metadata(metadata(2, 90_002, 999))
    assert matcher.duplicates == 1
    assert matcher.sdk_clock_discontinuities == 1


def test_out_of_order_metadata_does_not_fake_sdk_clock_discontinuity() -> None:
    matcher = FrameMetadataMatcher()

    matcher.add_metadata(metadata(10, 100_000, 1_000))
    matcher.add_metadata(metadata(12, 106_000, 1_066))
    matcher.add_metadata(metadata(11, 103_000, 1_033))

    assert matcher.sdk_clock_discontinuities == 0


def test_pending_state_is_bounded_and_counts_drops() -> None:
    matcher = FrameMetadataMatcher(max_pending=2)

    matcher.add_frame(10_000)
    matcher.add_frame(10_001)
    matcher.add_frame(10_002)
    matcher.add_metadata(metadata(1, 4))
    matcher.add_metadata(metadata(2, 5))
    matcher.add_metadata(metadata(3, 6))

    assert matcher.pending_frames == 2
    assert matcher.pending_metadata == 2
    assert matcher.dropped == 2
