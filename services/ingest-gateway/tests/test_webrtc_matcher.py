import pytest

from egoglass_ingest_gateway.webrtc_matcher import (
    DuplicateMetadataError,
    FrameMetadataMatcher,
)
from egoglass_ingest_gateway.webrtc_models import VideoFrameMetadata


def metadata(frame_id: int, rtp_timestamp: int, sdk_timestamp_ms: int = 1000) -> VideoFrameMetadata:
    return VideoFrameMetadata(
        frame_id=frame_id,
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
    assert matcher.add_frame(0)
    assert not matcher.add_frame(90_000)
    assert matcher.add_metadata(metadata(2, 180_000, 1050))

    assert matcher.matched == 2
    assert matcher.metadata_origin_90khz == 90_000
    assert matcher.pending_frames == 0
    assert matcher.pending_metadata == 0


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
    assert matcher.add_frame(4_410)
    matcher.add_metadata(metadata(3, 1_009_000, 1100))
    assert not matcher.add_frame(8_909)

    assert matcher.matched == 2
    assert matcher.max_timestamp_match_error_90khz == 90


def test_rejects_duplicate_and_records_sdk_clock_discontinuity() -> None:
    matcher = FrameMetadataMatcher()
    matcher.add_metadata(metadata(1, 90_000, 1000))

    with pytest.raises(DuplicateMetadataError):
        matcher.add_metadata(metadata(1, 90_001, 1001))

    matcher.add_metadata(metadata(2, 90_002, 999))
    assert matcher.duplicates == 1
    assert matcher.sdk_clock_discontinuities == 1


def test_pending_state_is_bounded_and_counts_drops() -> None:
    matcher = FrameMetadataMatcher(max_pending=2)

    matcher.add_frame(100)
    matcher.add_frame(101)
    matcher.add_frame(102)
    matcher.add_metadata(metadata(1, 4))
    matcher.add_metadata(metadata(2, 5))
    matcher.add_metadata(metadata(3, 6))

    assert matcher.pending_frames == 2
    assert matcher.pending_metadata == 2
    assert matcher.dropped == 2
