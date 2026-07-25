from __future__ import annotations

from ingest_gateway.webrtc_matcher import (
    MAX_TIMESTAMP_ERROR_90KHZ,
    FrameMetadataMatcher,
)
from ingest_gateway.webrtc_models import VideoFrameMetadata


def _metadata(frame_id: int, rtp_timestamp: int) -> VideoFrameMetadata:
    return VideoFrameMetadata(
        frame_id=frame_id,
        camera_start_generation=4,
        captured_at_rokid_sdk_ms=10_000 + frame_id * 33,
        received_at_elapsed_realtime_ns=20_000_000_000 + frame_id,
        video_at_monotonic_ns=20_000_000_000 + frame_id,
        rtp_timestamp_90khz=rtp_timestamp,
        width=1280,
        height=720,
        rotation_degrees=0,
        capture_config_id="720p30",
    )


def test_device_shaped_startup_drop_calibrates_to_dominant_offset_cluster() -> None:
    matcher = FrameMetadataMatcher()
    startup_origin = 1_902_440_120
    expected_offset = 1_902_537_900
    period_ns = 33_333_333
    base_ns = 8_000_000_000
    residuals = (-710, -404, -211, 0, 137, 388, 744)

    for index in range(31):
        matcher.add_metadata(
            _metadata(index, (startup_origin + index * 3_000) & 0xFFFFFFFF),
            received_at_client_monotonic_ns=base_ns - (31 - index) * period_ns,
        )

    receipts = []
    for index in range(100):
        receipt_ns = base_ns + index * period_ns
        receipts.append(receipt_ns)
        matcher.add_metadata(
            _metadata(
                1_000 + index,
                (
                    expected_offset
                    + index * 3_000
                    + residuals[index % len(residuals)]
                )
                & 0xFFFFFFFF,
            ),
            received_at_client_monotonic_ns=receipt_ns,
        )

    matches = []
    for index in range(100):
        nearest_receipt = receipts[index + 1] if index < 28 else receipts[index]
        first = matcher.add_frame(
            index * 3_000,
            frame_index=index,
            time_base_num=1,
            time_base_den=90_000,
            received_at_client_monotonic_ns=nearest_receipt + 800_000,
        )
        if first is not None:
            matches.append(first)
        matches.extend(matcher.drain_matches())

    assert matcher.metadata_origin_90khz == expected_offset
    assert matcher.metadata_origin_90khz != startup_origin
    assert matcher.matched == 100
    assert {match.metadata.frame_id for match in matches} == set(range(1_000, 1_100))
    assert max(match.timestamp_match_error_90khz for match in matches) <= 744
    assert matcher.max_timestamp_match_error_90khz <= MAX_TIMESTAMP_ERROR_90KHZ


def test_long_device_shaped_stream_keeps_metadata_coverage_above_95_percent() -> None:
    matcher = FrameMetadataMatcher()
    offset = 1_347_000_000
    period_ns = 33_333_333
    base_ns = 30_000_000_000
    residuals = (53, -281, 1_450, -1_300, 181, -1_800, 420, -99, 0)
    dropped_inputs = {119, 287, 463}
    expected_ids = set(range(603)) - dropped_inputs
    matches = []

    for source_index in range(603):
        first = matcher.add_metadata(
            _metadata(
                source_index,
                (
                    offset
                    + source_index * 3_000
                    + residuals[source_index % len(residuals)]
                )
                & 0xFFFFFFFF,
            ),
            received_at_client_monotonic_ns=base_ns + source_index * period_ns,
        )
        if first is not None:
            matches.append(first)
        if source_index not in dropped_inputs:
            second = matcher.add_frame(
                source_index * 3_000,
                frame_index=source_index,
                time_base_num=1,
                time_base_den=90_000,
                received_at_client_monotonic_ns=(
                    base_ns + source_index * period_ns + 4_000_000
                ),
            )
            if second is not None:
                matches.append(second)
        matches.extend(matcher.drain_matches())

    coverage = matcher.matched / len(expected_ids)
    assert coverage >= 0.95
    assert matcher.matched == len(expected_ids)
    assert {match.metadata.frame_id for match in matches} == expected_ids
    assert matcher.pending_frames == 0
    assert matcher.pending_metadata == 0
    assert matcher.dropped == len(dropped_inputs)
