from dataclasses import FrozenInstanceError
from fractions import Fraction
from pathlib import Path

import pytest

from perception.sensor_preprocessing import (
    AlignmentStatus,
    ImuSensorType,
    MetadataMatchStatus,
    Mp4Timestamp,
    RawFrameRef,
    RawImuSample,
    StoredAlignment,
    TimeEstimate,
    TimeStatus,
)


def _pending_alignment() -> StoredAlignment:
    return StoredAlignment(
        status=AlignmentStatus.PENDING,
        session_time_ns=None,
        uncertainty_ns=None,
        clock_mapping_segment_id=None,
    )


def _matched_frame_values(tmp_path: Path) -> dict[str, object]:
    return {
        "video_frame_row_id": 1,
        "session_id": "session-1",
        "clip_id": "clip-1",
        "frame_index": 0,
        "media_path": (tmp_path / "clip.mp4").resolve(),
        "mp4_timestamp": Mp4Timestamp(0, 1, 30),
        "metadata_match_status": MetadataMatchStatus.EXACT,
        "video_frame_metadata_id": "metadata-1",
        "frame_metadata_match_id": 1,
        "timestamp_match_error_90khz": 0,
        "connection_session_id": "connection-1",
        "camera_start_generation": 1,
        "frame_id": 0,
        "captured_at_rokid_sdk_ms": 1_000,
        "received_at_elapsed_realtime_ns": 2_000,
        "video_at_monotonic_ns": 1_900,
        "rtp_timestamp_90khz": 90,
        "received_at_client_perf_counter_ns": 2_100,
        "metadata_received_at_client_perf_counter_ns": 2_050,
        "width": 1280,
        "height": 720,
        "rotation_degrees": 0,
        "capture_config_id": "720p30",
        "source_frame_timestamp": Mp4Timestamp(0, 1, 90_000),
        "stored_alignment": _pending_alignment(),
    }


def test_mp4_timestamp_preserves_exact_rational_time() -> None:
    timestamp = Mp4Timestamp(
        pts=3_003,
        time_base_numerator=1,
        time_base_denominator=90_000,
    )

    assert timestamp.presentation_time_seconds == Fraction(1_001, 30_000)


@pytest.mark.parametrize("numerator, denominator", [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_mp4_timestamp_rejects_invalid_time_base(numerator: int, denominator: int) -> None:
    with pytest.raises(ValueError, match="time-base"):
        Mp4Timestamp(
            pts=0,
            time_base_numerator=numerator,
            time_base_denominator=denominator,
        )


def test_unavailable_time_preserves_only_source_timestamp() -> None:
    estimate = TimeEstimate(
        source_clock_id="rokid_sdk_ms",
        source_timestamp=1_000,
        status=TimeStatus.UNAVAILABLE,
    )

    assert estimate.session_time_ns is None
    assert estimate.uncertainty_ns is None
    assert estimate.clock_mapping_id is None


@pytest.mark.parametrize("status", [TimeStatus.VERIFIED, TimeStatus.ESTIMATED])
def test_resolved_time_requires_complete_mapping_provenance(status: TimeStatus) -> None:
    estimate = TimeEstimate(
        source_clock_id="sensor_event_monotonic_ns",
        source_timestamp=1_000,
        status=status,
        session_time_ns=200,
        uncertainty_ns=10,
        clock_mapping_id="mapping-1",
    )

    assert estimate.status is status
    assert estimate.session_time_ns == 200


@pytest.mark.parametrize(
    "session_time_ns, uncertainty_ns, clock_mapping_id",
    [(None, 10, "mapping-1"), (200, None, "mapping-1"), (200, 10, None)],
)
def test_resolved_time_rejects_incomplete_mapping_provenance(
    session_time_ns: int | None,
    uncertainty_ns: int | None,
    clock_mapping_id: str | None,
) -> None:
    with pytest.raises(ValueError, match="requires"):
        TimeEstimate(
            source_clock_id="sensor_event_monotonic_ns",
            source_timestamp=1_000,
            status=TimeStatus.VERIFIED,
            session_time_ns=session_time_ns,
            uncertainty_ns=uncertainty_ns,
            clock_mapping_id=clock_mapping_id,
        )


def test_unavailable_time_rejects_derived_values() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        TimeEstimate(
            source_clock_id="rokid_sdk_ms",
            source_timestamp=1_000,
            status=TimeStatus.UNAVAILABLE,
            session_time_ns=200,
        )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"source_clock_id": ""}, "source_clock_id"),
        ({"source_clock_id": "   "}, "source_clock_id"),
        ({"source_timestamp": -1}, "source_timestamp"),
        ({"session_time_ns": -1}, "cannot be negative"),
        ({"uncertainty_ns": -1}, "cannot be negative"),
    ],
)
def test_time_estimate_rejects_invalid_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "source_clock_id": "sensor_event_monotonic_ns",
        "source_timestamp": 1_000,
        "status": TimeStatus.VERIFIED,
        "session_time_ns": 200,
        "uncertainty_ns": 10,
        "clock_mapping_id": "mapping-1",
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        TimeEstimate(**values)  # type: ignore[arg-type]


def test_time_estimate_rejects_non_enum_status() -> None:
    with pytest.raises(TypeError, match="TimeStatus"):
        TimeEstimate(
            source_clock_id="rokid_sdk_ms",
            source_timestamp=1_000,
            status="unavailable",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("clock_mapping_id", ["", "   "])
def test_resolved_time_rejects_empty_mapping_id(clock_mapping_id: str) -> None:
    with pytest.raises(ValueError, match="clock_mapping_id"):
        TimeEstimate(
            source_clock_id="sensor_event_monotonic_ns",
            source_timestamp=1_000,
            status=TimeStatus.ESTIMATED,
            session_time_ns=200,
            uncertainty_ns=10,
            clock_mapping_id=clock_mapping_id,
        )


def test_sensor_records_are_immutable() -> None:
    timestamp = Mp4Timestamp(pts=0, time_base_numerator=1, time_base_denominator=1)

    with pytest.raises(FrozenInstanceError):
        timestamp.pts = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "session_time_ns, uncertainty_ns, mapping_id",
    [(1, None, None), (None, 1, None), (None, None, "mapping-1")],
)
def test_stored_alignment_rejects_partial_results(
    session_time_ns: int | None,
    uncertainty_ns: int | None,
    mapping_id: str | None,
) -> None:
    with pytest.raises(ValueError, match="requires all mapped fields"):
        StoredAlignment(
            status=AlignmentStatus.MAPPED,
            session_time_ns=session_time_ns,
            uncertainty_ns=uncertainty_ns,
            clock_mapping_segment_id=mapping_id,
        )


def test_pending_alignment_rejects_mapped_fields() -> None:
    with pytest.raises(ValueError, match="pending alignment"):
        StoredAlignment(
            status=AlignmentStatus.PENDING,
            session_time_ns=1,
            uncertainty_ns=1,
            clock_mapping_segment_id="mapping-1",
        )


@pytest.mark.parametrize(
    "android_sensor_type, unit",
    [(4, "m_s2"), (1, "rad_s"), (1, "meters_per_second_squared")],
)
def test_imu_sample_rejects_type_or_unit_mismatch(
    android_sensor_type: int,
    unit: str,
) -> None:
    with pytest.raises(ValueError, match="do not match"):
        RawImuSample(
            sample_id=1,
            session_id="session-1",
            connection_session_id="connection-1",
            sensor_type=ImuSensorType.ACCELEROMETER,
            android_sensor_type=android_sensor_type,
            sequence_number=0,
            sensor_event_monotonic_ns=1_000,
            received_at_elapsed_realtime_ns=1_100,
            received_at_client_perf_counter_ns=1_200,
            accuracy=3,
            values=(0.1, 0.2, 9.8),
            unit=unit,
            stored_alignment=_pending_alignment(),
        )


def test_matched_frame_requires_complete_camera_metadata(tmp_path: Path) -> None:
    values = _matched_frame_values(tmp_path)
    values["width"] = None

    with pytest.raises(ValueError, match="requires complete"):
        RawFrameRef(**values)  # type: ignore[arg-type]


def test_unmatched_frame_rejects_camera_metadata(tmp_path: Path) -> None:
    values = _matched_frame_values(tmp_path)
    values.update(
        {
            "metadata_match_status": MetadataMatchStatus.UNMATCHED,
            "connection_session_id": None,
            "camera_start_generation": None,
            "frame_id": None,
            "captured_at_rokid_sdk_ms": None,
            "received_at_elapsed_realtime_ns": None,
            "video_at_monotonic_ns": None,
            "rtp_timestamp_90khz": None,
            "metadata_received_at_client_perf_counter_ns": None,
            "width": None,
            "height": None,
            "rotation_degrees": None,
            "capture_config_id": None,
        }
    )

    with pytest.raises(ValueError, match="cannot contain"):
        RawFrameRef(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "status, error",
    [
        (MetadataMatchStatus.EXACT, 1),
        (MetadataMatchStatus.WITHIN_TOLERANCE, 0),
    ],
)
def test_frame_rejects_match_status_error_disagreement(
    tmp_path: Path,
    status: MetadataMatchStatus,
    error: int,
) -> None:
    values = _matched_frame_values(tmp_path)
    values["metadata_match_status"] = status
    values["timestamp_match_error_90khz"] = error

    with pytest.raises(ValueError, match="timestamp error"):
        RawFrameRef(**values)  # type: ignore[arg-type]
