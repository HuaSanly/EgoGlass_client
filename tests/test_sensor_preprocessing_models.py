from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest

from perception.sensor_preprocessing import Mp4Timestamp, TimeEstimate, TimeStatus


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
