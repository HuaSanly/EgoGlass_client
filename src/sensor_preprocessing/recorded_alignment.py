"""Derive a strict session clock from immutable recorded timing evidence."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .clock_mapping import (
    ClockMappingSegment,
    SegmentedClockMapper,
    frame_callback_observation,
    glasses_elapsed_source_instance_id,
    mp4_source_instance_id,
    rtp_match_error_to_uncertainty_ns,
)
from .models import ClockId, RawFrameRef, RawImuSample, TimeStatus

_ARTIFACT_SCHEMA_VERSION = "1.0"
_ARTIFACT_CONTRACT_ID = "sensor-clock-mapping-v1"
_FIT_PROVENANCE_VERSION = "v2"


class RecordedAlignmentError(RuntimeError):
    """Recorded clocks do not contain enough evidence for strict preprocessing."""


@dataclass(frozen=True, slots=True)
class RecordedClockMapping:
    """Derived mapper plus the evidence identity persisted beside the raw session."""

    mapper: SegmentedClockMapper
    evidence_sha256: str
    frame_evidence_count: int
    imu_evidence_count: int

    def to_json_dict(self) -> dict[str, object]:
        """Return the deterministic, versioned derived-artifact payload."""

        return {
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "contract_id": _ARTIFACT_CONTRACT_ID,
            "session_id": self.mapper.session_id,
            "evidence_sha256": self.evidence_sha256,
            "frame_evidence_count": self.frame_evidence_count,
            "imu_evidence_count": self.imu_evidence_count,
            "segments": [_segment_payload(segment) for segment in self.mapper.segments],
        }


def derive_recorded_clock_mapping(
    session_id: str,
    frames: tuple[RawFrameRef, ...],
    imu_samples: tuple[RawImuSample, ...],
) -> RecordedClockMapping:
    """Fit device and MP4 clocks to one session timeline without mutating raw data."""

    if not session_id.strip():
        raise RecordedAlignmentError("recorded alignment session_id cannot be empty")
    if not frames:
        raise RecordedAlignmentError("recorded alignment requires indexed video frames")
    if not imu_samples:
        raise RecordedAlignmentError("recorded alignment requires IMU samples")
    if any(frame.session_id != session_id for frame in frames) or any(
        sample.session_id != session_id for sample in imu_samples
    ):
        raise RecordedAlignmentError("recorded timing evidence spans multiple sessions")

    evidence_sha256 = _evidence_digest(session_id, frames, imu_samples)
    provenance_id = (
        f"recorded-clock-fit-{_FIT_PROVENANCE_VERSION}-{evidence_sha256}"
    )
    device_segments = _derive_device_segments(
        session_id,
        frames,
        imu_samples,
        provenance_id,
    )
    device_mapper = SegmentedClockMapper(session_id, device_segments)
    media_segments = _derive_media_segments(
        session_id,
        frames,
        device_mapper,
        provenance_id,
    )
    return RecordedClockMapping(
        mapper=SegmentedClockMapper(session_id, (*device_segments, *media_segments)),
        evidence_sha256=evidence_sha256,
        frame_evidence_count=len(frames),
        imu_evidence_count=len(imu_samples),
    )


def persist_recorded_clock_mapping(
    mapping: RecordedClockMapping,
    session_directory: str | Path,
) -> Path:
    """Atomically persist the derived mapping outside the immutable raw database."""

    session_path = Path(session_directory).resolve(strict=True)
    if session_path.name != mapping.mapper.session_id:
        raise RecordedAlignmentError("clock mapping session does not match its directory")
    output_directory = session_path / "derived" / "sensor-preprocessing"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "clock-mapping-v1.json"
    encoded = (
        json.dumps(
            mapping.to_json_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if output_path.is_file() and output_path.read_bytes() == encoded:
        return output_path
    temporary_path = output_directory / ".clock-mapping-v1.json.tmp"
    temporary_path.write_bytes(encoded)
    os.replace(temporary_path, output_path)
    return output_path


def _derive_device_segments(
    session_id: str,
    frames: tuple[RawFrameRef, ...],
    imu_samples: tuple[RawImuSample, ...],
    provenance_id: str,
) -> tuple[ClockMappingSegment, ...]:
    evidence_by_connection: dict[str, list[tuple[int, int]]] = {}
    for sample in imu_samples:
        evidence_by_connection.setdefault(sample.connection_session_id, []).append(
            (
                sample.sensor_event_monotonic_ns,
                sample.received_at_client_perf_counter_ns,
            )
        )
    for frame in frames:
        if (
            frame.connection_session_id is not None
            and frame.video_at_monotonic_ns is not None
            and frame.metadata_received_at_client_perf_counter_ns is not None
        ):
            evidence_by_connection.setdefault(frame.connection_session_id, []).append(
                (
                    frame.video_at_monotonic_ns,
                    frame.metadata_received_at_client_perf_counter_ns,
                )
            )
    if not evidence_by_connection:
        raise RecordedAlignmentError("recording has no device-clock evidence")

    connection_fits: list[tuple[str, int, int, int, int]] = []
    for connection_id, evidence in evidence_by_connection.items():
        if len(evidence) < 2:
            raise RecordedAlignmentError(
                f"connection {connection_id} has fewer than two clock observations"
            )
        sources = [source_ns for source_ns, _client_ns in evidence]
        offsets = [client_ns - source_ns for source_ns, client_ns in evidence]
        source_from = min(sources)
        source_to = max(sources)
        if source_from == source_to:
            raise RecordedAlignmentError(
                f"connection {connection_id} clock evidence has no time span"
            )
        minimum_offset = min(offsets)
        receipt_jitter_ns = max(offsets) - minimum_offset
        estimated_client_anchor = source_from + minimum_offset
        connection_fits.append(
            (
                connection_id,
                source_from,
                source_to,
                estimated_client_anchor,
                receipt_jitter_ns,
            )
        )

    session_client_origin = min(fit[3] for fit in connection_fits)
    segments = [
        ClockMappingSegment(
            session_id=session_id,
            source_clock_id=ClockId.GLASSES_ELAPSED_REALTIME_NS,
            source_instance_id=glasses_elapsed_source_instance_id(
                session_id,
                connection_id,
            ),
            segment_index=0,
            source_from=source_from,
            source_to=source_to,
            source_anchor=source_from,
            target_anchor_ns=estimated_client_anchor - session_client_origin,
            scale_numerator_ns=1,
            scale_denominator_source_units=1,
            uncertainty_ns=receipt_jitter_ns,
            status=TimeStatus.ESTIMATED,
            fit_method="identity_device_clock_minimum_delay_anchor",
            provenance_id=provenance_id,
            uncertainty_basis="observed client-receipt offset range",
        )
        for (
            connection_id,
            source_from,
            source_to,
            estimated_client_anchor,
            receipt_jitter_ns,
        ) in sorted(connection_fits, key=lambda fit: (fit[3], fit[0]))
    ]
    return tuple(segments)


def _derive_media_segments(
    session_id: str,
    frames: tuple[RawFrameRef, ...],
    device_mapper: SegmentedClockMapper,
    provenance_id: str,
) -> tuple[ClockMappingSegment, ...]:
    frames_by_clip: dict[str, list[RawFrameRef]] = {}
    for frame in frames:
        frames_by_clip.setdefault(frame.clip_id, []).append(frame)

    segments: list[ClockMappingSegment] = []
    for clip_id, clip_frames in sorted(frames_by_clip.items()):
        ordered_frames = sorted(clip_frames, key=lambda frame: frame.frame_index)
        time_bases = {
            (
                frame.mp4_timestamp.time_base_numerator,
                frame.mp4_timestamp.time_base_denominator,
            )
            for frame in ordered_frames
        }
        if len(time_bases) != 1:
            raise RecordedAlignmentError(f"clip {clip_id} changes MP4 time base")
        time_base_numerator, time_base_denominator = next(iter(time_bases))
        mapped_evidence: list[tuple[int, int, int]] = []
        for frame in ordered_frames:
            observation = frame_callback_observation(frame)
            if observation is None:
                continue
            estimate = device_mapper.map(
                observation,
                additional_uncertainty_ns=rtp_match_error_to_uncertainty_ns(
                    frame.timestamp_match_error_90khz or 0
                ),
                additional_status=TimeStatus.ESTIMATED,
            )
            if estimate.session_time_ns is None or estimate.uncertainty_ns is None:
                continue
            pts = frame.mp4_timestamp.pts
            mapped_evidence.append(
                (pts, estimate.session_time_ns, estimate.uncertainty_ns)
            )
        if len(mapped_evidence) < 2:
            raise RecordedAlignmentError(
                f"clip {clip_id} has fewer than two camera-to-MP4 matches"
            )

        ordered_evidence = sorted(mapped_evidence)
        if any(
            current_pts <= previous_pts or current_time_ns <= previous_time_ns
            for (
                previous_pts,
                previous_time_ns,
                _previous_uncertainty_ns,
            ), (
                current_pts,
                current_time_ns,
                _current_uncertainty_ns,
            ) in zip(ordered_evidence, ordered_evidence[1:], strict=False)
        ):
            raise RecordedAlignmentError(
                f"clip {clip_id} camera-to-MP4 evidence is not strictly increasing"
            )

        scale = _least_squares_scale(ordered_evidence)
        if scale <= 0:
            raise RecordedAlignmentError(
                f"clip {clip_id} camera-to-MP4 fit has a non-positive scale"
            )
        median_offset = _median_fraction(
            [
                Fraction(session_time_ns) - pts * scale
                for pts, session_time_ns, _uncertainty_ns in ordered_evidence
            ]
        )
        source_from = min(frame.mp4_timestamp.pts for frame in ordered_frames)
        source_to = max(frame.mp4_timestamp.pts for frame in ordered_frames)
        target_anchor_ns = _round_fraction(median_offset + source_from * scale)
        residual_bound_ns = max(
            _ceil_fraction_abs(Fraction(session_time_ns) - (pts * scale + median_offset))
            + uncertainty_ns
            for pts, session_time_ns, uncertainty_ns in ordered_evidence
        )
        segments.append(
            ClockMappingSegment(
                session_id=session_id,
                source_clock_id=ClockId.MP4_PRESENTATION_TICKS,
                source_instance_id=mp4_source_instance_id(
                    session_id,
                    clip_id,
                    time_base_numerator,
                    time_base_denominator,
                ),
                segment_index=0,
                source_from=source_from,
                source_to=source_to,
                source_anchor=source_from,
                target_anchor_ns=target_anchor_ns,
                scale_numerator_ns=scale.numerator,
                scale_denominator_source_units=scale.denominator,
                uncertainty_ns=residual_bound_ns,
                status=TimeStatus.ESTIMATED,
                fit_method="least_squares_scale_median_offset_camera_to_mp4",
                provenance_id=provenance_id,
                uncertainty_basis=(
                    "maximum affine callback residual plus device mapping bound"
                ),
            )
        )
    return tuple(segments)


def _evidence_digest(
    session_id: str,
    frames: tuple[RawFrameRef, ...],
    imu_samples: tuple[RawImuSample, ...],
) -> str:
    payload = {
        "session_id": session_id,
        "frames": [
            {
                "clip_id": frame.clip_id,
                "frame_index": frame.frame_index,
                "pts": frame.mp4_timestamp.pts,
                "time_base_numerator": frame.mp4_timestamp.time_base_numerator,
                "time_base_denominator": frame.mp4_timestamp.time_base_denominator,
                "connection_session_id": frame.connection_session_id,
                "video_at_monotonic_ns": frame.video_at_monotonic_ns,
                "received_at_client_perf_counter_ns": (
                    frame.metadata_received_at_client_perf_counter_ns
                ),
                "timestamp_match_error_90khz": frame.timestamp_match_error_90khz,
            }
            for frame in sorted(frames, key=lambda item: (item.clip_id, item.frame_index))
        ],
        "imu_samples": [
            {
                "sample_id": sample.sample_id,
                "connection_session_id": sample.connection_session_id,
                "sensor_event_monotonic_ns": sample.sensor_event_monotonic_ns,
                "received_at_client_perf_counter_ns": (
                    sample.received_at_client_perf_counter_ns
                ),
            }
            for sample in sorted(imu_samples, key=lambda item: item.sample_id)
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _segment_payload(segment: ClockMappingSegment) -> dict[str, object]:
    return {
        "schema_version": segment.schema_version,
        "clock_mapping_id": segment.clock_mapping_id,
        "session_id": segment.session_id,
        "source_clock_id": segment.source_clock_id.value,
        "source_instance_id": segment.source_instance_id,
        "segment_index": segment.segment_index,
        "source_from": segment.source_from,
        "source_to": segment.source_to,
        "source_anchor": segment.source_anchor,
        "target_clock_id": segment.target_clock_id.value,
        "target_anchor_ns": segment.target_anchor_ns,
        "scale_numerator_ns": segment.scale_numerator_ns,
        "scale_denominator_source_units": segment.scale_denominator_source_units,
        "uncertainty_ns": segment.uncertainty_ns,
        "status": segment.status.value,
        "fit_method": segment.fit_method,
        "provenance_id": segment.provenance_id,
        "uncertainty_basis": segment.uncertainty_basis,
        "evidence_id": segment.evidence_id,
    }


def _median_fraction(values: list[Fraction]) -> Fraction:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _least_squares_scale(evidence: list[tuple[int, int, int]]) -> Fraction:
    count = len(evidence)
    sum_source = sum(source for source, _target, _uncertainty in evidence)
    sum_target = sum(target for _source, target, _uncertainty in evidence)
    covariance = sum(
        (count * source - sum_source) * (count * target - sum_target)
        for source, target, _uncertainty in evidence
    )
    variance = sum(
        (count * source - sum_source) ** 2
        for source, _target, _uncertainty in evidence
    )
    if variance == 0:
        raise RecordedAlignmentError("camera-to-MP4 evidence has no source time span")
    return Fraction(covariance, variance)


def _round_fraction(value: Fraction) -> int:
    quotient, remainder = divmod(value.numerator, value.denominator)
    return quotient + int(remainder * 2 >= value.denominator)


def _ceil_fraction_abs(value: Fraction) -> int:
    absolute = abs(value)
    quotient, remainder = divmod(absolute.numerator, absolute.denominator)
    return quotient + int(remainder != 0)
