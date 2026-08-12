from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TextIO
from uuid import uuid4

from pydantic import ValidationError

from schemas.recording import (
    CaptureRecordingManifest,
    CaptureRecordingProvenance,
    CaptureRecordingQualityReport,
    CaptureRecordingState,
    FrameMetadataMatchStatus,
    ImuSensorType,
    RecordingArtifacts,
    RecordingFileInfo,
    RecordingFrameRow,
    RecordingImuRow,
    RecordingOutput,
    RecordingQualityCheck,
    RecordingQualityCounts,
    RecordingSummary,
    RecordingTimeOrigin,
)

FRAME_CSV_COLUMNS = (
    "frame_index",
    "recording_time_ns",
    "mp4_pts",
    "mp4_time_base_num",
    "mp4_time_base_den",
    "connection_session_id",
    "frame_id",
    "camera_start_generation",
    "captured_at_rokid_sdk_ms",
    "received_at_elapsed_realtime_ns",
    "video_at_monotonic_ns",
    "rtp_timestamp_90khz",
    "received_at_client_monotonic_ns",
    "metadata_match_status",
    "timestamp_match_error_90khz",
)

IMU_CSV_COLUMNS = (
    "sample_index",
    "recording_time_ns",
    "connection_session_id",
    "sensor_type",
    "sequence_number",
    "sensor_event_monotonic_ns",
    "received_at_elapsed_realtime_ns",
    "received_at_client_monotonic_ns",
    "accuracy",
    "x",
    "y",
    "z",
    "inside_video_span",
)

_REQUIRED_TOP_LEVEL_ENTRIES = {
    "manifest.json",
    "video.mp4",
    "imu.csv",
    "frames.csv",
    "quality.json",
    "annotations",
    "derived",
}
_INTEGER_PATTERN = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")


class CaptureRecordingError(RuntimeError):
    """Base error for recording storage failures."""


class CaptureRecordingReadError(CaptureRecordingError):
    """Raised when a recording fails its persisted-data contract."""


class CaptureRecordingWriter:
    """Write one recording into a private directory and publish it atomically."""

    def __init__(
        self,
        recordings_root: Path,
        recording_id: str,
        *,
        video_profile: RecordingOutput,
        time_origin: RecordingTimeOrigin,
        provenance: CaptureRecordingProvenance | None = None,
    ) -> None:
        self.recordings_root = Path(recordings_root)
        self.recording_id = recording_id
        self.partial_directory = self.recordings_root / f".recording-{recording_id}.partial"
        self.final_directory = self.recordings_root / recording_id
        self.video_path = self.partial_directory / "video.mp4"
        self.imu_path = self.partial_directory / "imu.csv"
        self.frames_path = self.partial_directory / "frames.csv"
        self._manifest_path = self.partial_directory / "manifest.json"
        self._quality_path = self.partial_directory / "quality.json"
        self._video_profile = video_profile
        self._time_origin = time_origin
        self._provenance = provenance or CaptureRecordingProvenance()
        self._frames: list[RecordingFrameRow] = []
        self._imu_rows: list[RecordingImuRow] = []
        self._frames_file: TextIO | None = None
        self._imu_file: TextIO | None = None
        self._frames_writer: csv.DictWriter[str] | None = None
        self._imu_writer: csv.DictWriter[str] | None = None
        self._closed = False

        # Validation happens before any path is created.
        self._draft_manifest(CaptureRecordingState.RECORDING)
        self.recordings_root.mkdir(parents=True, exist_ok=True)
        if self.partial_directory.exists() or self.final_directory.exists():
            raise FileExistsError(f"recording already exists: {recording_id}")
        self.partial_directory.mkdir()
        (self.partial_directory / "annotations").mkdir()
        (self.partial_directory / "derived").mkdir()
        self.video_path.touch(exist_ok=False)
        self._open_csv_files(mode="w")
        self._write_draft_files()

    @classmethod
    def create(
        cls,
        recordings_root: Path,
        *,
        recording_id: str | None = None,
        video_profile: RecordingOutput | None = None,
        countdown_started_at_unix_ns: int | None = None,
        countdown_started_at_client_monotonic_ns: int | None = None,
        provenance: CaptureRecordingProvenance | None = None,
    ) -> CaptureRecordingWriter:
        unix_ns = (
            time.time_ns() if countdown_started_at_unix_ns is None else countdown_started_at_unix_ns
        )
        monotonic_ns = (
            time.perf_counter_ns()
            if countdown_started_at_client_monotonic_ns is None
            else countdown_started_at_client_monotonic_ns
        )
        return cls(
            recordings_root,
            recording_id or uuid4().hex,
            video_profile=video_profile or RecordingOutput(),
            time_origin=RecordingTimeOrigin(
                countdown_started_at_unix_ns=unix_ns,
                countdown_started_at_client_monotonic_ns=monotonic_ns,
            ),
            provenance=provenance,
        )

    @classmethod
    def recover(cls, recordings_root: Path, recording_id: str) -> CaptureRecordingWriter:
        root = Path(recordings_root)
        partial_directory = root / f".recording-{recording_id}.partial"
        manifest = _read_manifest(partial_directory / "manifest.json")
        if manifest.recording_id != recording_id:
            raise CaptureRecordingReadError(
                "partial directory recording ID does not match manifest"
            )
        if manifest.state is CaptureRecordingState.COMPLETE:
            raise CaptureRecordingReadError(
                "completed manifest cannot remain in a partial directory"
            )
        _validate_layout(partial_directory)
        frames = _read_frame_csv(partial_directory / "frames.csv")
        imu_rows = _read_imu_csv(partial_directory / "imu.csv")

        writer = cls.__new__(cls)
        writer.recordings_root = root
        writer.recording_id = recording_id
        writer.partial_directory = partial_directory
        writer.final_directory = root / recording_id
        writer.video_path = partial_directory / "video.mp4"
        writer.imu_path = partial_directory / "imu.csv"
        writer.frames_path = partial_directory / "frames.csv"
        writer._manifest_path = partial_directory / "manifest.json"
        writer._quality_path = partial_directory / "quality.json"
        writer._video_profile = manifest.video_profile
        writer._time_origin = manifest.time_origin
        writer._provenance = manifest.provenance
        writer._frames = list(frames)
        writer._imu_rows = list(imu_rows)
        writer._frames_file = None
        writer._imu_file = None
        writer._frames_writer = None
        writer._imu_writer = None
        writer._closed = False
        writer._open_csv_files(mode="a")
        return writer

    def append_frame(self, row: RecordingFrameRow) -> None:
        self._require_open()
        expected_index = len(self._frames)
        if row.frame_index != expected_index:
            raise ValueError(f"frame_index must be contiguous; expected {expected_index}")
        if self._frames:
            previous = self._frames[-1]
            if row.recording_time_ns <= previous.recording_time_ns:
                raise ValueError("frame recording_time_ns must be strictly increasing")
            if not _mp4_time_is_after(row, previous):
                raise ValueError("frame MP4 presentation time must be strictly increasing")
        assert self._frames_writer is not None and self._frames_file is not None
        self._frames_writer.writerow(_frame_to_csv(row))
        self._frames_file.flush()
        self._frames.append(row)

    def append_imu(self, row: RecordingImuRow) -> None:
        self._require_open()
        expected_index = len(self._imu_rows)
        if row.sample_index != expected_index:
            raise ValueError(f"sample_index must be contiguous; expected {expected_index}")
        if self._imu_rows and row.recording_time_ns < self._imu_rows[-1].recording_time_ns:
            raise ValueError("IMU recording_time_ns must be monotonic")
        assert self._imu_writer is not None and self._imu_file is not None
        self._imu_writer.writerow(_imu_to_csv(row))
        self._imu_file.flush()
        self._imu_rows.append(row)

    def finalize(
        self,
        *,
        ended_at_unix_ns: int | None = None,
        telemetry_queue_overflow_count: int = 0,
        timestamp_mapping_residual_ns: int | None = None,
    ) -> CaptureRecordingReader:
        self._require_open()
        if telemetry_queue_overflow_count < 0:
            raise ValueError("telemetry queue overflow count cannot be negative")
        if not self._frames:
            raise CaptureRecordingError("cannot finalize a recording without video frames")
        self._close_csv_files()
        self._normalize_imu_video_span()
        self._frames = list(_read_frame_csv(self.frames_path))
        self._imu_rows = list(_read_imu_csv(self.imu_path))
        if self.video_path.stat().st_size < 1:
            raise CaptureRecordingError("cannot finalize an empty video.mp4")

        quality = _build_quality_report(
            recording_id=self.recording_id,
            frames=self._frames,
            imu_rows=self._imu_rows,
            telemetry_queue_overflow_count=telemetry_queue_overflow_count,
            timestamp_mapping_residual_ns=timestamp_mapping_residual_ns,
        )
        _atomic_write_json(self._quality_path, quality.model_dump(mode="json"))
        artifacts = RecordingArtifacts(
            video=_file_info(self.video_path),
            imu=_file_info(self.imu_path),
            frames=_file_info(self.frames_path),
            quality=_file_info(self._quality_path),
        )
        end_ns = time.time_ns() if ended_at_unix_ns is None else ended_at_unix_ns
        first_frame_time = self._frames[0].recording_time_ns
        last_frame_time = self._frames[-1].recording_time_ns
        time_origin = self._time_origin.model_copy(
            update={
                "first_video_frame_recording_time_ns": first_frame_time,
                "last_video_frame_recording_time_ns": last_frame_time,
            }
        )
        manifest = CaptureRecordingManifest(
            recording_id=self.recording_id,
            state=CaptureRecordingState.COMPLETE,
            started_at_unix_ns=time_origin.countdown_started_at_unix_ns,
            ended_at_unix_ns=end_ns,
            duration_ns=last_frame_time,
            video_profile=self._video_profile,
            frame_count=len(self._frames),
            imu_sample_count=len(self._imu_rows),
            time_origin=time_origin,
            provenance=self._provenance,
            artifacts=artifacts,
        )
        _atomic_write_json(self._manifest_path, manifest.model_dump(mode="json"))
        _validate_recording(self.partial_directory, verify_hashes=True, allow_partial_name=True)
        if self.final_directory.exists():
            raise FileExistsError(f"completed recording already exists: {self.recording_id}")
        os.replace(self.partial_directory, self.final_directory)
        self._closed = True
        return CaptureRecordingReader.open(self.final_directory)

    def close_incomplete(self) -> None:
        if self._closed:
            return
        self._close_csv_files()
        manifest = self._draft_manifest(CaptureRecordingState.INCOMPLETE)
        _atomic_write_json(self._manifest_path, manifest.model_dump(mode="json"))
        self._closed = True

    def _normalize_imu_video_span(self) -> None:
        first_frame_time = self._frames[0].recording_time_ns
        last_frame_time = self._frames[-1].recording_time_ns
        retained = [row for row in self._imu_rows if row.recording_time_ns <= last_frame_time]
        normalized = [
            row.model_copy(
                update={
                    "sample_index": index,
                    "inside_video_span": first_frame_time <= row.recording_time_ns,
                }
            )
            for index, row in enumerate(retained)
        ]
        temp_path = self.imu_path.with_suffix(".csv.tmp")
        with temp_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=IMU_CSV_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(_imu_to_csv(row) for row in normalized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, self.imu_path)
        self._imu_rows = normalized

    def _draft_manifest(self, state: CaptureRecordingState) -> CaptureRecordingManifest:
        return CaptureRecordingManifest(
            recording_id=self.recording_id,
            state=state,
            started_at_unix_ns=self._time_origin.countdown_started_at_unix_ns,
            video_profile=self._video_profile,
            frame_count=len(self._frames),
            imu_sample_count=len(self._imu_rows),
            time_origin=self._time_origin,
            provenance=self._provenance,
        )

    def _write_draft_files(self) -> None:
        manifest = self._draft_manifest(CaptureRecordingState.RECORDING)
        quality = _build_quality_report(
            recording_id=self.recording_id,
            frames=(),
            imu_rows=(),
            telemetry_queue_overflow_count=0,
            timestamp_mapping_residual_ns=None,
            incomplete=True,
        )
        _atomic_write_json(self._manifest_path, manifest.model_dump(mode="json"))
        _atomic_write_json(self._quality_path, quality.model_dump(mode="json"))

    def _open_csv_files(self, *, mode: str) -> None:
        self._frames_file = self.frames_path.open(mode, encoding="utf-8", newline="")
        self._imu_file = self.imu_path.open(mode, encoding="utf-8", newline="")
        self._frames_writer = csv.DictWriter(
            self._frames_file,
            fieldnames=FRAME_CSV_COLUMNS,
            lineterminator="\n",
        )
        self._imu_writer = csv.DictWriter(
            self._imu_file,
            fieldnames=IMU_CSV_COLUMNS,
            lineterminator="\n",
        )
        if mode == "w":
            self._frames_writer.writeheader()
            self._imu_writer.writeheader()
            self._frames_file.flush()
            self._imu_file.flush()

    def _close_csv_files(self) -> None:
        for stream in (self._frames_file, self._imu_file):
            if stream is not None:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
        self._frames_file = None
        self._imu_file = None
        self._frames_writer = None
        self._imu_writer = None

    def _require_open(self) -> None:
        if self._closed:
            raise CaptureRecordingError("recording writer is closed")


class CaptureRecordingReader:
    """Strictly validate and expose one finalized recording."""

    def __init__(
        self,
        directory: Path,
        manifest: CaptureRecordingManifest,
        quality: CaptureRecordingQualityReport,
        frames: Sequence[RecordingFrameRow],
        imu_rows: Sequence[RecordingImuRow],
        *,
        hashes_verified: bool,
    ) -> None:
        self.directory = directory
        self.manifest = manifest
        self.quality = quality
        self._frames = tuple(frames)
        self._imu_rows = tuple(imu_rows)
        self.hashes_verified = hashes_verified

    @classmethod
    def open(
        cls,
        recording_directory: Path,
        *,
        verify_hashes: bool = True,
    ) -> CaptureRecordingReader:
        return _validate_recording(Path(recording_directory), verify_hashes=verify_hashes)

    def iter_frames(self) -> Iterator[RecordingFrameRow]:
        return iter(self._frames)

    def iter_imu_samples(self) -> Iterator[RecordingImuRow]:
        return iter(self._imu_rows)

    @property
    def video_path(self) -> Path:
        return self.directory / self.manifest.storage.video_path

    def summary(self) -> RecordingSummary:
        assert self.manifest.ended_at_unix_ns is not None
        assert self.manifest.duration_ns is not None
        assert self.manifest.artifacts.video.size_bytes is not None
        return RecordingSummary(
            recording_id=self.manifest.recording_id,
            recorded_at_unix_ns=self.manifest.started_at_unix_ns,
            ended_at_unix_ns=self.manifest.ended_at_unix_ns,
            duration_ns=self.manifest.duration_ns,
            width=self.manifest.video_profile.width,
            height=self.manifest.video_profile.height,
            fps=self.manifest.video_profile.fps,
            file_size_bytes=self.manifest.artifacts.video.size_bytes,
            frame_count=self.manifest.frame_count,
            imu_sample_count=self.manifest.imu_sample_count,
            hashes_verified=self.hashes_verified,
        )


def discover_recordings(
    recordings_root: Path,
    *,
    verify_hashes: bool = False,
) -> tuple[CaptureRecordingReader, ...]:
    root = Path(recordings_root)
    if not root.exists():
        return ()
    readers: list[CaptureRecordingReader] = []
    for candidate in sorted(root.iterdir(), reverse=True):
        if not candidate.is_dir() or candidate.name.startswith(".recording-"):
            continue
        try:
            readers.append(CaptureRecordingReader.open(candidate, verify_hashes=verify_hashes))
        except CaptureRecordingReadError:
            continue
    return tuple(readers)


def _validate_recording(
    directory: Path,
    *,
    verify_hashes: bool,
    allow_partial_name: bool = False,
) -> CaptureRecordingReader:
    _validate_layout(directory)
    manifest = _read_manifest(directory / "manifest.json")
    if manifest.state is not CaptureRecordingState.COMPLETE:
        raise CaptureRecordingReadError("recording manifest is not complete")
    expected_name = (
        f".recording-{manifest.recording_id}.partial"
        if allow_partial_name
        else manifest.recording_id
    )
    if directory.name != expected_name:
        raise CaptureRecordingReadError("recording directory name does not match manifest")
    quality = _read_quality(directory / "quality.json")
    if quality.recording_id != manifest.recording_id:
        raise CaptureRecordingReadError("quality report recording ID does not match manifest")
    frames = _read_frame_csv(directory / "frames.csv")
    imu_rows = _read_imu_csv(directory / "imu.csv")
    if len(frames) != manifest.frame_count:
        raise CaptureRecordingReadError("frames.csv row count does not match manifest")
    if len(imu_rows) != manifest.imu_sample_count:
        raise CaptureRecordingReadError("imu.csv row count does not match manifest")
    if quality.counts.video_frame_count != len(frames):
        raise CaptureRecordingReadError("quality video frame count does not match frames.csv")
    if quality.counts.imu_sample_count != len(imu_rows):
        raise CaptureRecordingReadError("quality IMU count does not match imu.csv")
    if manifest.time_origin.first_video_frame_recording_time_ns != frames[0].recording_time_ns:
        raise CaptureRecordingReadError("manifest first video frame time does not match frames.csv")
    if manifest.time_origin.last_video_frame_recording_time_ns != frames[-1].recording_time_ns:
        raise CaptureRecordingReadError("manifest last video frame time does not match frames.csv")
    first_time = frames[0].recording_time_ns
    last_time = frames[-1].recording_time_ns
    for row in imu_rows:
        expected_inside = first_time <= row.recording_time_ns <= last_time
        if row.inside_video_span != expected_inside:
            raise CaptureRecordingReadError(
                "IMU inside_video_span does not match video frame range"
            )
        if row.recording_time_ns > last_time:
            raise CaptureRecordingReadError("imu.csv extends beyond the last video frame")
    if verify_hashes:
        _verify_artifacts(directory, manifest)
    return CaptureRecordingReader(
        directory,
        manifest,
        quality,
        frames,
        imu_rows,
        hashes_verified=verify_hashes,
    )


def _validate_layout(directory: Path) -> None:
    if not directory.is_dir():
        raise CaptureRecordingReadError(f"recording directory does not exist: {directory}")
    entries = {entry.name for entry in directory.iterdir()}
    if entries != _REQUIRED_TOP_LEVEL_ENTRIES:
        missing = sorted(_REQUIRED_TOP_LEVEL_ENTRIES - entries)
        unexpected = sorted(entries - _REQUIRED_TOP_LEVEL_ENTRIES)
        raise CaptureRecordingReadError(
            f"recording layout mismatch; missing={missing}, unexpected={unexpected}"
        )
    for name in ("manifest.json", "video.mp4", "imu.csv", "frames.csv", "quality.json"):
        if not (directory / name).is_file():
            raise CaptureRecordingReadError(f"recording entry must be a file: {name}")
    for name in ("annotations", "derived"):
        if not (directory / name).is_dir():
            raise CaptureRecordingReadError(f"recording entry must be a directory: {name}")


def _read_manifest(path: Path) -> CaptureRecordingManifest:
    try:
        return CaptureRecordingManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError, ValueError) as error:
        raise CaptureRecordingReadError(f"invalid manifest.json: {error}") from error


def _read_quality(path: Path) -> CaptureRecordingQualityReport:
    try:
        return CaptureRecordingQualityReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError, ValueError) as error:
        raise CaptureRecordingReadError(f"invalid quality.json: {error}") from error


def _read_frame_csv(path: Path) -> tuple[RecordingFrameRow, ...]:
    rows = _read_csv(path, FRAME_CSV_COLUMNS, _parse_frame_row)
    previous: RecordingFrameRow | None = None
    for index, row in enumerate(rows):
        if row.frame_index != index:
            raise CaptureRecordingReadError(
                "frames.csv contains duplicate or non-contiguous frames"
            )
        if previous is not None:
            if row.recording_time_ns <= previous.recording_time_ns:
                raise CaptureRecordingReadError(
                    "frames.csv recording time is not strictly monotonic"
                )
            if not _mp4_time_is_after(row, previous):
                raise CaptureRecordingReadError("frames.csv MP4 time is not strictly monotonic")
        previous = row
    return rows


def _read_imu_csv(path: Path) -> tuple[RecordingImuRow, ...]:
    rows = _read_csv(path, IMU_CSV_COLUMNS, _parse_imu_row)
    previous: RecordingImuRow | None = None
    for index, row in enumerate(rows):
        if row.sample_index != index:
            raise CaptureRecordingReadError("imu.csv contains duplicate or non-contiguous samples")
        if previous is not None and row.recording_time_ns < previous.recording_time_ns:
            raise CaptureRecordingReadError("imu.csv recording time is not monotonic")
        previous = row
    return rows


def _read_csv(path: Path, columns: tuple[str, ...], parser):
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != columns:
                raise CaptureRecordingReadError(
                    f"{path.name} columns do not match the required schema"
                )
            parsed = []
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise CaptureRecordingReadError(f"{path.name}:{line_number} has extra columns")
                try:
                    parsed.append(parser(row))
                except (KeyError, TypeError, ValueError, ValidationError) as error:
                    raise CaptureRecordingReadError(
                        f"{path.name}:{line_number} contains invalid data: {error}"
                    ) from error
            return tuple(parsed)
    except CaptureRecordingReadError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise CaptureRecordingReadError(f"cannot read {path.name}: {error}") from error


def _parse_frame_row(row: dict[str | None, str | None]) -> RecordingFrameRow:
    return RecordingFrameRow(
        frame_index=_int_value(row, "frame_index"),
        recording_time_ns=_int_value(row, "recording_time_ns"),
        mp4_pts=_int_value(row, "mp4_pts"),
        mp4_time_base_num=_int_value(row, "mp4_time_base_num"),
        mp4_time_base_den=_int_value(row, "mp4_time_base_den"),
        connection_session_id=_text_value(row, "connection_session_id"),
        frame_id=_int_value(row, "frame_id", optional=True),
        camera_start_generation=_int_value(row, "camera_start_generation", optional=True),
        captured_at_rokid_sdk_ms=_int_value(row, "captured_at_rokid_sdk_ms", optional=True),
        received_at_elapsed_realtime_ns=_int_value(
            row, "received_at_elapsed_realtime_ns", optional=True
        ),
        video_at_monotonic_ns=_int_value(row, "video_at_monotonic_ns", optional=True),
        rtp_timestamp_90khz=_int_value(row, "rtp_timestamp_90khz", optional=True),
        received_at_client_monotonic_ns=_int_value(row, "received_at_client_monotonic_ns"),
        metadata_match_status=FrameMetadataMatchStatus(_text_value(row, "metadata_match_status")),
        timestamp_match_error_90khz=_int_value(row, "timestamp_match_error_90khz", optional=True),
    )


def _parse_imu_row(row: dict[str | None, str | None]) -> RecordingImuRow:
    return RecordingImuRow(
        sample_index=_int_value(row, "sample_index"),
        recording_time_ns=_int_value(row, "recording_time_ns"),
        connection_session_id=_text_value(row, "connection_session_id"),
        sensor_type=ImuSensorType(_text_value(row, "sensor_type")),
        sequence_number=_int_value(row, "sequence_number"),
        sensor_event_monotonic_ns=_int_value(row, "sensor_event_monotonic_ns"),
        received_at_elapsed_realtime_ns=_int_value(row, "received_at_elapsed_realtime_ns"),
        received_at_client_monotonic_ns=_int_value(row, "received_at_client_monotonic_ns"),
        accuracy=_int_value(row, "accuracy"),
        x=_float_value(row, "x"),
        y=_float_value(row, "y"),
        z=_float_value(row, "z"),
        inside_video_span=_bool_value(row, "inside_video_span"),
    )


def _int_value(
    row: dict[str | None, str | None],
    field: str,
    *,
    optional: bool = False,
) -> int | None:
    value = row[field]
    if optional and value == "":
        return None
    if value is None or not _INTEGER_PATTERN.fullmatch(value):
        raise ValueError(f"{field} is not a canonical integer")
    return int(value)


def _float_value(row: dict[str | None, str | None], field: str) -> float:
    value = row[field]
    if value is None or value == "" or value != value.strip():
        raise ValueError(f"{field} is not a float")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _text_value(row: dict[str | None, str | None], field: str) -> str:
    value = row[field]
    if value is None or not value or value != value.strip():
        raise ValueError(f"{field} must contain trimmed text")
    return value


def _bool_value(row: dict[str | None, str | None], field: str) -> bool:
    value = row[field]
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{field} must be true or false")


def _frame_to_csv(row: RecordingFrameRow) -> dict[str, str | int]:
    values = row.model_dump(mode="json")
    return {key: "" if value is None else value for key, value in values.items()}


def _imu_to_csv(row: RecordingImuRow) -> dict[str, str | int | float]:
    values = row.model_dump(mode="json")
    values["inside_video_span"] = "true" if row.inside_video_span else "false"
    return values


def _mp4_time_is_after(current: RecordingFrameRow, previous: RecordingFrameRow) -> bool:
    current_numerator = current.mp4_pts * current.mp4_time_base_num
    previous_numerator = previous.mp4_pts * previous.mp4_time_base_num
    return (
        current_numerator * previous.mp4_time_base_den
        > previous_numerator * current.mp4_time_base_den
    )


def _build_quality_report(
    *,
    recording_id: str,
    frames: Sequence[RecordingFrameRow],
    imu_rows: Sequence[RecordingImuRow],
    telemetry_queue_overflow_count: int,
    timestamp_mapping_residual_ns: int | None,
    incomplete: bool = False,
) -> CaptureRecordingQualityReport:
    gaps, duplicates, out_of_order = _imu_sequence_quality(imu_rows)
    matched = sum(
        row.metadata_match_status is not FrameMetadataMatchStatus.UNMATCHED for row in frames
    )
    coverage = None if not frames else matched / len(frames)
    has_warning = any((gaps, duplicates, out_of_order, telemetry_queue_overflow_count)) or (
        coverage is not None and coverage < 1.0
    )
    checks = [
        RecordingQualityCheck(
            check_id="frame_metadata_coverage",
            status=(
                "not_evaluated" if coverage is None else ("pass" if coverage == 1.0 else "warn")
            ),
            metric_value=coverage,
            threshold=1.0,
            unit="ratio",
            evidence=f"{matched}/{len(frames)} encoded frames have matched Rokid metadata",
        ),
        RecordingQualityCheck(
            check_id="imu_sequence_integrity",
            status="pass" if gaps + duplicates + out_of_order == 0 else "warn",
            metric_value=float(gaps + duplicates + out_of_order),
            threshold=0.0,
            unit="samples",
            evidence=(f"gaps={gaps}, duplicates={duplicates}, out_of_order={out_of_order}"),
        ),
    ]
    counts = RecordingQualityCounts(
        video_frame_count=len(frames),
        matched_video_frame_count=matched,
        imu_sample_count=len(imu_rows),
        imu_inside_video_span_count=sum(row.inside_video_span for row in imu_rows),
        accelerometer_sample_count=sum(
            row.sensor_type is ImuSensorType.ACCELEROMETER for row in imu_rows
        ),
        gyroscope_sample_count=sum(row.sensor_type is ImuSensorType.GYROSCOPE for row in imu_rows),
        imu_sequence_gap_count=gaps,
        imu_duplicate_sample_count=duplicates,
        imu_out_of_order_sample_count=out_of_order,
        telemetry_queue_overflow_count=telemetry_queue_overflow_count,
    )
    return CaptureRecordingQualityReport(
        recording_id=recording_id,
        generated_at_unix_ns=time.time_ns(),
        status="incomplete" if incomplete else ("warn" if has_warning else "pass"),
        counts=counts,
        frame_metadata_coverage=coverage,
        timestamp_mapping_residual_ns=timestamp_mapping_residual_ns,
        checks=checks,
    )


def _imu_sequence_quality(rows: Sequence[RecordingImuRow]) -> tuple[int, int, int]:
    last_sequence: dict[tuple[str, ImuSensorType], int] = {}
    seen: dict[tuple[str, ImuSensorType], set[int]] = {}
    gaps = duplicates = out_of_order = 0
    for row in rows:
        key = (row.connection_session_id, row.sensor_type)
        prior = last_sequence.get(key)
        key_seen = seen.setdefault(key, set())
        if row.sequence_number in key_seen:
            duplicates += 1
        elif prior is not None and row.sequence_number < prior:
            out_of_order += 1
        elif prior is not None and row.sequence_number > prior + 1:
            gaps += row.sequence_number - prior - 1
        key_seen.add(row.sequence_number)
        last_sequence[key] = max(row.sequence_number, prior if prior is not None else -1)
    return gaps, duplicates, out_of_order


def _file_info(path: Path) -> RecordingFileInfo:
    return RecordingFileInfo(
        path=path.name,
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
    )


def _verify_artifacts(directory: Path, manifest: CaptureRecordingManifest) -> None:
    for artifact in (
        manifest.artifacts.video,
        manifest.artifacts.imu,
        manifest.artifacts.frames,
        manifest.artifacts.quality,
    ):
        path = directory / artifact.path
        if artifact.size_bytes != path.stat().st_size:
            raise CaptureRecordingReadError(f"artifact size mismatch: {artifact.path}")
        if artifact.sha256 != _sha256(path):
            raise CaptureRecordingReadError(f"artifact SHA256 mismatch: {artifact.path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: object) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=True, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)
