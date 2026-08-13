from __future__ import annotations

import csv
import math
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO
from uuid import uuid4

import av
import yaml
from pydantic import ValidationError

from schemas.recording import (
    CalibrationSnapshot,
    CameraFrameRow,
    ImuSensorType,
    RecordingImuRow,
    RecordingOutput,
    RecordingSummary,
)

CAMERA_CSV_COLUMNS = (
    "frame_idx",
    "frame_id",
    "rokid_timestamp_ns",
    "device_monotonic_ns",
)
IMU_CSV_COLUMNS = (
    "sensor_type",
    "sequence",
    "timestamp_ns",
    "x",
    "y",
    "z",
)
_STAGED_IMU_COLUMNS = (*IMU_CSV_COLUMNS, "received_at_client_monotonic_ns")
_REQUIRED_ENTRIES = {"video.mp4", "camera.csv", "imu.csv", "calibration.yaml"}
_RECORDING_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_INTEGER_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)\Z")


class CaptureRecordingError(RuntimeError):
    """Base error for minimal recording protocol failures."""


class CaptureRecordingReadError(CaptureRecordingError):
    """Raised when a published recording violates the protocol."""


@dataclass(frozen=True, slots=True)
class StagedImuSample:
    row: RecordingImuRow
    received_at_client_monotonic_ns: int

    def __post_init__(self) -> None:
        if self.received_at_client_monotonic_ns < 0:
            raise ValueError("client receive timestamp cannot be negative")


@dataclass(frozen=True, slots=True)
class StagedCameraFrame:
    row: CameraFrameRow
    mp4_pts: int
    mp4_time_base_num: int
    mp4_time_base_den: int
    received_at_client_monotonic_ns: int

    def __post_init__(self) -> None:
        if self.mp4_pts < 0:
            raise ValueError("MP4 PTS cannot be negative")
        if self.mp4_time_base_num <= 0 or self.mp4_time_base_den <= 0:
            raise ValueError("MP4 time base must be positive")
        if self.received_at_client_monotonic_ns < 0:
            raise ValueError("client receive timestamp cannot be negative")


@dataclass(frozen=True, slots=True)
class VideoIndexEntry:
    frame_idx: int
    pts: int
    time_base_num: int
    time_base_den: int


class CaptureRecordingWriter:
    """Stage raw capture data and atomically publish the strict four-file unit."""

    def __init__(
        self,
        recordings_root: Path,
        recording_id: str,
        *,
        video_profile: RecordingOutput,
    ) -> None:
        if not _RECORDING_ID_PATTERN.fullmatch(recording_id):
            raise ValueError("recording_id must be 32 lowercase hexadecimal characters")
        self.recordings_root = Path(recordings_root).resolve()
        self.recording_id = recording_id
        self.partial_directory = self.recordings_root / f".recording-{recording_id}.partial"
        self.final_directory = self.recordings_root / recording_id
        self.video_path = self.partial_directory / "video.mp4"
        self.camera_path = self.partial_directory / "camera.csv"
        self.imu_path = self.partial_directory / "imu.csv"
        self.calibration_path = self.partial_directory / "calibration.yaml"
        self._staged_imu_path = self.partial_directory / ".imu-staging.csv"
        self._video_profile = video_profile
        self._calibration = CalibrationSnapshot.placeholder(
            video_profile.width,
            video_profile.height,
        )
        self._staged_imu_file: TextIO | None = None
        self._staged_imu_writer: csv.DictWriter[str] | None = None
        self._last_imu_sequence: dict[ImuSensorType, int] = {}
        self._last_imu_timestamp: dict[ImuSensorType, int] = {}
        self._closed = False

        self.recordings_root.mkdir(parents=True, exist_ok=True)
        if self.partial_directory.exists() or self.final_directory.exists():
            raise FileExistsError(f"recording already exists: {recording_id}")
        self.partial_directory.mkdir()
        self.video_path.touch(exist_ok=False)
        _atomic_write_yaml(self.calibration_path, self._calibration.model_dump(mode="python"))
        self._open_staging_file()

    @classmethod
    def create(
        cls,
        recordings_root: Path,
        *,
        recording_id: str | None = None,
        video_profile: RecordingOutput | None = None,
    ) -> CaptureRecordingWriter:
        return cls(
            recordings_root,
            recording_id or uuid4().hex,
            video_profile=video_profile or RecordingOutput(),
        )

    def append_imu(self, sample: StagedImuSample) -> None:
        self._require_open()
        row = sample.row
        previous_sequence = self._last_imu_sequence.get(row.sensor_type)
        previous_timestamp = self._last_imu_timestamp.get(row.sensor_type)
        if previous_sequence is not None and row.sequence <= previous_sequence:
            raise ValueError(f"{row.sensor_type} sequence must be strictly increasing")
        if previous_timestamp is not None and row.timestamp_ns <= previous_timestamp:
            raise ValueError(f"{row.sensor_type} timestamp must be strictly increasing")
        assert self._staged_imu_writer is not None and self._staged_imu_file is not None
        values = row.model_dump(mode="json")
        values["received_at_client_monotonic_ns"] = sample.received_at_client_monotonic_ns
        self._staged_imu_writer.writerow(values)
        self._staged_imu_file.flush()
        self._last_imu_sequence[row.sensor_type] = row.sequence
        self._last_imu_timestamp[row.sensor_type] = row.timestamp_ns

    def finalize(self, camera_frames: Sequence[StagedCameraFrame]) -> CaptureRecordingReader:
        self._require_open()
        self._close_staging_file()
        if not camera_frames:
            raise CaptureRecordingError("cannot publish a recording without camera frames")
        if self.video_path.stat().st_size < 1:
            raise CaptureRecordingError("cannot publish an empty video.mp4")

        staged_imu = _read_staged_imu(self._staged_imu_path)
        final_imu = tuple(
            sample.row
            for sample in staged_imu
            if sample.received_at_client_monotonic_ns
            <= camera_frames[-1].received_at_client_monotonic_ns
        )
        _atomic_write_csv(
            self.camera_path,
            CAMERA_CSV_COLUMNS,
            (frame.row.model_dump(mode="json") for frame in camera_frames),
        )
        _atomic_write_csv(
            self.imu_path,
            IMU_CSV_COLUMNS,
            (row.model_dump(mode="json") for row in final_imu),
        )
        reader = _validate_recording(
            self.partial_directory,
            allow_partial_name=True,
            allow_staging_file=True,
        )
        self._staged_imu_path.unlink()
        reader = _validate_recording(
            self.partial_directory,
            allow_partial_name=True,
        )
        if self.final_directory.exists():
            raise FileExistsError(f"completed recording already exists: {self.recording_id}")
        os.replace(self.partial_directory, self.final_directory)
        self._closed = True
        return reader._with_directory(self.final_directory)

    def close_incomplete(self) -> None:
        if self._closed:
            return
        self._close_staging_file()
        self._closed = True

    def _open_staging_file(self) -> None:
        self._staged_imu_file = self._staged_imu_path.open(
            "w",
            encoding="utf-8",
            newline="",
        )
        self._staged_imu_writer = csv.DictWriter(
            self._staged_imu_file,
            fieldnames=_STAGED_IMU_COLUMNS,
            lineterminator="\n",
        )
        self._staged_imu_writer.writeheader()
        self._staged_imu_file.flush()

    def _close_staging_file(self) -> None:
        if self._staged_imu_file is not None:
            self._staged_imu_file.flush()
            os.fsync(self._staged_imu_file.fileno())
            self._staged_imu_file.close()
        self._staged_imu_file = None
        self._staged_imu_writer = None

    def _require_open(self) -> None:
        if self._closed:
            raise CaptureRecordingError("recording writer is closed")


class CaptureRecordingReader:
    """Strictly validate and expose one minimal protocol recording."""

    def __init__(
        self,
        directory: Path,
        camera_frames: Sequence[CameraFrameRow],
        imu_rows: Sequence[RecordingImuRow],
        calibration: CalibrationSnapshot,
        video_index: Sequence[VideoIndexEntry],
        *,
        fps: float,
    ) -> None:
        self.directory = directory
        self.recording_id = _recording_id_from_directory(directory)
        self._camera_frames = tuple(camera_frames)
        self._imu_rows = tuple(imu_rows)
        self.calibration = calibration
        self.video_index = tuple(video_index)
        self.fps = fps

    @classmethod
    def open(cls, recording_directory: Path) -> CaptureRecordingReader:
        return _validate_recording(Path(recording_directory))

    @property
    def video_path(self) -> Path:
        return self.directory / "video.mp4"

    def iter_camera_frames(self) -> Iterator[CameraFrameRow]:
        return iter(self._camera_frames)

    def iter_imu_samples(self) -> Iterator[RecordingImuRow]:
        return iter(self._imu_rows)

    def summary(self) -> RecordingSummary:
        stat = self.video_path.stat()
        duration_ns = (
            self._camera_frames[-1].device_monotonic_ns
            - self._camera_frames[0].device_monotonic_ns
        )
        return RecordingSummary(
            recording_id=self.recording_id,
            recorded_at_unix_ns=stat.st_ctime_ns,
            ended_at_unix_ns=stat.st_mtime_ns,
            duration_ns=duration_ns,
            width=self.calibration.camera.resolution[0],
            height=self.calibration.camera.resolution[1],
            fps=self.fps,
            file_size_bytes=stat.st_size,
            frame_count=len(self._camera_frames),
            imu_sample_count=len(self._imu_rows),
            camera_frame_gap_count=_camera_frame_gaps(self._camera_frames),
            imu_sequence_gap_count=_imu_sequence_gaps(self._imu_rows),
        )

    def _with_directory(self, directory: Path) -> CaptureRecordingReader:
        return CaptureRecordingReader(
            directory,
            self._camera_frames,
            self._imu_rows,
            self.calibration,
            self.video_index,
            fps=self.fps,
        )


def discover_recordings(recordings_root: Path) -> tuple[CaptureRecordingReader, ...]:
    root = Path(recordings_root)
    if not root.exists():
        return ()
    readers: list[CaptureRecordingReader] = []
    for candidate in root.iterdir():
        if not candidate.is_dir() or not _RECORDING_ID_PATTERN.fullmatch(candidate.name):
            continue
        try:
            readers.append(CaptureRecordingReader.open(candidate))
        except CaptureRecordingReadError:
            continue
    readers.sort(key=lambda reader: reader.summary().recorded_at_unix_ns, reverse=True)
    return tuple(readers)


def recover_completed_recordings(recordings_root: Path) -> tuple[str, ...]:
    root = Path(recordings_root).resolve()
    recovered: list[str] = []
    for partial in root.glob(".recording-*.partial"):
        recording_id = _recording_id_from_directory(partial)
        final = root / recording_id
        if final.exists():
            continue
        try:
            staging_path = partial / ".imu-staging.csv"
            _validate_recording(
                partial,
                allow_partial_name=True,
                allow_staging_file=staging_path.is_file(),
            )
        except CaptureRecordingReadError:
            continue
        if staging_path.is_file():
            staging_path.unlink()
            _validate_recording(partial, allow_partial_name=True)
        os.replace(partial, final)
        recovered.append(recording_id)
    return tuple(recovered)


def _validate_recording(
    directory: Path,
    *,
    allow_partial_name: bool = False,
    allow_staging_file: bool = False,
) -> CaptureRecordingReader:
    _validate_layout(directory, allow_staging_file=allow_staging_file)
    recording_id = _recording_id_from_directory(directory)
    expected_name = (
        f".recording-{recording_id}.partial" if allow_partial_name else recording_id
    )
    if directory.name != expected_name:
        raise CaptureRecordingReadError("recording directory name does not match recording ID")
    camera_frames = _read_camera_csv(directory / "camera.csv")
    imu_rows = _read_imu_csv(directory / "imu.csv")
    calibration = _read_calibration(directory / "calibration.yaml")
    video_index, fps = _inspect_video(
        directory / "video.mp4",
        calibration.camera.resolution,
    )
    if len(video_index) != len(camera_frames):
        raise CaptureRecordingReadError(
            "decoded video frame count does not match camera.csv"
        )
    if not imu_rows:
        raise CaptureRecordingReadError("imu.csv must contain raw sensor samples")
    first_camera_ns = camera_frames[0].device_monotonic_ns
    last_camera_ns = camera_frames[-1].device_monotonic_ns
    imu_timestamps = [row.timestamp_ns for row in imu_rows]
    if min(imu_timestamps) > first_camera_ns or max(imu_timestamps) < last_camera_ns:
        raise CaptureRecordingReadError("IMU timestamps do not cover the camera time range")
    return CaptureRecordingReader(
        directory,
        camera_frames,
        imu_rows,
        calibration,
        video_index,
        fps=fps,
    )


def _validate_layout(directory: Path, *, allow_staging_file: bool = False) -> None:
    if not directory.is_dir():
        raise CaptureRecordingReadError(f"recording directory does not exist: {directory}")
    entries = {entry.name for entry in directory.iterdir()}
    allowed_entries = (
        _REQUIRED_ENTRIES | {".imu-staging.csv"}
        if allow_staging_file
        else _REQUIRED_ENTRIES
    )
    if entries != allowed_entries:
        raise CaptureRecordingReadError(
            "recording layout mismatch; "
            f"missing={sorted(allowed_entries - entries)}, "
            f"unexpected={sorted(entries - allowed_entries)}"
        )
    if not all((directory / name).is_file() for name in _REQUIRED_ENTRIES):
        raise CaptureRecordingReadError("every recording entry must be a regular file")


def _recording_id_from_directory(directory: Path) -> str:
    name = directory.name
    recording_id = (
        name.removeprefix(".recording-").removesuffix(".partial")
        if name.startswith(".recording-") and name.endswith(".partial")
        else name
    )
    if not _RECORDING_ID_PATTERN.fullmatch(recording_id):
        raise CaptureRecordingReadError("recording directory has an invalid ID")
    return recording_id


def _read_camera_csv(path: Path) -> tuple[CameraFrameRow, ...]:
    rows = _read_csv(
        path,
        CAMERA_CSV_COLUMNS,
        lambda row: CameraFrameRow(
            frame_idx=_int_value(row, "frame_idx"),
            frame_id=_int_value(row, "frame_id"),
            rokid_timestamp_ns=_int_value(row, "rokid_timestamp_ns"),
            device_monotonic_ns=_int_value(row, "device_monotonic_ns"),
        ),
    )
    if not rows:
        raise CaptureRecordingReadError("camera.csv must contain at least one frame")
    previous: CameraFrameRow | None = None
    for index, row in enumerate(rows):
        if row.frame_idx != index:
            raise CaptureRecordingReadError("camera frame_idx must be contiguous from zero")
        if previous is not None and (
            row.frame_id <= previous.frame_id
            or row.rokid_timestamp_ns <= previous.rokid_timestamp_ns
            or row.device_monotonic_ns <= previous.device_monotonic_ns
        ):
            raise CaptureRecordingReadError("camera frame IDs and timestamps must increase")
        previous = row
    return rows


def _read_imu_csv(path: Path) -> tuple[RecordingImuRow, ...]:
    rows = _read_csv(
        path,
        IMU_CSV_COLUMNS,
        lambda row: RecordingImuRow(
            sensor_type=ImuSensorType(_text_value(row, "sensor_type")),
            sequence=_int_value(row, "sequence"),
            timestamp_ns=_int_value(row, "timestamp_ns"),
            x=_float_value(row, "x"),
            y=_float_value(row, "y"),
            z=_float_value(row, "z"),
        ),
    )
    last_sequence: dict[ImuSensorType, int] = {}
    last_timestamp: dict[ImuSensorType, int] = {}
    for row in rows:
        if row.sensor_type in last_sequence and row.sequence <= last_sequence[row.sensor_type]:
            raise CaptureRecordingReadError("IMU sequence must increase per sensor")
        if (
            row.sensor_type in last_timestamp
            and row.timestamp_ns <= last_timestamp[row.sensor_type]
        ):
            raise CaptureRecordingReadError("IMU timestamp must increase per sensor")
        last_sequence[row.sensor_type] = row.sequence
        last_timestamp[row.sensor_type] = row.timestamp_ns
    if set(last_sequence) != {ImuSensorType.ACCELEROMETER, ImuSensorType.GYROSCOPE}:
        raise CaptureRecordingReadError("imu.csv must contain accelerometer and gyroscope data")
    return rows


def _read_staged_imu(path: Path) -> tuple[StagedImuSample, ...]:
    return _read_csv(
        path,
        _STAGED_IMU_COLUMNS,
        lambda row: StagedImuSample(
            row=RecordingImuRow(
                sensor_type=ImuSensorType(_text_value(row, "sensor_type")),
                sequence=_int_value(row, "sequence"),
                timestamp_ns=_int_value(row, "timestamp_ns"),
                x=_float_value(row, "x"),
                y=_float_value(row, "y"),
                z=_float_value(row, "z"),
            ),
            received_at_client_monotonic_ns=_int_value(
                row,
                "received_at_client_monotonic_ns",
            ),
        ),
    )


def _read_calibration(path: Path) -> CalibrationSnapshot:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return CalibrationSnapshot.model_validate(payload)
    except (OSError, UnicodeError, ValueError, ValidationError, yaml.YAMLError) as error:
        raise CaptureRecordingReadError(f"invalid calibration.yaml: {error}") from error


def _inspect_video(
    path: Path,
    resolution: tuple[int, int],
) -> tuple[tuple[VideoIndexEntry, ...], float]:
    try:
        with av.open(str(path), mode="r") as container:
            if len(container.streams.video) != 1:
                raise CaptureRecordingReadError("video.mp4 must contain exactly one video stream")
            stream = container.streams.video[0]
            if stream.codec_context.name != "h264":
                raise CaptureRecordingReadError("video.mp4 must use H.264")
            if (stream.width, stream.height) != resolution:
                raise CaptureRecordingReadError(
                    "video dimensions do not match calibration resolution"
                )
            entries: list[VideoIndexEntry] = []
            for index, frame in enumerate(container.decode(stream)):
                if frame.pts is None or frame.time_base is None:
                    raise CaptureRecordingReadError("decoded video frame is missing PTS")
                if (frame.width, frame.height) != resolution:
                    raise CaptureRecordingReadError("video resolution changes within recording")
                entries.append(
                    VideoIndexEntry(
                        frame_idx=index,
                        pts=frame.pts,
                        time_base_num=frame.time_base.numerator,
                        time_base_den=frame.time_base.denominator,
                    )
                )
            if not entries:
                raise CaptureRecordingReadError("video.mp4 contains no decodable frames")
            rate = stream.average_rate or stream.guessed_rate
            fps = float(rate) if rate is not None and float(rate) > 0 else 30.0
            return tuple(entries), fps
    except CaptureRecordingReadError:
        raise
    except (OSError, ValueError, av.error.FFmpegError) as error:
        raise CaptureRecordingReadError(f"invalid video.mp4: {error}") from error


def _read_csv(path: Path, columns: tuple[str, ...], parser):
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != columns:
                raise CaptureRecordingReadError(
                    f"{path.name} columns do not match the protocol"
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


def _int_value(row: dict[str | None, str | None], field: str) -> int:
    value = row[field]
    if value is None or not _INTEGER_PATTERN.fullmatch(value):
        raise ValueError(f"{field} is not a canonical non-negative integer")
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


def _atomic_write_csv(
    path: Path,
    columns: tuple[str, ...],
    rows,
) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


def _atomic_write_yaml(path: Path, payload: object) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


def _camera_frame_gaps(rows: Sequence[CameraFrameRow]) -> int:
    return sum(
        max(0, current.frame_id - previous.frame_id - 1)
        for previous, current in zip(rows, rows[1:], strict=False)
    )


def _imu_sequence_gaps(rows: Sequence[RecordingImuRow]) -> int:
    last: dict[ImuSensorType, int] = {}
    gaps = 0
    for row in rows:
        previous = last.get(row.sensor_type)
        if previous is not None:
            gaps += max(0, row.sequence - previous - 1)
        last[row.sensor_type] = row.sequence
    return gaps
