from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from schemas.recording import ImuSensorType, RecordingImuRow, RecordingOutput
from tests.recording_support import (
    append_covering_imu,
    create_recording,
    staged_camera_frames,
    write_h264_video,
)
from ui.gateway.capture_recording import (
    CAMERA_CSV_COLUMNS,
    IMU_CSV_COLUMNS,
    CaptureRecordingError,
    CaptureRecordingReader,
    CaptureRecordingReadError,
    CaptureRecordingWriter,
    StagedImuSample,
    discover_recordings,
    recover_completed_recordings,
)

RECORDING_ID = "a" * 32


def _rewrite_csv(path: Path, mutator) -> None:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    fields, rows = mutator(fields, rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_writer_publishes_exact_four_file_protocol(tmp_path: Path) -> None:
    reader = create_recording(tmp_path)

    assert {entry.name for entry in reader.directory.iterdir()} == {
        "video.mp4",
        "camera.csv",
        "imu.csv",
        "calibration.yaml",
    }
    assert not (tmp_path / f".recording-{RECORDING_ID}.partial").exists()
    assert tuple(next(csv.reader((reader.directory / "camera.csv").open()))) == (
        CAMERA_CSV_COLUMNS
    )
    assert tuple(next(csv.reader((reader.directory / "imu.csv").open()))) == IMU_CSV_COLUMNS
    camera = tuple(reader.iter_camera_frames())
    assert camera[0].rokid_timestamp_ns == 5_000_000_000
    assert [row.frame_idx for row in camera] == [0, 1]
    assert reader.summary().protocol_validated is True


def test_placeholder_calibration_is_written_at_video_resolution(tmp_path: Path) -> None:
    reader = create_recording(tmp_path, width=40, height=30)
    payload = yaml.safe_load((reader.directory / "calibration.yaml").read_text())

    assert payload["camera"]["resolution"] == [40, 30]
    assert payload["camera"]["intrinsics"] == [1.0, 1.0, 0.0, 0.0]
    assert payload["camera"]["distortion_coeffs"] == [0.0, 0.0, 0.0, 0.0]
    assert payload["T_cam_imu"] == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    assert set(payload["imu"].values()) == {None}
    assert "unverified" not in payload


def test_consecutive_recordings_are_independent(tmp_path: Path) -> None:
    first = create_recording(tmp_path, recording_id="a" * 32)
    second = create_recording(tmp_path, recording_id="b" * 32)

    assert first.directory != second.directory
    assert {reader.recording_id for reader in discover_recordings(tmp_path)} == {
        "a" * 32,
        "b" * 32,
    }


def test_completed_partial_is_recovered_atomically(tmp_path: Path) -> None:
    reader = create_recording(tmp_path)
    partial = tmp_path / f".recording-{RECORDING_ID}.partial"
    reader.directory.rename(partial)

    assert recover_completed_recordings(tmp_path) == (RECORDING_ID,)
    assert CaptureRecordingReader.open(tmp_path / RECORDING_ID).recording_id == RECORDING_ID


def test_completed_partial_with_debug_staging_is_recovered(tmp_path: Path) -> None:
    reader = create_recording(tmp_path)
    partial = tmp_path / f".recording-{RECORDING_ID}.partial"
    reader.directory.rename(partial)
    (partial / ".imu-staging.csv").write_text(
        "sensor_type,sequence,timestamp_ns,x,y,z,received_at_client_monotonic_ns\n",
        encoding="utf-8",
    )

    assert recover_completed_recordings(tmp_path) == (RECORDING_ID,)
    completed = tmp_path / RECORDING_ID
    assert {path.name for path in completed.iterdir()} == {
        "video.mp4",
        "camera.csv",
        "imu.csv",
        "calibration.yaml",
    }


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda fields, rows: (fields[:-1], rows), "columns"),
        (
            lambda fields, rows: (fields, [{**rows[0], "frame_idx": "1"}, *rows[1:]]),
            "contiguous",
        ),
        (
            lambda fields, rows: (
                fields,
                [rows[0], {**rows[1], "rokid_timestamp_ns": rows[0]["rokid_timestamp_ns"]}],
            ),
            "must increase",
        ),
    ],
)
def test_reader_rejects_bad_camera_csv(tmp_path: Path, mutator, message: str) -> None:
    reader = create_recording(tmp_path)
    _rewrite_csv(reader.directory / "camera.csv", mutator)

    with pytest.raises(CaptureRecordingReadError, match=message):
        CaptureRecordingReader.open(reader.directory)


@pytest.mark.parametrize("sequence", [0, 2])
def test_imu_sequence_gaps_are_allowed(tmp_path: Path, sequence: int) -> None:
    writer = CaptureRecordingWriter.create(
        tmp_path,
        recording_id=RECORDING_ID,
        video_profile=RecordingOutput(width=32, height=24, fps=10),
    )
    video_index = write_h264_video(writer.video_path)
    frames = staged_camera_frames(video_index)
    for sensor in (ImuSensorType.ACCELEROMETER, ImuSensorType.GYROSCOPE):
        writer.append_imu(
            StagedImuSample(
                row=RecordingImuRow(
                    sensor_type=sensor,
                    sequence=sequence,
                    timestamp_ns=frames[0].row.device_monotonic_ns - 1,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                ),
                received_at_client_monotonic_ns=frames[0].received_at_client_monotonic_ns,
            )
        )
        writer.append_imu(
            StagedImuSample(
                row=RecordingImuRow(
                    sensor_type=sensor,
                    sequence=sequence + 2,
                    timestamp_ns=frames[-1].row.device_monotonic_ns + 1,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                ),
                received_at_client_monotonic_ns=frames[-1].received_at_client_monotonic_ns,
            )
        )

    assert writer.finalize(frames).summary().imu_sequence_gap_count == 2


@pytest.mark.parametrize("field", ["sequence", "timestamp_ns"])
def test_reader_rejects_duplicate_or_reversed_imu(tmp_path: Path, field: str) -> None:
    reader = create_recording(tmp_path)

    def duplicate(fields, rows):
        same_sensor = [row for row in rows if row["sensor_type"] == "accelerometer"]
        same_sensor[1][field] = same_sensor[0][field]
        return fields, rows

    _rewrite_csv(reader.directory / "imu.csv", duplicate)
    with pytest.raises(CaptureRecordingReadError, match="IMU .* must increase"):
        CaptureRecordingReader.open(reader.directory)


def test_reader_rejects_missing_imu_coverage(tmp_path: Path) -> None:
    reader = create_recording(tmp_path)

    def move_after_camera(fields, rows):
        for row in rows:
            row["timestamp_ns"] = str(int(row["timestamp_ns"]) + 1_000_000_000)
        return fields, rows

    _rewrite_csv(reader.directory / "imu.csv", move_after_camera)
    with pytest.raises(CaptureRecordingReadError, match="do not cover"):
        CaptureRecordingReader.open(reader.directory)


def test_reader_rejects_corrupt_video_bad_yaml_and_extra_files(tmp_path: Path) -> None:
    first = create_recording(tmp_path, recording_id="a" * 32)
    first.video_path.write_bytes(b"not-an-mp4")
    with pytest.raises(CaptureRecordingReadError, match="video.mp4"):
        CaptureRecordingReader.open(first.directory)

    second = create_recording(tmp_path, recording_id="b" * 32)
    (second.directory / "calibration.yaml").write_text("camera: [", encoding="utf-8")
    with pytest.raises(CaptureRecordingReadError, match="calibration.yaml"):
        CaptureRecordingReader.open(second.directory)

    third = create_recording(tmp_path, recording_id="c" * 32)
    (third.directory / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CaptureRecordingReadError, match="unexpected"):
        CaptureRecordingReader.open(third.directory)


def test_reader_rejects_decoded_frame_count_mismatch(tmp_path: Path) -> None:
    reader = create_recording(tmp_path)

    def drop_last_row(fields, rows):
        return fields, rows[:-1]

    _rewrite_csv(reader.directory / "camera.csv", drop_last_row)
    with pytest.raises(CaptureRecordingReadError, match="frame count does not match"):
        CaptureRecordingReader.open(reader.directory)


def test_reader_rejects_calibration_resolution_mismatch(tmp_path: Path) -> None:
    reader = create_recording(tmp_path)
    path = reader.directory / "calibration.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["camera"]["resolution"] = [64, 48]
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(CaptureRecordingReadError, match="dimensions do not match"):
        CaptureRecordingReader.open(reader.directory)


def test_disk_failure_cannot_publish_final_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = CaptureRecordingWriter.create(
        tmp_path,
        recording_id=RECORDING_ID,
        video_profile=RecordingOutput(width=32, height=24, fps=10),
    )
    video_index = write_h264_video(writer.video_path)
    frames = staged_camera_frames(video_index)
    append_covering_imu(writer, frames)

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("simulated disk failure")

    monkeypatch.setattr("ui.gateway.capture_recording.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated disk failure"):
        writer.finalize(frames)

    assert not (tmp_path / RECORDING_ID).exists()
    assert (tmp_path / f".recording-{RECORDING_ID}.partial").is_dir()


def test_failed_validation_keeps_partial_and_staging_data(tmp_path: Path) -> None:
    writer = CaptureRecordingWriter.create(
        tmp_path,
        recording_id=RECORDING_ID,
        video_profile=RecordingOutput(width=32, height=24, fps=10),
    )
    video_index = write_h264_video(writer.video_path, frame_count=1)
    frames = staged_camera_frames(video_index)
    append_covering_imu(writer, frames)
    writer.video_path.write_bytes(b"corrupt")

    with pytest.raises(CaptureRecordingError):
        writer.finalize(frames)

    partial = tmp_path / f".recording-{RECORDING_ID}.partial"
    assert partial.is_dir()
    assert (partial / ".imu-staging.csv").is_file()
    assert not (tmp_path / RECORDING_ID).exists()
