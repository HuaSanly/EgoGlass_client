from __future__ import annotations

import csv
from pathlib import Path

import pytest

from schemas.recording import (
    FrameMetadataMatchStatus,
    ImuSensorType,
    RecordingFrameRow,
    RecordingImuRow,
    RecordingOutput,
)
from ui.gateway.capture_recording import (
    FRAME_CSV_COLUMNS,
    CaptureRecordingReader,
    CaptureRecordingReadError,
    CaptureRecordingWriter,
    discover_recordings,
)

RECORDING_ID = "a" * 32


def _frame(index: int, recording_time_ns: int) -> RecordingFrameRow:
    return RecordingFrameRow(
        frame_index=index,
        recording_time_ns=recording_time_ns,
        mp4_pts=index * 3_000,
        mp4_time_base_num=1,
        mp4_time_base_den=90_000,
        connection_session_id="connection-1",
        frame_id=index,
        camera_start_generation=1,
        captured_at_rokid_sdk_ms=10_000 + index * 33,
        received_at_elapsed_realtime_ns=20_000_000 + index * 33_000_000,
        video_at_monotonic_ns=30_000_000 + index * 33_000_000,
        rtp_timestamp_90khz=12_000 + index * 3_000,
        received_at_client_monotonic_ns=40_000_000 + index * 33_000_000,
        metadata_match_status=FrameMetadataMatchStatus.EXACT,
        timestamp_match_error_90khz=0,
    )


def _imu(
    index: int,
    recording_time_ns: int,
    *,
    sensor_type: ImuSensorType = ImuSensorType.ACCELEROMETER,
    sequence_number: int | None = None,
) -> RecordingImuRow:
    return RecordingImuRow(
        sample_index=index,
        recording_time_ns=recording_time_ns,
        connection_session_id="connection-1",
        sensor_type=sensor_type,
        sequence_number=index if sequence_number is None else sequence_number,
        sensor_event_monotonic_ns=1_000_000 + recording_time_ns,
        received_at_elapsed_realtime_ns=2_000_000 + recording_time_ns,
        received_at_client_monotonic_ns=3_000_000 + recording_time_ns,
        accuracy=3,
        x=1.25,
        y=-2.5,
        z=9.75,
        inside_video_span=False,
    )


def _complete_recording(
    root: Path,
    recording_id: str = RECORDING_ID,
) -> CaptureRecordingReader:
    writer = CaptureRecordingWriter.create(
        root,
        recording_id=recording_id,
        video_profile=RecordingOutput(width=640, height=480, fps=30.0),
        countdown_started_at_unix_ns=1_000_000,
        countdown_started_at_client_monotonic_ns=5_000,
    )
    writer.video_path.write_bytes(b"synthetic-mp4")
    writer.append_imu(_imu(0, 100))
    writer.append_frame(_frame(0, 1_000))
    writer.append_imu(_imu(1, 1_500, sensor_type=ImuSensorType.GYROSCOPE))
    writer.append_frame(_frame(1, 2_000))
    writer.append_imu(_imu(2, 2_500))
    return writer.finalize(ended_at_unix_ns=2_000_000)


def _rewrite_frame_csv(path: Path, mutator) -> None:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    fieldnames = list(FRAME_CSV_COLUMNS)
    fieldnames, rows = mutator(fieldnames, rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_writer_publishes_exact_layout_and_preserves_countdown_imu(tmp_path: Path) -> None:
    reader = _complete_recording(tmp_path)

    assert not (tmp_path / f".recording-{RECORDING_ID}.partial").exists()
    assert {entry.name for entry in reader.directory.iterdir()} == {
        "manifest.json",
        "video.mp4",
        "imu.csv",
        "frames.csv",
        "quality.json",
        "annotations",
        "derived",
    }
    imu_rows = list(reader.iter_imu_samples())
    assert [row.recording_time_ns for row in imu_rows] == [100, 1_500]
    assert [row.inside_video_span for row in imu_rows] == [False, True]
    assert reader.manifest.frame_count == 2
    assert reader.manifest.imu_sample_count == 2
    assert all(
        artifact.sha256
        for artifact in (
            reader.manifest.artifacts.video,
            reader.manifest.artifacts.imu,
            reader.manifest.artifacts.frames,
            reader.manifest.artifacts.quality,
        )
    )


def test_consecutive_recordings_are_independent(tmp_path: Path) -> None:
    first = _complete_recording(tmp_path, "a" * 32)
    second = _complete_recording(tmp_path, "b" * 32)

    assert first.directory != second.directory
    assert [reader.manifest.recording_id for reader in discover_recordings(tmp_path)] == [
        "b" * 32,
        "a" * 32,
    ]


def test_incomplete_writer_can_be_recovered_and_atomically_completed(tmp_path: Path) -> None:
    writer = CaptureRecordingWriter.create(
        tmp_path,
        recording_id=RECORDING_ID,
        countdown_started_at_unix_ns=100,
        countdown_started_at_client_monotonic_ns=200,
    )
    writer.video_path.write_bytes(b"recoverable-video")
    writer.append_imu(_imu(0, 100))
    writer.append_frame(_frame(0, 1_000))
    writer.close_incomplete()

    recovered = CaptureRecordingWriter.recover(tmp_path, RECORDING_ID)
    recovered.append_frame(_frame(1, 2_000))
    reader = recovered.finalize(ended_at_unix_ns=3_000)

    assert reader.directory == tmp_path / RECORDING_ID
    assert not (tmp_path / f".recording-{RECORDING_ID}.partial").exists()
    assert [row.frame_index for row in reader.iter_frames()] == [0, 1]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda fields, rows: ([field for field in fields if field != "mp4_pts"], rows),
            "columns",
        ),
        (
            lambda fields, rows: (fields, [{**rows[0], "mp4_pts": "not-an-int"}, rows[1]]),
            "invalid data",
        ),
        (
            lambda fields, rows: (fields, [rows[0], {**rows[1], "frame_index": "0"}]),
            "duplicate or non-contiguous",
        ),
        (
            lambda fields, rows: (
                fields,
                [rows[0], {**rows[1], "recording_time_ns": rows[0]["recording_time_ns"]}],
            ),
            "not strictly monotonic",
        ),
    ],
)
def test_reader_rejects_invalid_frame_csv(tmp_path: Path, mutator, message: str) -> None:
    reader = _complete_recording(tmp_path)
    _rewrite_frame_csv(reader.directory / "frames.csv", mutator)

    with pytest.raises(CaptureRecordingReadError, match=message):
        CaptureRecordingReader.open(reader.directory, verify_hashes=False)


def test_reader_rejects_artifact_hash_failure(tmp_path: Path) -> None:
    reader = _complete_recording(tmp_path)
    with reader.video_path.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(CaptureRecordingReadError, match="artifact size mismatch"):
        CaptureRecordingReader.open(reader.directory)


def test_reader_rejects_bad_imu_types_even_without_hash_check(tmp_path: Path) -> None:
    reader = _complete_recording(tmp_path)
    path = reader.directory / "imu.csv"
    text = path.read_text(encoding="utf-8").replace(",1.25,", ",nan,")
    path.write_text(text, encoding="utf-8", newline="")

    with pytest.raises(CaptureRecordingReadError, match="must be finite"):
        CaptureRecordingReader.open(reader.directory, verify_hashes=False)


def test_quality_counts_sequence_faults_without_rejecting_raw_samples(tmp_path: Path) -> None:
    writer = CaptureRecordingWriter.create(
        tmp_path,
        recording_id=RECORDING_ID,
        countdown_started_at_unix_ns=100,
        countdown_started_at_client_monotonic_ns=200,
    )
    writer.video_path.write_bytes(b"quality-video")
    writer.append_frame(_frame(0, 1_000))
    writer.append_imu(_imu(0, 1_100, sequence_number=4))
    writer.append_imu(_imu(1, 1_200, sequence_number=6))
    writer.append_imu(_imu(2, 1_300, sequence_number=6))
    writer.append_imu(_imu(3, 1_400, sequence_number=3))
    writer.append_frame(_frame(1, 2_000))

    reader = writer.finalize(ended_at_unix_ns=3_000, telemetry_queue_overflow_count=2)

    assert reader.quality.status == "warn"
    assert reader.quality.counts.imu_sequence_gap_count == 1
    assert reader.quality.counts.imu_duplicate_sample_count == 1
    assert reader.quality.counts.imu_out_of_order_sample_count == 1
    assert reader.quality.counts.telemetry_queue_overflow_count == 2
