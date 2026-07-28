from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from ingest_gateway.adapters.mp4_recorder import RecordedVideoFrame
from ingest_gateway.capture_session import CaptureSessionDatabase
from ingest_gateway.recording_models import (
    CaptureSessionClip,
    CaptureSessionLifecycle,
    CaptureSessionManifest,
    CaptureSessionTimeOrigin,
    CaptureVideoProfile,
)
from ingest_gateway.webrtc_matcher import FrameMetadataMatch
from ingest_gateway.webrtc_models import ImuSample, VideoFrameMetadata
from ingest_gateway.webrtc_models import ImuSensorType as GatewayImuSensorType
from perception.sensor_preprocessing import (
    CaptureSessionReader,
    CaptureSessionReadError,
    ImuSensorType,
    MetadataMatchStatus,
)

SESSION_ID = "a" * 32
CLIP_ID = "b" * 32
CONNECTION_ID = "c" * 32


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _mutate_telemetry(session_directory: Path, statement: str) -> None:
    database_path = session_directory / "telemetry" / "telemetry.sqlite"
    with sqlite3.connect(database_path) as database:
        database.execute(statement)
        database.commit()
        database.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def _imu_sample(sensor_type: GatewayImuSensorType, sequence: int, event_ns: int) -> ImuSample:
    return ImuSample(
        sensor_type=sensor_type,
        android_sensor_type=1 if sensor_type is GatewayImuSensorType.ACCELEROMETER else 4,
        sequence_number=sequence,
        sensor_event_monotonic_ns=event_ns,
        received_at_elapsed_realtime_ns=event_ns + 200,
        accuracy=3,
        values=(0.1, 0.2, 9.8),
    )


def _frame_match() -> FrameMetadataMatch:
    metadata = VideoFrameMetadata(
        frame_id=0,
        camera_start_generation=1,
        captured_at_rokid_sdk_ms=1_000,
        received_at_elapsed_realtime_ns=3_000_000_000,
        video_at_monotonic_ns=2_999_900_000,
        rtp_timestamp_90khz=0,
        width=1280,
        height=720,
        rotation_degrees=90,
        capture_config_id="720p30",
    )
    return FrameMetadataMatch(
        metadata=metadata,
        decoded_frame_pts=0,
        decoded_frame_index=0,
        decoded_frame_time_base_num=1,
        decoded_frame_time_base_den=90_000,
        decoded_frame_received_at_client_monotonic_ns=3_001_000_000,
        timestamp_match_error_90khz=0,
    )


def _build_session(
    root: Path,
    *,
    lifecycle_state: str = "complete",
    media_hash: str | None = None,
    relative_media_path: str | None = None,
    manifest_frame_count: int = 2,
) -> Path:
    session_directory = root / SESSION_ID
    media_directory = session_directory / "media"
    telemetry_directory = session_directory / "telemetry"
    media_directory.mkdir(parents=True)
    telemetry_directory.mkdir()
    media_path = media_directory / f"{CLIP_ID}.mp4"
    media_path.write_bytes(b"synthetic-h264-mp4")

    database = CaptureSessionDatabase(
        SESSION_ID,
        telemetry_directory / "telemetry.sqlite",
    )
    database.record_imu_sample(
        CONNECTION_ID,
        _imu_sample(GatewayImuSensorType.GYROSCOPE, 0, 3_000),
        9_000,
    )
    database.record_imu_sample(
        CONNECTION_ID,
        _imu_sample(GatewayImuSensorType.ACCELEROMETER, 0, 1_000),
        9_001,
    )
    match = _frame_match()
    database.record_video_frame_metadata(
        CONNECTION_ID,
        match.metadata,
        3_001_000_000,
        1,
        "accepted",
    )
    database.record_frame_match(CONNECTION_ID, match)
    database.record_clip_frames(
        CLIP_ID,
        CONNECTION_ID,
        1,
        (
            RecordedVideoFrame(
                frame_index=0,
                source_frame_pts=0,
                source_frame_time_base_num=1,
                source_frame_time_base_den=90_000,
                mp4_pts=0,
                mp4_time_base_num=1,
                mp4_time_base_den=15_360,
                received_at_client_perf_counter_ns=3_001_000_000,
            ),
            RecordedVideoFrame(
                frame_index=1,
                source_frame_pts=None,
                source_frame_time_base_num=None,
                source_frame_time_base_den=None,
                mp4_pts=512,
                mp4_time_base_num=1,
                mp4_time_base_den=15_360,
                received_at_client_perf_counter_ns=3_034_000_000,
            ),
        ),
        expected_frame_count=2,
    )
    database.checkpoint_and_close()

    manifest = CaptureSessionManifest(
        session_id=SESSION_ID,
        display_name="Synthetic session",
        display_name_source="operator",
        lifecycle=CaptureSessionLifecycle(
            state=lifecycle_state,
            started_at_unix_ns=1,
            ended_at_unix_ns=2,
            end_reason="client_shutdown",
        ),
        session_time_origin=CaptureSessionTimeOrigin(),
        clips=[
            CaptureSessionClip(
                clip_id=CLIP_ID,
                state="complete",
                relative_media_path=f"media/{CLIP_ID}.mp4",
                video_profile=CaptureVideoProfile(
                    width=1280,
                    height=720,
                    nominal_fps=30.0,
                ),
                frame_count=2,
                sha256=_sha256(media_path),
            )
        ],
    ).model_dump(mode="json")
    manifest["clips"][0]["frame_count"] = manifest_frame_count
    if media_hash is not None:
        manifest["clips"][0]["sha256"] = media_hash
    if relative_media_path is not None:
        manifest["clips"][0]["relative_media_path"] = relative_media_path
    (session_directory / "session.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    return session_directory


def test_reader_loads_matched_and_unmatched_frames_with_exact_timestamps(
    tmp_path: Path,
) -> None:
    session_directory = _build_session(tmp_path)

    reader = CaptureSessionReader.open(session_directory)
    frames = list(reader.iter_frames(CLIP_ID))

    assert reader.session.session_id == SESSION_ID
    assert reader.session.clips[0].frame_count == 2
    assert [frame.frame_index for frame in frames] == [0, 1]
    assert frames[0].metadata_match_status is MetadataMatchStatus.EXACT
    assert frames[0].frame_metadata_match_id == 1
    assert frames[0].timestamp_match_error_90khz == 0
    assert frames[0].metadata_received_at_client_perf_counter_ns == 3_001_000_000
    assert frames[0].camera_start_generation == 1
    assert frames[0].rotation_degrees == 90
    assert frames[0].source_frame_timestamp is not None
    assert frames[0].source_frame_timestamp.presentation_time_seconds == 0
    assert frames[0].mp4_timestamp.presentation_time_seconds == 0
    assert frames[1].mp4_timestamp.pts == 512
    assert frames[1].metadata_match_status is MetadataMatchStatus.UNMATCHED
    assert frames[1].video_frame_metadata_id is None
    assert frames[1].source_frame_timestamp is None


def test_reader_preserves_imu_database_order_and_raw_values(tmp_path: Path) -> None:
    reader = CaptureSessionReader.open(_build_session(tmp_path))

    samples = list(reader.iter_imu_samples())

    assert [sample.sample_id for sample in samples] == [1, 2]
    assert [sample.sensor_event_monotonic_ns for sample in samples] == [3_000, 1_000]
    assert samples[0].sensor_type is ImuSensorType.GYROSCOPE
    assert samples[0].unit == "rad_s"
    assert samples[1].sensor_type is ImuSensorType.ACCELEROMETER
    assert samples[1].values == (0.1, 0.2, 9.8)
    assert all(sample.stored_alignment.status == "pending" for sample in samples)


def test_reader_does_not_modify_telemetry_database(tmp_path: Path) -> None:
    session_directory = _build_session(tmp_path)
    before = _tree_hashes(session_directory)
    reader = CaptureSessionReader.open(session_directory)

    list(reader.iter_frames(CLIP_ID))
    list(reader.iter_imu_samples())

    assert _tree_hashes(session_directory) == before


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"lifecycle_state": "incomplete"}, "must be complete"),
        ({"media_hash": "0" * 64}, "hash mismatch"),
        ({"manifest_frame_count": 3}, "frame index count"),
        ({"relative_media_path": "../outside.mp4"}, "escapes"),
    ],
)
def test_reader_rejects_invalid_session_inputs(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    if kwargs.get("relative_media_path"):
        (tmp_path / "outside.mp4").write_bytes(b"outside")
    session_directory = _build_session(tmp_path, **kwargs)  # type: ignore[arg-type]

    with pytest.raises(CaptureSessionReadError, match=message):
        CaptureSessionReader.open(session_directory)


def test_reader_can_skip_media_hash_for_explicit_fast_replay(tmp_path: Path) -> None:
    session_directory = _build_session(tmp_path, media_hash="0" * 64)

    reader = CaptureSessionReader.open(session_directory, verify_media_hashes=False)

    assert reader.session.clips[0].clip_id == CLIP_ID


def test_reader_rejects_unknown_clip_without_silently_returning_empty(tmp_path: Path) -> None:
    reader = CaptureSessionReader.open(_build_session(tmp_path))

    with pytest.raises(KeyError, match="unknown complete clip"):
        list(reader.iter_frames("missing"))


def test_reader_rejects_invalid_imu_contract_row(tmp_path: Path) -> None:
    session_directory = _build_session(tmp_path)
    _mutate_telemetry(
        session_directory,
        "UPDATE imu_samples SET unit = 'invalid' WHERE sample_id = 1",
    )
    reader = CaptureSessionReader.open(session_directory)

    with pytest.raises(CaptureSessionReadError, match="invalid IMU"):
        list(reader.iter_imu_samples())


def test_reader_rejects_non_contiguous_frame_indices(tmp_path: Path) -> None:
    session_directory = _build_session(tmp_path)
    _mutate_telemetry(
        session_directory,
        "UPDATE video_frame_index SET frame_index = 2 WHERE frame_index = 1",
    )

    with pytest.raises(CaptureSessionReadError, match="not contiguous"):
        CaptureSessionReader.open(session_directory)


def test_reader_rejects_uncheckpointed_wal_instead_of_reading_stale_data(
    tmp_path: Path,
) -> None:
    session_directory = _build_session(tmp_path)
    database_path = session_directory / "telemetry" / "telemetry.sqlite"
    database = sqlite3.connect(database_path)
    try:
        database.execute("UPDATE video_frame_index SET frame_index = 2 WHERE frame_index = 1")
        database.commit()
        wal_path = database_path.with_name(f"{database_path.name}-wal")
        assert wal_path.stat().st_size > 0

        with pytest.raises(CaptureSessionReadError, match="uncheckpointed wal"):
            CaptureSessionReader.open(session_directory)
    finally:
        database.close()


def test_reader_rejects_non_integer_frame_timestamp(tmp_path: Path) -> None:
    session_directory = _build_session(tmp_path)
    _mutate_telemetry(
        session_directory,
        "UPDATE video_frame_index SET mp4_pts = 1.5 WHERE frame_index = 0",
    )
    reader = CaptureSessionReader.open(session_directory)

    with pytest.raises(CaptureSessionReadError, match="invalid video frame"):
        list(reader.iter_frames(CLIP_ID))


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE video_frame_metadata_raw SET frame_id = 7",
        "UPDATE video_frame_metadata_raw SET session_id = 'other-session'",
        "UPDATE video_frame_metadata_raw SET ingest_status = 'rejected'",
        "UPDATE video_frame_matches SET connection_session_id = 'other-connection'",
    ],
)
def test_reader_rejects_inconsistent_joined_camera_metadata(
    tmp_path: Path,
    statement: str,
) -> None:
    session_directory = _build_session(tmp_path)
    _mutate_telemetry(session_directory, statement)
    reader = CaptureSessionReader.open(session_directory)

    with pytest.raises(CaptureSessionReadError, match="invalid video frame"):
        list(reader.iter_frames(CLIP_ID))
