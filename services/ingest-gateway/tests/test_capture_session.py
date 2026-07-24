from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from egoglass_ingest_gateway.adapters.mp4_recorder import RecordedVideoFrame
from egoglass_ingest_gateway.adapters.webrtc import WebRtcVideoRecordingSource
from egoglass_ingest_gateway.capture_session import (
    CaptureSessionDatabase,
    CaptureSessionWriter,
    read_capture_quality,
)
from egoglass_ingest_gateway.recording import RecordingRuntime
from egoglass_ingest_gateway.recording_models import (
    CaptureSessionClip,
    CaptureVideoProfile,
)
from egoglass_ingest_gateway.webrtc_matcher import FrameMetadataMatch
from egoglass_ingest_gateway.webrtc_models import (
    ImuSample,
    ImuSensorType,
    VideoFrameMetadata,
)

SESSION_ID = "c" * 32
CONNECTION_ID = "d" * 32


def test_complete_clip_contract_rejects_missing_frame_count_and_hash() -> None:
    with pytest.raises(ValueError, match="complete clip requires"):
        CaptureSessionClip(
            clip_id="a" * 32,
            state="complete",
            relative_media_path=f"media/{'a' * 32}.mp4",
            video_profile=CaptureVideoProfile(
                width=1280,
                height=720,
                nominal_fps=30,
            ),
            frame_count=1,
        )


def test_complete_clip_contract_allows_deferred_session_time() -> None:
    clip = CaptureSessionClip(
        clip_id="a" * 32,
        state="complete",
        relative_media_path=f"media/{'a' * 32}.mp4",
        video_profile=CaptureVideoProfile(
            width=1280,
            height=720,
            nominal_fps=30,
        ),
        frame_count=1,
        sha256="b" * 64,
    )

    assert clip.started_at_session_time_ns is None
    assert clip.ended_at_session_time_ns is None


def imu_sample(sensor_type: str, sequence: int, event_ns: int) -> ImuSample:
    sensor = ImuSensorType(sensor_type)
    return ImuSample(
        sensor_type=sensor,
        android_sensor_type=1 if sensor is ImuSensorType.ACCELEROMETER else 4,
        sequence_number=sequence,
        sensor_event_monotonic_ns=event_ns,
        received_at_elapsed_realtime_ns=event_ns + 200_000,
        accuracy=3,
        values=(0.1, 0.2, 9.8),
    )


def frame_match(
    frame_id: int,
    sdk_ms: int,
    callback_ns: int,
    *,
    camera_start_generation: int = 1,
) -> FrameMetadataMatch:
    metadata = VideoFrameMetadata(
        frame_id=frame_id,
        camera_start_generation=camera_start_generation,
        captured_at_rokid_sdk_ms=sdk_ms,
        received_at_elapsed_realtime_ns=callback_ns,
        video_at_monotonic_ns=callback_ns - 100_000,
        rtp_timestamp_90khz=(frame_id * 3_000) & 0xFFFFFFFF,
        width=1280,
        height=720,
        capture_config_id="720p30",
    )
    return FrameMetadataMatch(
        metadata=metadata,
        decoded_frame_pts=frame_id * 3_000,
        decoded_frame_index=frame_id,
        decoded_frame_time_base_num=1,
        decoded_frame_time_base_den=90_000,
        decoded_frame_received_at_client_monotonic_ns=callback_ns + 1_000_000,
        timestamp_match_error_90khz=0,
    )


def recorded_frame(frame_index: int, source_pts: int, mp4_pts: int) -> RecordedVideoFrame:
    return RecordedVideoFrame(
        frame_index=frame_index,
        source_frame_pts=source_pts,
        source_frame_time_base_num=1,
        source_frame_time_base_den=90_000,
        mp4_pts=mp4_pts,
        mp4_time_base_num=1,
        mp4_time_base_den=15_360,
        received_at_client_perf_counter_ns=9_000_000_000 + frame_index,
    )


def test_capture_finalization_preserves_source_clocks_without_alignment(
    tmp_path: Path,
) -> None:
    path = tmp_path / "telemetry.sqlite"
    database = CaptureSessionDatabase(SESSION_ID, path)
    for sequence in range(12):
        event_ns = 1_000_000_000 + sequence * 10_000_000
        database.record_imu_sample(
            CONNECTION_ID,
            imu_sample("accelerometer", sequence, event_ns),
            5_000_000_000 + sequence,
        )
        database.record_imu_sample(
            CONNECTION_ID,
            imu_sample("gyroscope", sequence, event_ns + 1_000),
            5_000_100_000 + sequence,
        )
    for frame_id in range(15):
        sdk_ms = 1_000 + frame_id * 33
        callback_ns = 2_000_000_000 + sdk_ms * 1_000_100
        if frame_id == 7:
            callback_ns += 20_000_000
        match = frame_match(frame_id, sdk_ms, callback_ns)
        database.record_video_frame_metadata(
            CONNECTION_ID,
            match.metadata,
            callback_ns + 1_000_000,
            1,
            "accepted",
        )
        database.record_frame_match(CONNECTION_ID, match)
    reset_match = frame_match(
        100,
        10,
        4_000_000_000,
        camera_start_generation=2,
    )
    database.record_video_frame_metadata(
        CONNECTION_ID,
        reset_match.metadata,
        4_001_000_000,
        2,
        "accepted",
    )
    database.record_frame_match(CONNECTION_ID, reset_match)
    rollback_match = frame_match(
        101,
        5,
        4_100_000_000,
        camera_start_generation=2,
    )
    database.record_video_frame_metadata(
        CONNECTION_ID,
        rollback_match.metadata,
        4_101_000_000,
        2,
        "accepted",
    )
    database.record_frame_match(CONNECTION_ID, rollback_match)
    database.flush()

    result = database.finalize_capture(0)
    database.checkpoint_and_close()

    connection = sqlite3.connect(path)
    mapping_count = connection.execute("SELECT COUNT(*) FROM clock_mapping_segments").fetchone()[0]
    imu_rows = connection.execute(
        """
        SELECT sensor_event_monotonic_ns, received_at_elapsed_realtime_ns,
               alignment_status, session_time_ns,
               timestamp_uncertainty_ns, clock_mapping_segment_id
        FROM imu_samples ORDER BY sample_id
        """
    ).fetchall()
    camera_rows = connection.execute(
        """
        SELECT captured_at_rokid_sdk_ms, received_at_elapsed_realtime_ns
        FROM video_frame_metadata_raw ORDER BY metadata_row_id
        """
    ).fetchall()
    metadata_keys = dict(connection.execute("SELECT key, value FROM session_metadata"))
    connection.close()

    assert mapping_count == 0
    assert len(imu_rows) == 24
    assert all(row[2:] == ("pending", None, None, None) for row in imu_rows)
    assert imu_rows[0][:2] == (1_000_000_000, 1_000_200_000)
    assert camera_rows[0][0] == 1_000
    assert camera_rows[-1][0] == 5
    assert "origin_elapsed_realtime_ns" not in metadata_keys
    assert "origin_event" not in metadata_keys
    assert result.quality.timestamp_mapping_segment_count == 0
    assert result.quality.unaligned_imu_sample_count == 24


def test_legacy_mapped_capture_quality_remains_readable(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.sqlite"
    database = CaptureSessionDatabase(SESSION_ID, path)
    database.record_imu_sample(
        CONNECTION_ID,
        imu_sample("accelerometer", 0, 1_000),
        10_000,
    )
    database._connection.execute(
        """
        INSERT INTO clock_mapping_segments(
            session_id, connection_session_id, source_instance_id,
            segment_index, clock_mapping_segment_id, source_clock_id,
            target_clock_id, source_from, source_to, target_from_ns,
            target_to_ns, scale_target_ns_per_source_unit, offset_target_ns,
            uncertainty_ns, fit_method, sample_count, residual_p95_ns,
            residual_max_ns, status, estimator_state, segment_reason,
            outlier_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            SESSION_ID,
            CONNECTION_ID,
            "accelerometer",
            0,
            "legacy-map",
            "sensor_event_monotonic_ns",
            "glasses_elapsed_realtime_ns",
            1_000,
            1_000,
            1_200,
            1_200,
            1.0,
            200.0,
            200,
            "identity",
            1,
            0,
            0,
            "provisional",
            "estimated",
            "session_start",
            0,
        ),
    )
    database._connection.execute(
        """
        UPDATE imu_samples
        SET alignment_status = 'mapped', session_time_ns = 0,
            timestamp_uncertainty_ns = 200,
            clock_mapping_segment_id = 'legacy-map'
        """
    )
    database.checkpoint_and_close()

    quality = read_capture_quality(path)

    assert quality.timestamp_mapping_segment_count == 1
    assert quality.rejected_clock_mapping_segment_count == 0
    assert quality.unaligned_imu_sample_count == 0


def test_every_mp4_frame_requires_exact_muxed_pts_and_connection_scoped_match(
    tmp_path: Path,
) -> None:
    database = CaptureSessionDatabase(SESSION_ID, tmp_path / "telemetry.sqlite")
    match = frame_match(1, 1_000, 3_000_000_000)
    database.record_video_frame_metadata(
        CONNECTION_ID,
        match.metadata,
        3_001_000_000,
        1,
        "accepted",
    )
    database.record_frame_match(CONNECTION_ID, match)
    other_match = frame_match(1, 1_000, 3_000_000_000)
    database.record_video_frame_metadata(
        "e" * 32,
        other_match.metadata,
        3_001_000_001,
        1,
        "accepted",
    )
    database.record_frame_match("e" * 32, other_match)
    frame = recorded_frame(0, match.decoded_frame_pts, 512)

    with pytest.raises(ValueError, match="every encoded frame"):
        database.record_clip_frames("a" * 32, CONNECTION_ID, 1, [frame], 2)

    database.record_clip_frames("a" * 32, CONNECTION_ID, 1, [frame], 1)
    database.flush()
    row = database._connection.execute(
        """
        SELECT mp4_pts, mp4_time_base_numerator,
               mp4_time_base_denominator, frame_id
        FROM video_frame_index
        """
    ).fetchone()
    database.checkpoint_and_close()

    assert row == (512, 1, 15_360, 1)


def test_mp4_match_is_scoped_to_camera_start_generation(tmp_path: Path) -> None:
    database = CaptureSessionDatabase(SESSION_ID, tmp_path / "telemetry.sqlite")
    old_match = frame_match(1, 1_000, 3_000_000_000)
    restarted_metadata = frame_match(
        2,
        10,
        4_000_000_000,
        camera_start_generation=2,
    ).metadata
    restarted_match = FrameMetadataMatch(
        metadata=restarted_metadata,
        decoded_frame_pts=old_match.decoded_frame_pts,
        decoded_frame_index=2,
        decoded_frame_time_base_num=1,
        decoded_frame_time_base_den=90_000,
        decoded_frame_received_at_client_monotonic_ns=4_001_000_000,
        timestamp_match_error_90khz=0,
    )
    for match, received_at_ns in (
        (old_match, 3_001_000_000),
        (restarted_match, 4_001_000_000),
    ):
        database.record_video_frame_metadata(
            CONNECTION_ID,
            match.metadata,
            received_at_ns,
            match.metadata.camera_start_generation,
            "accepted",
        )
        database.record_frame_match(CONNECTION_ID, match)
    frame = recorded_frame(0, old_match.decoded_frame_pts, 512)
    database.record_clip_frames("a" * 32, CONNECTION_ID, 2, [frame], 1)
    database.flush()

    frame_id = database._connection.execute("SELECT frame_id FROM video_frame_index").fetchone()[0]
    database.checkpoint_and_close()

    assert frame_id == 2


def test_mp4_match_compares_equivalent_rescaled_time_bases(tmp_path: Path) -> None:
    database = CaptureSessionDatabase(SESSION_ID, tmp_path / "telemetry.sqlite")
    match = frame_match(30, 1_000, 3_000_000_000)
    database.record_video_frame_metadata(
        CONNECTION_ID,
        match.metadata,
        3_001_000_000,
        1,
        "accepted",
    )
    database.record_frame_match(CONNECTION_ID, match)
    frame = RecordedVideoFrame(
        frame_index=0,
        source_frame_pts=30,
        source_frame_time_base_num=1,
        source_frame_time_base_den=30,
        mp4_pts=512,
        mp4_time_base_num=1,
        mp4_time_base_den=15_360,
        received_at_client_perf_counter_ns=9_000_000_000,
    )

    database.record_clip_frames("a" * 32, CONNECTION_ID, 1, [frame], 1)
    database.flush()
    frame_id = database._connection.execute("SELECT frame_id FROM video_frame_index").fetchone()[0]
    database.checkpoint_and_close()

    assert frame_id == 30


def test_bounded_writer_counts_overflow_and_drains_before_checkpoint(tmp_path: Path) -> None:
    async def scenario() -> None:
        database = CaptureSessionDatabase(SESSION_ID, tmp_path / "telemetry.sqlite")
        writer = CaptureSessionWriter(database, max_queue_size=2, max_batch_size=2)
        assert writer.enqueue(
            "record_imu_sample",
            CONNECTION_ID,
            imu_sample("accelerometer", 0, 1_000),
            10_000,
        )
        assert writer.enqueue(
            "record_imu_sample",
            CONNECTION_ID,
            imu_sample("gyroscope", 0, 1_100),
            10_100,
        )
        assert not writer.enqueue(
            "record_imu_sample",
            CONNECTION_ID,
            imu_sample("accelerometer", 1, 2_000),
            11_000,
        )

        result = await writer.finalize()

        assert result.quality.imu_sample_count == 2
        assert result.quality.telemetry_queue_overflow_count == 1
        assert not (tmp_path / "telemetry.sqlite-wal").exists()

    asyncio.run(scenario())


class _VideoSource:
    def subscribe(self, *, buffered: bool) -> object:
        assert buffered
        return object()


class _Recorder:
    def __init__(self, path: Path, _track: object) -> None:
        self.path = path
        self.frames_received = 1
        self.frame_records = (recorded_frame(0, 0, 0),)
        self.finished = asyncio.Event()

    async def start(self) -> None:
        return None

    async def wait(self) -> None:
        await self.finished.wait()

    async def stop(self) -> None:
        self.path.write_bytes(b"mp4")


async def _no_delay(_seconds: float) -> None:
    return None


def test_automatic_session_captures_countdown_and_between_clip_imu_then_new_arms_next(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        session_ids = iter(["1" * 32, "2" * 32])
        runtime = RecordingRuntime(
            tmp_path,
            lambda: asyncio.sleep(
                0,
                result=WebRtcVideoRecordingSource(
                    CONNECTION_ID,
                    _VideoSource(),
                    1280,
                    720,
                    1,
                ),
            ),
            recorder_factory=_Recorder,
            sleep=_no_delay,
            session_id_factory=lambda: next(session_ids),
        )
        started = await runtime.start()
        first_session_id = started.session_id
        assert first_session_id == "1" * 32
        await runtime.on_imu_sample(
            CONNECTION_ID,
            imu_sample("accelerometer", 0, 1_000),
            10_000,
        )
        await runtime.on_imu_sample(
            CONNECTION_ID,
            imu_sample("gyroscope", 0, 1_100),
            10_100,
        )
        for _attempt in range(20):
            if (await runtime.status()).state == "recording":
                break
            await asyncio.sleep(0)
        video_match = frame_match(0, 1_000, 3_000_000)
        await runtime.on_video_frame_metadata(
            CONNECTION_ID,
            video_match.metadata,
            4_000_000,
            1,
            "accepted",
        )
        await runtime.on_frame_metadata_match(CONNECTION_ID, video_match)
        await runtime.stop()
        await runtime.on_imu_sample(
            CONNECTION_ID,
            imu_sample("accelerometer", 1, 2_000),
            11_000,
        )
        await runtime.on_imu_sample(
            CONNECTION_ID,
            imu_sample("gyroscope", 1, 2_100),
            11_100,
        )

        new_status = await runtime.session_command("new")
        assert new_status.session_id is None
        assert new_status.session_state is None
        manifest = json.loads(
            (tmp_path / first_session_id / "session.json").read_text(encoding="utf-8")
        )
        quality = json.loads(
            (tmp_path / first_session_id / "quality.json").read_text(encoding="utf-8")
        )
        connection = sqlite3.connect(tmp_path / first_session_id / "telemetry" / "telemetry.sqlite")
        samples = connection.execute("SELECT COUNT(*) FROM imu_samples").fetchone()[0]
        pending_imu = connection.execute(
            """
            SELECT COUNT(*) FROM imu_samples
            WHERE alignment_status = 'pending'
              AND session_time_ns IS NULL
              AND timestamp_uncertainty_ns IS NULL
              AND clock_mapping_segment_id IS NULL
            """
        ).fetchone()[0]
        pending_frames = connection.execute(
            """
            SELECT COUNT(*) FROM video_frame_index
            WHERE alignment_status = 'pending'
              AND session_time_ns IS NULL
              AND timestamp_uncertainty_ns IS NULL
              AND clock_mapping_segment_id IS NULL
            """
        ).fetchone()[0]
        mapping_count = connection.execute(
            "SELECT COUNT(*) FROM clock_mapping_segments"
        ).fetchone()[0]
        connection.close()
        assert manifest["lifecycle"]["state"] == "complete"
        assert manifest["lifecycle"]["end_reason"] == "manual_new_session"
        assert manifest["clips"][0]["state"] == "complete"
        assert manifest["session_time_origin"] == {
            "status": "pending",
            "clock_id": "glasses_elapsed_realtime_ns",
            "origin_elapsed_realtime_ns": None,
            "origin_event": None,
        }
        assert manifest["clips"][0]["requested_at_session_time_ns"] is None
        assert manifest["clips"][0]["started_at_session_time_ns"] is None
        assert manifest["clips"][0]["ended_at_session_time_ns"] is None
        assert manifest["provenance"]["device"]["manufacturer"] == "Rokid"
        assert quality["counts"]["accelerometer_sample_count"] == 2
        assert quality["counts"]["unaligned_imu_sample_count"] == 4
        assert quality["counts"]["clock_mapping_segment_count"] == 0
        assert quality["training_eligibility"] == "ineligible"
        assert quality["checks"][0]["check_id"] == "perception_alignment"
        assert quality["checks"][0]["status"] == "not_evaluated"
        assert samples == 4
        assert pending_imu == 4
        assert pending_frames == 1
        assert mapping_count == 0
        assert not (tmp_path / ("2" * 32)).exists()

        second = await runtime.start()
        assert second.session_id == "2" * 32
        await runtime.session_command("finalize")

    asyncio.run(scenario())
