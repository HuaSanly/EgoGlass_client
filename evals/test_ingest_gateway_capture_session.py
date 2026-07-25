from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

from ingest_gateway.adapters.mp4_recorder import RecordedVideoFrame
from ingest_gateway.adapters.webrtc import WebRtcVideoRecordingSource
from ingest_gateway.capture_session import CaptureSessionDatabase
from ingest_gateway.recording import RecordingRuntime
from ingest_gateway.recording_models import (
    CaptureSessionLifecycle,
    CaptureSessionManifest,
    CaptureSessionState,
    CaptureSessionTimeOrigin,
)
from ingest_gateway.webrtc_matcher import FrameMetadataMatch
from ingest_gateway.webrtc_models import (
    ImuSample,
    ImuSensorType,
    VideoFrameMetadata,
)

SESSION_ID = "7" * 32
CONNECTION_ID = "8" * 32


def _imu(sensor_type: ImuSensorType, sequence: int, event_ns: int) -> ImuSample:
    return ImuSample(
        sensor_type=sensor_type,
        android_sensor_type=1 if sensor_type is ImuSensorType.ACCELEROMETER else 4,
        sequence_number=sequence,
        sensor_event_monotonic_ns=event_ns,
        received_at_elapsed_realtime_ns=event_ns + 100_000,
        accuracy=3,
        values=(0.1, 0.2, 0.3),
    )


def _metadata(frame_id: int, source_pts: int) -> tuple[VideoFrameMetadata, FrameMetadataMatch]:
    sdk_ms = 1_000 + frame_id * 33
    callback_ns = 2_000_000_000 + sdk_ms * 1_000_000
    metadata = VideoFrameMetadata(
        frame_id=frame_id,
        camera_start_generation=1,
        captured_at_rokid_sdk_ms=sdk_ms,
        received_at_elapsed_realtime_ns=callback_ns,
        video_at_monotonic_ns=callback_ns - 50_000,
        rtp_timestamp_90khz=(90_000 + source_pts) & 0xFFFFFFFF,
        width=1280,
        height=720,
        capture_config_id="720p30",
    )
    return metadata, FrameMetadataMatch(
        metadata=metadata,
        decoded_frame_pts=source_pts,
        decoded_frame_index=frame_id,
        decoded_frame_time_base_num=1,
        decoded_frame_time_base_den=90_000,
        decoded_frame_received_at_client_monotonic_ns=5_000_000_000 + frame_id,
        timestamp_match_error_90khz=0,
    )


class _Source:
    def subscribe(self, *, buffered: bool) -> object:
        assert buffered
        return object()


class _Writer:
    next_index = 0

    def __init__(self, path: Path, _track: object) -> None:
        self.path = path
        self.finished = asyncio.Event()
        index = _Writer.next_index
        _Writer.next_index += 1
        self.frames_received = 1
        self.frame_records = (
            RecordedVideoFrame(
                frame_index=0,
                source_frame_pts=index * 3_000,
                source_frame_time_base_num=1,
                source_frame_time_base_den=90_000,
                mp4_pts=index * 512,
                mp4_time_base_num=1,
                mp4_time_base_den=15_360,
                received_at_client_perf_counter_ns=6_000_000_000 + index,
            ),
        )

    async def start(self) -> None:
        return None

    async def wait(self) -> None:
        await self.finished.wait()

    async def stop(self) -> None:
        self.path.write_bytes(b"synthetic-mp4")


async def _no_delay(_seconds: float) -> None:
    return None


async def _wait_for_recording(runtime: RecordingRuntime) -> None:
    for _attempt in range(50):
        if (await runtime.status()).state == "recording":
            return
        await asyncio.sleep(0)
    raise AssertionError("recording did not start")


def test_multiclip_collection_keeps_continuous_imu_and_raw_metadata() -> None:
    async def scenario(tmp_path: Path) -> None:
        _Writer.next_index = 0
        session_ids = iter([SESSION_ID, "9" * 32])
        runtime = RecordingRuntime(
            tmp_path,
            lambda: asyncio.sleep(
                0,
                result=WebRtcVideoRecordingSource(
                    CONNECTION_ID,
                    _Source(),
                    1280,
                    720,
                    1,
                ),
            ),
            recorder_factory=_Writer,
            sleep=_no_delay,
            session_id_factory=lambda: next(session_ids),
        )
        for sequence in range(3):
            if sequence < 2:
                await runtime.start()
                await _wait_for_recording(runtime)
                metadata, match = _metadata(sequence, sequence * 3_000)
                await runtime.on_video_frame_metadata(
                    CONNECTION_ID,
                    metadata,
                    5_000_000_000 + sequence,
                    1,
                    "accepted",
                )
                await runtime.on_frame_metadata_match(CONNECTION_ID, match)
                if sequence == 0:
                    unmatched, _unused = _metadata(99, 99_000)
                    await runtime.on_video_frame_metadata(
                        CONNECTION_ID,
                        unmatched,
                        5_000_000_099,
                        1,
                        "accepted",
                    )
                await runtime.stop()
            event_ns = 1_000_000_000 + sequence * 10_000_000
            for sensor_type in ImuSensorType:
                await runtime.on_imu_sample(
                    CONNECTION_ID,
                    _imu(sensor_type, sequence, event_ns),
                    7_000_000_000 + sequence,
                )

        status = await runtime.session_command("new")
        assert status.session_id is None
        assert not (tmp_path / ("9" * 32)).exists()

        manifest = json.loads((tmp_path / SESSION_ID / "session.json").read_text(encoding="utf-8"))
        quality = json.loads((tmp_path / SESSION_ID / "quality.json").read_text(encoding="utf-8"))
        connection = sqlite3.connect(tmp_path / SESSION_ID / "telemetry" / "telemetry.sqlite")
        counts = connection.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM imu_samples),
                (SELECT COUNT(*) FROM video_frame_metadata_raw),
                (SELECT COUNT(*) FROM video_frame_index),
                (SELECT COUNT(*) FROM clock_mapping_segments),
                (SELECT COUNT(*) FROM imu_samples
                 WHERE alignment_status = 'pending'
                   AND session_time_ns IS NULL
                   AND timestamp_uncertainty_ns IS NULL
                   AND clock_mapping_segment_id IS NULL),
                (SELECT COUNT(*) FROM video_frame_index
                 WHERE alignment_status = 'pending'
                   AND session_time_ns IS NULL
                   AND timestamp_uncertainty_ns IS NULL
                   AND clock_mapping_segment_id IS NULL)
            """
        ).fetchone()
        connection.close()

        assert manifest["lifecycle"]["state"] == "complete"
        assert [clip["state"] for clip in manifest["clips"]] == ["complete", "complete"]
        assert counts[:3] == (6, 3, 2)
        assert counts[3:] == (0, 6, 2)
        assert manifest["session_time_origin"]["status"] == "pending"
        assert all(
            clip["started_at_session_time_ns"] is None and clip["ended_at_session_time_ns"] is None
            for clip in manifest["clips"]
        )
        assert quality["counts"]["unmatched_video_metadata_count"] == 1
        assert quality["counts"]["unaligned_imu_sample_count"] == 6
        assert quality["counts"]["clock_mapping_segment_count"] == 0
        assert quality["training_eligibility"] == "ineligible"
        assert any(
            issue["issue_id"] == "perception_processing_required" for issue in quality["issues"]
        )
        assert any(
            issue["issue_id"] == "capture_provenance_incomplete" for issue in quality["issues"]
        )

    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        asyncio.run(scenario(Path(directory)))


def test_startup_recovery_preserves_unclean_session_as_incomplete() -> None:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        root = Path(directory)
        session_directory = root / SESSION_ID
        (session_directory / "telemetry").mkdir(parents=True)
        for relative in ("media", "annotations", "derived"):
            (session_directory / relative).mkdir()
        manifest = CaptureSessionManifest(
            session_id=SESSION_ID,
            display_name="Recovery eval",
            display_name_source="operator",
            lifecycle=CaptureSessionLifecycle(
                state=CaptureSessionState.ACTIVE,
                started_at_unix_ns=1,
            ),
            session_time_origin=CaptureSessionTimeOrigin(),
            clips=[],
        )
        (session_directory / "session.json").write_text(
            json.dumps(manifest.model_dump(mode="json")),
            encoding="utf-8",
        )
        database = CaptureSessionDatabase(
            SESSION_ID,
            session_directory / "telemetry" / "telemetry.sqlite",
        )
        for sensor_type in ImuSensorType:
            database.record_imu_sample(
                CONNECTION_ID,
                _imu(sensor_type, 0, 1_000),
                2_000,
            )
        database.flush()
        database.checkpoint_and_close()

        runtime = RecordingRuntime(root, lambda: asyncio.sleep(0, result=None))
        recovered = json.loads((session_directory / "session.json").read_text(encoding="utf-8"))
        quality = json.loads((session_directory / "quality.json").read_text(encoding="utf-8"))

        assert recovered["lifecycle"]["state"] == "incomplete"
        assert recovered["lifecycle"]["end_reason"] == "recovery_finalization"
        assert quality["status"] == "incomplete"
        assert quality["training_eligibility"] == "ineligible"
        assert quality["recoverable"] is True
        assert asyncio.run(runtime.library()).sessions[0].recoverable is True
