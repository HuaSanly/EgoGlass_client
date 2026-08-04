from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from ingest_gateway.adapters.mp4_recorder import RecordedVideoFrame
from ingest_gateway.capture_session import CaptureSessionDatabase
from ingest_gateway.webrtc_matcher import FrameMetadataMatch
from ingest_gateway.webrtc_models import VideoFrameMetadata
from perception.sensor_preprocessing import (
    PreparedFrameBundle,
    SensorCalibration,
    SensorPreprocessingConfig,
)
from perception.spatial_perception.hand_tracking import HandTrackingConfig
from perception.video_processing import (
    ProcessingJob,
    ProcessingJobState,
    ProcessingPreset,
    ProcessingResultStore,
    SessionProcessingRunner,
)
from tests.test_sensor_preprocessing_pipeline import (
    CLIP_ID,
    CONNECTION_ID,
    SESSION_ID,
    _recorded_session,
    _write_calibration,
    _write_preprocessing_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FakeTracker:
    def __init__(self) -> None:
        self.frames: list[tuple[str, int, int]] = []

    def process_frame(self, bundle: PreparedFrameBundle) -> _FakeResult:
        self.frames.append(
            (bundle.sequence_id, bundle.frame_index, bundle.session_time_ns)
        )
        return _FakeResult(bundle)


class _FakeResult:
    def __init__(self, bundle: PreparedFrameBundle) -> None:
        self.bundle = bundle
        self.hands: tuple[object, ...] = ()

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "session_id": self.bundle.session_id,
            "sequence_id": self.bundle.sequence_id,
            "frame_index": self.bundle.frame_index,
            "session_time_ns": self.bundle.session_time_ns,
            "source_image_width_px": self.bundle.image_bgr.shape[1],
            "source_image_height_px": self.bundle.image_bgr.shape[0],
            "hands": [],
        }


def _add_strict_frame_evidence(
    session: Path,
    timings: list[tuple[int, Fraction]],
) -> None:
    database = CaptureSessionDatabase(
        SESSION_ID,
        session / "telemetry" / "telemetry.sqlite",
    )
    database._connection.execute("DELETE FROM video_frame_index")
    recorded: list[RecordedVideoFrame] = []
    for index, (mp4_pts, mp4_time_base) in enumerate(timings):
        callback_ns = 10_000_000 + index * 10_000_000
        metadata = VideoFrameMetadata(
            frame_id=index,
            camera_start_generation=1,
            captured_at_rokid_sdk_ms=1000 + index * 33,
            received_at_elapsed_realtime_ns=callback_ns + 100,
            video_at_monotonic_ns=callback_ns,
            rtp_timestamp_90khz=index * 3000,
            width=8,
            height=6,
            capture_config_id="tiny",
        )
        match = FrameMetadataMatch(
            metadata=metadata,
            decoded_frame_pts=index,
            decoded_frame_index=index,
            decoded_frame_time_base_num=1,
            decoded_frame_time_base_den=30,
            decoded_frame_received_at_client_monotonic_ns=callback_ns + 200,
            timestamp_match_error_90khz=0,
        )
        database.record_video_frame_metadata(
            CONNECTION_ID,
            metadata,
            callback_ns + 200,
            1,
            "accepted",
        )
        database.record_frame_match(CONNECTION_ID, match)
        recorded.append(
            RecordedVideoFrame(
                frame_index=index,
                source_frame_pts=index,
                source_frame_time_base_num=1,
                source_frame_time_base_den=30,
                mp4_pts=mp4_pts,
                mp4_time_base_num=mp4_time_base.numerator,
                mp4_time_base_den=mp4_time_base.denominator,
                received_at_client_perf_counter_ns=callback_ns + 200,
            )
        )
    database.record_clip_frames(CLIP_ID, CONNECTION_ID, 1, recorded, len(recorded))
    database.checkpoint_and_close()


def test_session_runner_writes_complete_immutable_run_from_capture_evidence(
    tmp_path: Path,
) -> None:
    session, timings = _recorded_session(tmp_path)
    _add_strict_frame_evidence(session, timings)
    calibration = _write_calibration(tmp_path / "calibration.json")
    sensor_config = _write_preprocessing_config(
        tmp_path / "sensor-preprocessing.yaml",
        calibration.name,
        verify_media_hashes=True,
        decode_threads=1,
    )
    tracker = _FakeTracker()
    runner = SessionProcessingRunner(
        sensor_config_path=sensor_config,
        tracker_factory=lambda: tracker,  # type: ignore[arg-type]
    )
    configuration_snapshot = json.dumps(
        {
            "sensor_preprocessing": SensorPreprocessingConfig.load(
                sensor_config
            ).model_dump(mode="json"),
            "sensor_calibration": SensorCalibration.load(calibration).model_dump(
                mode="json"
            ),
            # Legacy queued tasks used this key. New tasks use
            # ``offline_hand_tracking`` and are covered by runtime tests.
            "hand_tracking": HandTrackingConfig.load(
                PROJECT_ROOT / "config" / "offline-hand-tracking.yaml"
            ).model_dump(mode="json"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    sensor_config.write_text("invalid: [", encoding="utf-8")
    job = ProcessingJob(
        job_id="job-test",
        session_id=SESSION_ID,
        clip_id=None,
        preset=ProcessingPreset(inference_stride_frames=1),
        state=ProcessingJobState.RUNNING,
        created_at_unix_ns=1,
        updated_at_unix_ns=1,
        configuration_revision=4,
        configuration_sha256_by_file=(("hand-tracking.yaml", "deadbeef"),),
        configuration_snapshot_json=configuration_snapshot,
    )
    output = session / "derived" / "video-processing" / "run-test"
    progress: list[tuple[int, int]] = []

    summary = runner.run(
        job,
        session,
        output,
        progress=lambda current, total: progress.append((current, total)),
        is_canceled=lambda: False,
    )

    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    results = ProcessingResultStore(output / "results.sqlite", read_only=True)
    assert summary.input_frame_count == summary.inferred_frame_count == 2
    assert summary.detected_hand_count == 0
    assert progress == [(1, 2), (2, 2)]
    assert [identity[:2] for identity in tracker.frames] == [
        (CLIP_ID, 0),
        (CLIP_ID, 1),
    ]
    assert tracker.frames[1][2] > tracker.frames[0][2]
    assert manifest["state"] == "completed"
    assert manifest["input_frame_count"] == 2
    assert manifest["configuration"] == {
        "revision": 4,
        "sha256_by_file": {"hand-tracking.yaml": "deadbeef"},
        "snapshot": json.loads(configuration_snapshot),
    }
    assert results.count() == 2
    assert (output / "run.log").read_text(encoding="utf-8").endswith(
        "run completed\n"
    )

    runner.release_gpu()
    assert runner._tracker is None
