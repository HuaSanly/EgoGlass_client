from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from hand_tracking import HandTrackingConfig, HandTrackingResult, TemporalProcessingStats
from sensor_preprocessing import (
    PreparedFrameBundle,
    SensorCalibration,
    SensorPreprocessingConfig,
)
from tests.test_sensor_preprocessing_pipeline import (
    CLIP_ID,
    CONNECTION_ID,
    SESSION_ID,
    _recorded_session,
    _write_calibration,
    _write_preprocessing_config,
)
from ui.gateway.adapters.mp4_recorder import RecordedVideoFrame
from ui.gateway.capture_session import CaptureSessionDatabase
from ui.gateway.webrtc_matcher import FrameMetadataMatch
from ui.gateway.webrtc_models import VideoFrameMetadata
from ui.processing import (
    ProcessingJob,
    ProcessingJobState,
    ProcessingPreset,
    ProcessingResultStore,
    SessionProcessingRunner,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FakeTracker:
    def __init__(self) -> None:
        self.frames: list[tuple[str, int, int]] = []

    def process_frame(self, bundle: PreparedFrameBundle) -> HandTrackingResult:
        self.frames.append(
            (bundle.sequence_id, bundle.frame_index, bundle.session_time_ns)
        )
        return HandTrackingResult(
            schema_version="1.0",
            session_id=bundle.session_id,
            sequence_id=bundle.sequence_id,
            frame_index=bundle.frame_index,
            session_time_ns=bundle.session_time_ns,
            timestamp_uncertainty_ns=bundle.timestamp_uncertainty_ns,
            image_width_px=bundle.image_bgr.shape[1],
            image_height_px=bundle.image_bgr.shape[0],
            source_rotation_degrees=0,
            detector_backend="fake",
            requested_device="cpu",
            execution_device="cpu",
            hamer_loaded=False,
            inference_duration_ns=0,
            hands=(),
        )


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


def test_session_runner_writes_viewable_partial_run_without_vio(
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
    assert manifest["state"] == "partial"
    assert manifest["input_frame_count"] == 2
    assert manifest["configuration"] == {
        "revision": 4,
        "sha256_by_file": {"hand-tracking.yaml": "deadbeef"},
        "snapshot": json.loads(configuration_snapshot),
    }
    assert results.count() == 2
    assert results.count(raw=True) == 2
    assert (output / "run.log").read_text(encoding="utf-8").endswith(
        "run partial\n"
    )

    runner.release_gpu()
    assert runner._tracker is None


def test_run_manifest_binds_exact_vio_and_records_temporal_stage_metrics(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run-bound"
    output.mkdir()
    job = ProcessingJob(
        job_id="job-bound",
        session_id=SESSION_ID,
        clip_id=CLIP_ID,
        preset=ProcessingPreset(),
        state=ProcessingJobState.RUNNING,
        created_at_unix_ns=1,
        updated_at_unix_ns=1,
    )
    stats = TemporalProcessingStats(
        10,
        1,
        2,
        1,
        10,
        4,
        1,
        3,
        8,
        7,
        1234,
        confidence_filter_duration_ns=11,
        interpolation_duration_ns=12,
        segment_suppression_duration_ns=13,
        grasp_smoothing_duration_ns=14,
        world_mapping_duration_ns=15,
        kinematic_optimization_duration_ns=16,
    )

    SessionProcessingRunner._write_manifest(
        output / "run.json",
        job,
        state="partial",
        started_at_unix_ns=1,
        completed_at_unix_ns=2,
        vio_run_id="vio-bound",
        temporal_stats=stats,
    )

    payload = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert payload["vio_run_id"] == "vio-bound"
    assert payload["temporal_processing"]["interpolated_frames"] == 2
    assert payload["temporal_processing"]["vio_coverage_ratio"] == 0.8
    assert payload["temporal_processing"]["stage_durations_ns"] == {
        "confidence_filter": 11,
        "grasp_smoothing": 14,
        "interpolation": 12,
        "kinematic_optimization": 16,
        "segment_suppression": 13,
        "world_mapping": 15,
    }
