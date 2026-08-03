from __future__ import annotations

import json
from pathlib import Path

from perception.video_processing import (
    ProcessingJob,
    ProcessingJobState,
    ProcessingPreset,
    ProcessingResultStore,
    SessionProcessingRunner,
)
from tests.test_sensor_preprocessing_pipeline import (
    SESSION_ID,
    _recorded_session,
    _write_calibration,
    _write_preprocessing_config,
)
from tests.test_video_processing_runner import _add_strict_frame_evidence, _FakeTracker


def test_case_vp_005_pipeline_preserves_frame_identity_through_result_index(
    tmp_path: Path,
) -> None:
    session, timings = _recorded_session(tmp_path)
    _add_strict_frame_evidence(session, timings)
    calibration = _write_calibration(tmp_path / "calibration.json")
    sensor_config = _write_preprocessing_config(
        tmp_path / "sensor-preprocessing.yaml",
        calibration.name,
        decode_threads=1,
    )
    tracker = _FakeTracker()
    runner = SessionProcessingRunner(
        sensor_config_path=sensor_config,
        tracker_factory=lambda: tracker,  # type: ignore[arg-type]
    )
    job = ProcessingJob(
        "eval-job",
        SESSION_ID,
        None,
        ProcessingPreset(inference_stride_frames=1),
        ProcessingJobState.RUNNING,
        1,
        1,
    )
    output = session / "derived" / "video-processing" / "eval-run"

    runner.run(
        job,
        session,
        output,
        progress=lambda _current, _total: None,
        is_canceled=lambda: False,
    )

    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    store = ProcessingResultStore(output / "results.sqlite", read_only=True)
    identities = [
        (clip_id, frame_index, session_time_ns)
        for clip_id, frame_index, session_time_ns in tracker.frames
    ]
    indexed = [
        store.result_for_frame(clip_id, frame_index, session_time_ns)
        for clip_id, frame_index, session_time_ns in identities
    ]

    assert manifest["state"] == "completed"
    assert [
        (
            result["sequence_id"],
            result["frame_index"],
            result["session_time_ns"],
        )
        for result in indexed
        if result is not None
    ] == identities
