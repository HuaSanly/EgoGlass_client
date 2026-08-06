from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from process_video import process_session
from tests.test_sensor_preprocessing_pipeline import (
    _recorded_session,
    _write_calibration,
    _write_preprocessing_config,
)
from tests.test_video_processing_runner import _add_strict_frame_evidence, _FakeTracker


def test_headless_cli_help_starts_without_qt() -> None:
    repository = Path(__file__).parents[1]
    result = subprocess.run(
        [sys.executable, str(repository / "src" / "process_video.py"), "--help"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "without Qt" in result.stdout
    assert "PyQt6" not in result.stderr


def test_headless_run_artifacts_are_separate_from_ui_job_storage() -> None:
    source = (Path(__file__).parents[1] / "src" / "process_video.py").read_text(
        encoding="utf-8"
    )

    assert "results.jsonl" in source
    assert "jobs.sqlite3" not in source
    assert "ui.processing" not in source


def test_headless_vio_failure_writes_raw_and_viewable_final_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = Path(__file__).parents[1]
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
    monkeypatch.setattr(
        "process_video.HumanEgoHandTrackingPipeline.from_config",
        lambda _config: tracker,
    )
    monkeypatch.setattr(
        "process_video.run_vio_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("Basalt unavailable")),
    )
    output = tmp_path / "headless-run"

    summary = process_session(
        session,
        output,
        sensor_config_path=sensor_config,
        hand_tracking_config_path=repository / "config" / "offline-hand-tracking.yaml",
    )

    raw = [json.loads(line) for line in (output / "raw_results.jsonl").read_text().splitlines()]
    final = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert summary["state"] == "partial"
    assert summary["vio_error"] == "Basalt unavailable"
    assert set(summary["temporal_processing"]["stage_durations_ns"]) == {
        "confidence_filter",
        "interpolation",
        "segment_suppression",
        "grasp_smoothing",
        "world_mapping",
        "kinematic_optimization",
    }
    assert len(raw) == len(final) == 2
    assert raw == final
    assert not (output / "jobs.sqlite3").exists()
