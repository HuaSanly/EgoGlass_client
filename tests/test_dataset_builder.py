from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from qfluentwidgets import TableWidget

from schemas import QualityGate, QualityIssue, QualitySeverity
from tests.test_sensor_preprocessing_pipeline import CLIP_ID, SESSION_ID, _recorded_session
from tests.test_video_processing_runner import _add_strict_frame_evidence
from ui.dataset_builder import (
    DatasetBuilder,
    DatasetBuildError,
    DatasetCandidateSummary,
    DatasetCatalogService,
    DatasetQualityChecker,
    EpisodeInterval,
    split_valid_intervals,
)
from ui.processing import ProcessingResultStore
from ui.views.dataset_builder import DatasetView


def test_invalid_regions_split_virtual_episodes_without_touching_media() -> None:
    issue = QualityIssue(
        issue_id="blur",
        gate=QualityGate.SOFT,
        severity=QualitySeverity.WARNING,
        message="blurred frames",
        clip_id="clip",
        start_frame_index=3,
        end_frame_index_exclusive=5,
        restorable=True,
    )
    assert split_valid_intervals("clip", tuple(range(10)), (issue,), minimum_frames=2) == (
        EpisodeInterval("clip", 0, 3),
        EpisodeInterval("clip", 5, 10),
    )


def test_soft_quality_override_is_auditable_and_hard_gates_are_not_restored() -> None:
    issue = QualityIssue(
        issue_id="iou",
        gate=QualityGate.SOFT,
        severity=QualitySeverity.WARNING,
        message="overlap",
        clip_id="clip",
        start_frame_index=2,
        end_frame_index_exclusive=3,
        restorable=True,
    )
    from ui.dataset_builder.models import DatasetQualityReport

    report = DatasetQualityReport("session", "run", (), (issue,), ())
    restored = DatasetQualityChecker.restore_soft_issue(
        report,
        "iou",
        operator="reviewer",
        reason="manual frame review confirms two hands",
        restored_at_unix_ns=123,
    )
    assert restored.publishable
    assert restored.soft_issues[0].restored_by == "reviewer"
    assert restored.soft_issues[0].restore_reason == "manual frame review confirms two hands"


def test_builder_accepts_only_audited_soft_gate_restorations(tmp_path: Path) -> None:
    session, timings = _recorded_session(tmp_path)
    _add_strict_frame_evidence(session, timings)
    run = _write_complete_run(session)
    store = ProcessingResultStore(run / "results.sqlite")
    with sqlite3.connect(run / "results.sqlite") as database:
        database.execute("DELETE FROM frame_results")
    for frame_index in range(2):
        store.put_final(
            {
                "session_id": SESSION_ID,
                "sequence_id": CLIP_ID,
                "frame_index": frame_index,
                "session_time_ns": 1_000_000_000 + frame_index * 33_333_333,
                "hands": [],
            }
        )
    builder = DatasetBuilder(tmp_path, minimum_episode_frames=1)
    candidate = builder.candidate(SESSION_ID, run.name, annotation_revision_id="a" * 32)
    issue = next(
        item
        for item in candidate.quality.soft_issues
        if item.issue_id == "hand_coverage_low"
    )
    reviewed = DatasetQualityChecker.restore_soft_issue(
        candidate.quality,
        issue.issue_id,
        operator="reviewer",
        reason="the clip intentionally contains a hands-free setup interval",
        restored_at_unix_ns=123,
    )
    rebuilt = builder.candidate(
        SESSION_ID,
        run.name,
        annotation_revision_id="a" * 32,
        quality_report=reviewed,
    )
    assert rebuilt.publishable
    assert rebuilt.quality.soft_issues[0].restored_at_unix_ns == 123

    tampered = reviewed.__class__(
        reviewed.session_id,
        reviewed.run_id,
        reviewed.hard_issues,
        reviewed.soft_issues,
        (*reviewed.metrics, ("tampered", 1.0)),
    )
    with pytest.raises(DatasetBuildError, match="hard gates"):
        builder.candidate(
            SESSION_ID,
            run.name,
            annotation_revision_id="a" * 32,
            quality_report=tampered,
        )


def test_dataset_publish_is_traceable_immutable_and_session_grouped(tmp_path: Path) -> None:
    session, timings = _recorded_session(tmp_path)
    _add_strict_frame_evidence(session, timings)
    run = _write_complete_run(session)
    source_paths = (session / "session.json", session / "media" / f"{CLIP_ID}.mp4")
    before = tuple(_sha256(path) for path in source_paths)

    builder = DatasetBuilder(tmp_path, minimum_episode_frames=1, random_seed=17)
    candidate = builder.candidate(
        SESSION_ID,
        run.name,
        annotation_revision_id="a" * 32,
    )
    assert candidate.publishable
    assert DatasetCatalogService(tmp_path).scan()[0].annotation_revision_id == "a" * 32
    result = builder.publish("dataset-v1", (candidate,))

    assert tuple(_sha256(path) for path in source_paths) == before
    assert result.episode_count == 1
    assert result.sample_count == 2
    assert not list(result.output_directory.rglob("*.mp4"))
    samples = [
        json.loads(line)
        for line in (result.output_directory / "samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {sample["session_id"] for sample in samples} == {SESSION_ID}
    assert all(sample["source_sha256"] == before[1] for sample in samples)
    assert all(
        sample["hand_result_reference"] == "results.sqlite#frame_results"
        for sample in samples
    )
    assert all(
        sample["object_result_reference"] == "objects/object-result.json"
        for sample in samples
    )
    assert all(
        sample["objects"][0]["mask"]["mask_relative_path"].startswith("masks/obj1-")
        for sample in samples
    )
    assert all(sample["objects"][0]["track_visibility"] == 1.0 for sample in samples)
    assert all(sample["objects"][0]["pose"]["object_id"] == "obj1" for sample in samples)
    episodes = [
        json.loads(line)
        for line in (result.output_directory / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert episodes[0]["source_episode_id"] == "c" * 32
    assert episodes[0]["labels"]["episode"]["object"] == "cup"
    split_membership = {
        session_id: split.name
        for split in result.manifest.splits
        for session_id in split.session_ids
    }
    assert split_membership == {SESSION_ID: "train"}
    for artifact in (
        result.manifest.episodes_artifact,
        result.manifest.samples_artifact,
        result.manifest.quality_report_artifact,
        result.manifest.provenance_artifact,
        result.manifest.splits_artifact,
    ):
        assert _sha256(result.output_directory / artifact.relative_path) == artifact.sha256
    provenance = json.loads(
        (result.output_directory / "provenance.json").read_text(encoding="utf-8")
    )
    provenance_paths = {
        artifact["relative_path"]
        for session_provenance in provenance["sessions"]
        for artifact in session_provenance["artifacts"]
    }
    assert "objects/selected-keypoints.jsonl" in provenance_paths
    assert "objects/masks/obj1-000000.png" in provenance_paths
    annotation = provenance["sessions"][0]["annotation"]
    assert annotation["relative_path"].endswith(f"revisions/{'a' * 32}.json")
    assert annotation["content_sha256"] == "e" * 64


def test_partial_and_unverified_vio_runs_cannot_publish(tmp_path: Path) -> None:
    session, timings = _recorded_session(tmp_path)
    _add_strict_frame_evidence(session, timings)
    run = _write_complete_run(session)
    manifest_path = run / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["state"] = "partial"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    quality = DatasetQualityChecker().check(session, run)
    assert not quality.publishable
    assert "processing_not_completed" in {issue.issue_id for issue in quality.hard_issues}

    manifest["state"] = "completed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    vio_path = session / "derived" / "vio" / "basalt" / "vio-run" / "run.json"
    vio = json.loads(vio_path.read_text(encoding="utf-8"))
    vio["calibration_verified"] = False
    vio_path.write_text(json.dumps(vio), encoding="utf-8")
    quality = DatasetQualityChecker().check(session, run)
    assert "vio_calibration_unverified" in {issue.issue_id for issue in quality.hard_issues}


def test_annotation_revision_must_cover_every_dataset_episode(tmp_path: Path) -> None:
    session, timings = _recorded_session(tmp_path)
    _add_strict_frame_evidence(session, timings)
    run = _write_complete_run(session)
    annotation_path = (
        session
        / "annotations"
        / "episode-annotation-v1"
        / "revisions"
        / f"{'a' * 32}.json"
    )
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotation["episodes"][0]["end_exclusive"]["frame_index"] = 1
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")

    with pytest.raises(DatasetBuildError, match="does not cover"):
        DatasetBuilder(tmp_path, minimum_episode_frames=1).candidate(
            SESSION_ID,
            run.name,
            annotation_revision_id="a" * 32,
        )


def test_missing_or_corrupt_object_evidence_is_a_hard_publication_gate(
    tmp_path: Path,
) -> None:
    session, timings = _recorded_session(tmp_path)
    _add_strict_frame_evidence(session, timings)
    run = _write_complete_run(session)
    result_path = run / "objects" / "object-result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["triangulations"] = []
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    quality = DatasetQualityChecker().check(session, run)

    assert "object_triangulation_missing" in {
        issue.issue_id for issue in quality.hard_issues
    }

    result_path.write_text("{broken", encoding="utf-8")
    quality = DatasetQualityChecker().check(session, run)
    assert "object_stage_incomplete" in {issue.issue_id for issue in quality.hard_issues}


def test_dataset_view_uses_fluent_table_and_does_not_block_scan(
    qt_application: object,
    tmp_path: Path,
) -> None:
    runtime = SimpleNamespace(config=SimpleNamespace(recordings_root=tmp_path))
    view = DatasetView(runtime)  # type: ignore[arg-type]
    try:
        view.refresh()
        assert isinstance(view.table, TableWidget)
        assert isinstance(view.wizard.issue_table, TableWidget)
        assert view.table.columnCount() == 12
        assert view._scan_future is not None
        source = Path(view.__class__.__module__.replace(".", "/") + ".py")
        assert "QLabel" not in (Path(__file__).parents[1] / source).read_text(encoding="utf-8")
    finally:
        view.close_resources()


@pytest.mark.parametrize("size", ((1280, 800), (1440, 900), (1920, 1080)))
def test_dataset_view_layout_stays_inside_content_bounds(
    qt_application: object,
    tmp_path: Path,
    size: tuple[int, int],
) -> None:
    runtime = SimpleNamespace(config=SimpleNamespace(recordings_root=tmp_path))
    view = DatasetView(runtime)  # type: ignore[arg-type]
    try:
        view.resize(*size)
        view.show()
        qt_application.processEvents()
        content = view.rect()
        for child in (view.table, view.wizard):
            geometry = child.geometry()
            assert geometry.left() >= content.left()
            assert geometry.top() >= content.top()
            assert geometry.right() <= content.right()
            assert geometry.bottom() <= content.bottom()
        assert view.table.width() == view.wizard.width()
    finally:
        view.close_resources()


def test_dataset_hall_filters_run_provenance_and_quality(
    qt_application: object,
    tmp_path: Path,
) -> None:
    runtime = SimpleNamespace(config=SimpleNamespace(recordings_root=tmp_path))
    view = DatasetView(runtime)  # type: ignore[arg-type]
    rows = (
        DatasetCandidateSummary(
            "session-a",
            "clip-a",
            "run-ready",
            "completed",
            "verified",
            0.9,
            0.8,
            0.1,
            4,
            "ready",
            "annotation-a",
            True,
            "profile-a",
            "calibration-a",
            1.0,
        ),
        DatasetCandidateSummary(
            "session-b",
            "clip-b",
            "run-blocked",
            "partial",
            "unverified",
            0.4,
            0.0,
            0.7,
            0,
            "blocked",
            None,
            False,
            "profile-b",
            "calibration-b",
            0.4,
        ),
    )
    try:
        view._set_rows(rows)
        view.quality_filter.setCurrentIndex(view.quality_filter.findData("ready"))
        assert view._rows == rows[:1]
        view.quality_filter.setCurrentIndex(0)
        view.vio_coverage_filter.setValue(90)
        view.object_coverage_filter.setValue(50)
        assert view._rows == rows[:1]
        view.run_filter.setText("blocked")
        assert view._rows == ()
    finally:
        view.close_resources()


def _write_complete_run(session: Path) -> Path:
    run = session / "derived" / "video-processing" / "run-dataset"
    run.mkdir(parents=True)
    store = ProcessingResultStore(run / "results.sqlite")
    hand = {
        "handedness": "right",
        "confidence": 0.9,
        "final_confidence": 0.9,
        "bbox_xyxy_px": [0.0, 0.0, 2.0, 2.0],
        "keypoints_3d_camera_m": [[0.0, 0.0, 1.0] for _ in range(21)],
        "grasp_ratio": 0.8,
        "is_grasping": True,
        "temporal": {"source": "observed"},
        "kinematics": {"keypoints_3d_world_m": [[0.0, 0.0, 1.0] for _ in range(21)]},
    }
    for frame_index in range(2):
        store.put_final(
            {
                "session_id": SESSION_ID,
                "sequence_id": CLIP_ID,
                "frame_index": frame_index,
                "session_time_ns": 1_000_000_000 + frame_index * 33_333_333,
                "hands": [hand],
            }
        )
    (run / "run.json").write_text(
        json.dumps(
            {
                "run_id": run.name,
                "session_id": SESSION_ID,
                "clip_id": CLIP_ID,
                "state": "completed",
                "vio_run_id": "vio-run",
                "task_profile_id": "default-manipulation",
                "object_tracking": {"state": "completed"},
                "temporal_processing": {"vio_coverage_ratio": 1.0},
                "configuration": {"revision": 7, "sha256_by_file": {}, "snapshot": {}},
            }
        ),
        encoding="utf-8",
    )
    vio = session / "derived" / "vio" / "basalt" / "vio-run"
    vio.mkdir(parents=True)
    (vio / "run.json").write_text(
        json.dumps(
            {
                "run_id": "vio-run",
                "session_id": SESSION_ID,
                "clip_id": CLIP_ID,
                "state": "completed",
                "calibration_verified": True,
                "trajectory_pose_count": 2,
            }
        ),
        encoding="utf-8",
    )
    objects = run / "objects"
    objects.mkdir()
    masks = objects / "masks"
    masks.mkdir()
    for frame_index in range(2):
        (masks / f"obj1-{frame_index:06d}.png").write_bytes(b"synthetic-mask")
    pose = {
        "object_id": "obj1",
        "clip_id": CLIP_ID,
        "frame_index": 0,
        "session_time_ns": 1_000_000_000,
        "transform_object_to_world": list(_identity()),
        "source": "static_triangulation",
        "grasped_by": None,
        "dynamic": False,
    }
    (objects / "object-result.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "processing_run_id": run.name,
                "task_profile_id": "default-manipulation",
                "masks": [
                    {
                        "object_id": "obj1",
                        "clip_id": CLIP_ID,
                        "frame_index": frame_index,
                        "session_time_ns": 1_000_000_000 + frame_index * 33_333_333,
                        "mask_relative_path": f"masks/obj1-{frame_index:06d}.png",
                        "width_px": 640,
                        "height_px": 480,
                        "boxes": [],
                        "mask_area_ratio": 0.2,
                    }
                    for frame_index in range(2)
                ],
                "tracks": [
                    {
                        "object_id": "obj1",
                        "clip_id": CLIP_ID,
                        "frame_indices": [0, 1],
                        "session_times_ns": [1_000_000_000, 1_033_333_333],
                        "points_xy_px": [
                            [[10.0, 10.0], [20.0, 10.0], [10.0, 20.0]],
                            [[11.0, 10.0], [21.0, 10.0], [11.0, 20.0]],
                        ],
                        "visibility": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    }
                ],
                "triangulations": [
                    {
                        "object_id": "obj1",
                        "points_world_m": [
                            [0.0, 0.0, 1.0],
                            [0.1, 0.0, 1.0],
                            [0.0, 0.1, 1.0],
                        ],
                        "transform_object_to_world": list(_identity()),
                        "transform_object_to_camera": list(_identity()),
                        "valid_point_count": 3,
                        "mean_reprojection_error_px": 0.2,
                        "contributing_frame_count": 2,
                        "orientation_method": "pca1",
                    }
                ],
                "poses": [pose, {**pose, "frame_index": 1, "session_time_ns": 1_033_333_333}],
            }
        ),
        encoding="utf-8",
    )
    (objects / "selected-keypoints.jsonl").write_text("", encoding="utf-8")
    (objects / "tracks.json").write_text("[]\n", encoding="utf-8")
    (objects / "triangulation.json").write_text("[]\n", encoding="utf-8")
    (objects / "object-qa.json").write_text("{}\n", encoding="utf-8")
    phase_frames = [
        {
            "clip_id": CLIP_ID,
            "frame_index": index,
            "session_time_ns": 1_000_000_000 + index * 33_333_333,
            "phase": "manipulation",
            "confidence": 0.9,
            "head_linear_speed_m_s": 0.0,
            "head_angular_speed_rad_s": 0.0,
            "hand_linear_speed_m_s": 0.1,
            "grasping": True,
        }
        for index in range(2)
    ]
    (run / "phase-analysis.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "processing_run_id": run.name,
                "frames": phase_frames,
                "segments": [
                    {
                        "clip_id": CLIP_ID,
                        "start_frame_index": 0,
                        "end_frame_index_exclusive": 2,
                        "start_session_time_ns": 1_000_000_000,
                        "end_session_time_ns": 1_033_333_333,
                        "phase": "manipulation",
                        "confidence": 0.9,
                    }
                ],
                "object_centric_windows": [],
            }
        ),
        encoding="utf-8",
    )
    _write_annotation_revision(session)
    return run


def _write_annotation_revision(session: Path) -> None:
    revision_id = "a" * 32
    root = session / "annotations" / "episode-annotation-v1"
    revisions = root / "revisions"
    revisions.mkdir(parents=True)
    payload = {
        "schema_version": "1.0",
        "contract_id": "episode-annotation-v1",
        "annotation_revision_id": revision_id,
        "parent_draft_revision": 1,
        "session_id": SESSION_ID,
        "taxonomy_version": "egocentric-manipulation-v1",
        "segmentation_strategy": "manual",
        "published_at_unix_ns": 2_000_000_000,
        "content_sha256": "e" * 64,
        "episodes": [
            {
                "episode_id": "c" * 32,
                "clip_id": CLIP_ID,
                "start": {
                    "frame_index": 0,
                    "mp4_pts": 0,
                    "mp4_time_base_numerator": 1,
                    "mp4_time_base_denominator": 30,
                    "session_time_ns": 1_000_000_000,
                    "timing_status": "exact",
                },
                "end_exclusive": {
                    "frame_index": 2,
                    "mp4_pts": 2,
                    "mp4_time_base_numerator": 1,
                    "mp4_time_base_denominator": 30,
                    "session_time_ns": 1_066_666_666,
                    "timing_status": "exact",
                },
                "source_strategy": "manual",
                "labels": {
                    "task_id": "test-manipulation",
                    "instruction": "pick up the cup",
                    "verb": "pick",
                    "object": "cup",
                    "hand": "right",
                    "outcome": "success",
                },
                "phases": [
                    {
                        "phase_id": "d" * 32,
                        "start_frame_index": 0,
                        "end_frame_index_exclusive": 2,
                        "phase": "manipulate",
                        "active_hand": "right",
                        "object": "cup",
                    }
                ],
            }
        ],
        "quality": {
            "status": "pass",
            "episode_count": 1,
            "phase_count": 1,
            "checks": [
                {
                    "check_id": "fixture",
                    "status": "pass",
                    "evidence": "synthetic annotation fixture",
                }
            ],
        },
    }
    (revisions / f"{revision_id}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    (root / "latest.json").write_text(
        json.dumps({"annotation_revision_id": revision_id}),
        encoding="utf-8",
    )


def _identity() -> tuple[float, ...]:
    return (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
