from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from pydantic import ValidationError

from schemas import ObjectTrackingResult, QualityGate, QualityIssue, QualitySeverity
from sensor_preprocessing import CaptureSessionReader, CaptureSessionReadError
from ui.processing.results import ProcessingResultStore

from .models import DatasetQualityReport


class DatasetQualityChecker:
    """Run publication gates using only immutable session/run evidence."""

    def __init__(
        self, *, require_object_stage: bool = True, minimum_vio_coverage: float = 0.95
    ) -> None:
        self.require_object_stage = require_object_stage
        self.minimum_vio_coverage = minimum_vio_coverage

    def check(
        self,
        session_path: str | Path,
        run_directory: str | Path,
    ) -> DatasetQualityReport:
        session = Path(session_path).expanduser().resolve()
        run = Path(run_directory).expanduser().resolve()
        run_id = run.name
        session_id = session.name
        hard: list[QualityIssue] = []
        soft: list[QualityIssue] = []
        metrics: dict[str, float] = {}
        try:
            reader = CaptureSessionReader.open(session, verify_media_hashes=True)
            session_id = reader.session.session_id
        except (CaptureSessionReadError, OSError, ValueError) as error:
            hard.append(_issue("session_invalid", "session", str(error), hard=True))
            return DatasetQualityReport(session_id, run_id, tuple(hard), (), ())
        payload = _read_json(run / "run.json")
        if payload is None:
            hard.append(
                _issue(
                    "run_manifest_missing",
                    "session",
                    "processing run manifest is missing",
                    hard=True,
                )
            )
            return DatasetQualityReport(session_id, run_id, tuple(hard), (), ())
        if payload.get("state") != "completed":
            hard.append(
                _issue(
                    "processing_not_completed",
                    "session",
                    "only completed offline runs can be published",
                    hard=True,
                )
            )
        temporal = payload.get("temporal_processing")
        if isinstance(temporal, dict):
            coverage = float(temporal.get("vio_coverage_ratio", 0.0) or 0.0)
            metrics["vio_coverage"] = coverage
            if coverage < self.minimum_vio_coverage:
                hard.append(
                    _issue(
                        "vio_coverage_incomplete",
                        "session",
                        "VIO coverage is below the publication threshold",
                        hard=True,
                    )
                )
        else:
            hard.append(
                _issue(
                    "vio_coverage_missing",
                    "session",
                    "processing run does not record VIO coverage",
                    hard=True,
                )
            )
        results_path = run / "results.sqlite"
        if not results_path.is_file():
            hard.append(
                _issue("results_missing", "session", "results.sqlite is missing", hard=True)
            )
            rows: tuple[dict[str, object], ...] = ()
        else:
            try:
                store = ProcessingResultStore(results_path, read_only=True)
                rows = tuple(store.iter_results())
            except (OSError, sqlite3.DatabaseError, RuntimeError, ValueError) as error:
                hard.append(_issue("results_corrupt", "session", str(error), hard=True))
                rows = ()
        vio_id = payload.get("vio_run_id")
        if not isinstance(vio_id, str) or not vio_id:
            hard.append(
                _issue(
                    "vio_missing", "session", "processing run is not bound to a VIO run", hard=True
                )
            )
        else:
            vio_manifest = _read_json(session / "derived" / "vio" / "basalt" / vio_id / "run.json")
            if vio_manifest is None or vio_manifest.get("state") != "completed":
                hard.append(
                    _issue(
                        "vio_incomplete", "session", "VIO is incomplete or unavailable", hard=True
                    )
                )
            elif vio_manifest.get("calibration_verified") is not True:
                hard.append(
                    _issue(
                        "vio_calibration_unverified",
                        "session",
                        "VIO calibration is not verified",
                        hard=True,
                    )
                )
            else:
                pose_count = float(vio_manifest.get("trajectory_pose_count", 0) or 0)
                metrics["vio_pose_count"] = pose_count
                if pose_count < 2:
                    hard.append(
                        _issue(
                            "vio_trajectory_empty",
                            "session",
                            "VIO trajectory has fewer than two poses",
                            hard=True,
                        )
                    )
        object_result, object_error = _object_result(run / "objects" / "object-result.json")
        if self.require_object_stage:
            object_stage = payload.get("object_tracking")
            if not payload.get("task_profile_id"):
                hard.append(
                    _issue(
                        "object_profile_missing",
                        "session",
                        "processing run has no frozen object task profile",
                        hard=True,
                    )
                )
            elif not isinstance(object_stage, dict) or (
                object_stage.get("state") != "completed" or object_result is None
            ):
                hard.append(
                    _issue(
                        "object_stage_incomplete",
                        "session",
                        object_error or "object processing did not complete",
                        hard=True,
                    )
                )
            else:
                assert object_result is not None
                triangulations = object_result.triangulations
                valid = [item for item in triangulations if item.valid_point_count >= 3]
                metrics["object_triangulation_coverage"] = len(valid) / max(
                    1, len(triangulations)
                )
                if not triangulations:
                    hard.append(
                        _issue(
                            "object_triangulation_missing",
                            "session",
                            "object processing produced no triangulated object",
                            hard=True,
                        )
                    )
                elif len(valid) != len(triangulations):
                    hard.append(
                        _issue(
                            "object_triangulation_invalid",
                            "session",
                            "one or more objects lack enough 3D points",
                            hard=True,
                        )
                    )
                if not object_result.tracks or not object_result.poses:
                    hard.append(
                        _issue(
                            "object_motion_evidence_missing",
                            "session",
                            "object point tracks or propagated poses are missing",
                            hard=True,
                        )
                    )
                object_root = run / "objects"
                valid_masks = sum(
                    1
                    for item in object_result.masks
                    if item.mask_area_ratio > 0.0
                    and _artifact_inside(object_root, item.mask_relative_path)
                )
                metrics["object_mask_coverage"] = valid_masks / max(
                    1, len(object_result.masks)
                )
                visibility = [
                    value
                    for track in object_result.tracks
                    for frame in track.visibility
                    for value in frame
                ]
                metrics["object_track_visibility"] = (
                    sum(visibility) / len(visibility) if visibility else 0.0
                )
                metrics["object_pose_coverage"] = len(object_result.poses) / max(
                    1,
                    sum(len(track.frame_indices) for track in object_result.tracks),
                )
                reprojection_errors = [
                    item.mean_reprojection_error_px for item in triangulations
                ]
                metrics["object_reprojection_error_px"] = (
                    sum(reprojection_errors) / len(reprojection_errors)
                    if reprojection_errors
                    else 0.0
                )
                if metrics["object_mask_coverage"] < 0.7:
                    soft.append(
                        _issue(
                            "object_mask_coverage_low",
                            "session",
                            "fewer than 70% of object masks are valid artifacts",
                            hard=False,
                        )
                    )
                if metrics["object_track_visibility"] < 0.5:
                    soft.append(
                        _issue(
                            "object_track_visibility_low",
                            "session",
                            "mean CoTracker visibility is below 50%",
                            hard=False,
                        )
                    )
                if metrics["object_reprojection_error_px"] > 5.0:
                    soft.append(
                        _issue(
                            "object_reprojection_error_high",
                            "session",
                            "mean object reprojection error exceeds 5 px",
                            hard=False,
                        )
                    )
        by_clip: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            clip_id = row.get("sequence_id")
            if isinstance(clip_id, str):
                by_clip.setdefault(clip_id, []).append(row)
        for clip_id, clip_rows in by_clip.items():
            ordered = sorted(clip_rows, key=lambda item: int(item.get("frame_index", -1)))
            if any(
                int(current.get("frame_index", -1)) <= int(previous.get("frame_index", -1))
                or int(current.get("session_time_ns", -1))
                <= int(previous.get("session_time_ns", -1))
                for previous, current in zip(ordered, ordered[1:], strict=False)
            ):
                hard.append(
                    _issue(
                        "frame_time_not_monotonic",
                        clip_id,
                        "frame index or session_time_ns is not strictly increasing",
                        hard=True,
                    )
                )
        total_rows = sum(len(values) for values in by_clip.values())
        detected_rows = sum(1 for row in rows if row.get("hands"))
        metrics["hand_coverage"] = detected_rows / max(1, total_rows)
        interpolated = sum(
            1
            for row in rows
            for hand in row.get("hands", ())
            if isinstance(row.get("hands"), list)
            if isinstance(hand, dict)
            and isinstance(hand.get("temporal"), dict)
            and hand["temporal"].get("source") == "interpolated"
        )
        hand_count = sum(
            len(row.get("hands", ())) for row in rows if isinstance(row.get("hands"), list)
        )
        metrics["interpolation_ratio"] = interpolated / max(1, hand_count)
        if metrics["hand_coverage"] < 0.5:
            soft.append(
                _issue("hand_coverage_low", "clip", "hand coverage is below 50%", hard=False)
            )
        if metrics["interpolation_ratio"] > 0.5:
            soft.append(
                _issue(
                    "interpolation_high",
                    "clip",
                    "more than half of hand samples are interpolated",
                    hard=False,
                )
            )
        for row in rows:
            hands = row.get("hands")
            if not isinstance(hands, list) or len(hands) < 2:
                continue
            for first_index, first in enumerate(hands):
                for second in hands[first_index + 1 :]:
                    if not isinstance(first, dict) or not isinstance(second, dict):
                        continue
                    if _bbox_iou(first.get("bbox_xyxy_px"), second.get("bbox_xyxy_px")) < 0.8:
                        continue
                    clip_id = str(row.get("sequence_id", "clip"))
                    frame_index = int(row.get("frame_index", 0))
                    soft.append(
                        _issue(
                            f"hand_bbox_iou_{clip_id}_{frame_index}",
                            clip_id,
                            "left/right hand detections overlap with IoU >= 0.8",
                            hard=False,
                            start_frame_index=frame_index,
                            end_frame_index_exclusive=frame_index + 1,
                        )
                    )
        return DatasetQualityReport(
            session_id,
            run_id,
            tuple(hard),
            tuple(soft),
            tuple(sorted(metrics.items())),
        )

    @staticmethod
    def restore_soft_issue(
        report: DatasetQualityReport,
        issue_id: str,
        *,
        operator: str,
        reason: str,
        restored_at_unix_ns: int | None = None,
    ) -> DatasetQualityReport:
        """Record an auditable soft-gate override; hard gates remain immutable."""

        if not operator.strip() or not reason.strip():
            raise ValueError("quality override requires operator and reason")
        found = False
        restored = []
        for issue in report.soft_issues:
            if issue.issue_id != issue_id:
                restored.append(issue)
                continue
            found = True
            restored.append(
                issue.model_copy(
                    update={
                        "restored": True,
                        "restored_by": operator.strip(),
                        "restored_at_unix_ns": restored_at_unix_ns or time.time_ns(),
                        "restore_reason": reason.strip(),
                    }
                )
            )
        if not found:
            raise KeyError(issue_id)
        return DatasetQualityReport(
            report.session_id,
            report.run_id,
            report.hard_issues,
            tuple(restored),
            report.metrics,
        )


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _object_result(path: Path) -> tuple[ObjectTrackingResult | None, str | None]:
    try:
        result = ObjectTrackingResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError, ValueError) as error:
        return None, f"object result is missing or invalid: {error}"
    return result, None


def _artifact_inside(root: Path, relative_path: str) -> bool:
    candidate = (root / relative_path).resolve()
    return candidate.is_relative_to(root.resolve()) and candidate.is_file()


def _issue(
    issue_id: str,
    clip_id: str,
    message: str,
    *,
    hard: bool,
    start_frame_index: int = 0,
    end_frame_index_exclusive: int = 1,
) -> QualityIssue:
    return QualityIssue(
        issue_id=issue_id,
        gate=QualityGate.HARD if hard else QualityGate.SOFT,
        severity=QualitySeverity.ERROR if hard else QualitySeverity.WARNING,
        message=message,
        clip_id=clip_id,
        start_frame_index=start_frame_index,
        end_frame_index_exclusive=end_frame_index_exclusive,
        restorable=not hard,
    )


def _bbox_iou(first: object, second: object) -> float:
    if not isinstance(first, list) or not isinstance(second, list):
        return 0.0
    if len(first) != 4 or len(second) != 4:
        return 0.0
    try:
        ax1, ay1, ax2, ay2 = (float(value) for value in first)
        bx1, by1, bx2, by2 = (float(value) for value in second)
    except (TypeError, ValueError):
        return 0.0
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0
