from __future__ import annotations

import json
from pathlib import Path

from .models import DatasetCandidateSummary
from .quality import DatasetQualityChecker


class DatasetCatalogService:
    """Discover offline run candidates without importing Qt."""

    def __init__(
        self,
        recordings_root: str | Path,
        *,
        quality_checker: DatasetQualityChecker | None = None,
    ) -> None:
        self.recordings_root = Path(recordings_root).expanduser().resolve()
        self.quality_checker = quality_checker or DatasetQualityChecker()

    def scan(self) -> tuple[DatasetCandidateSummary, ...]:
        rows: list[DatasetCandidateSummary] = []
        if not self.recordings_root.is_dir():
            return ()
        for session_path in sorted(self.recordings_root.iterdir()):
            if not session_path.is_dir() or not (session_path / "session.json").is_file():
                continue
            run_root = session_path / "derived" / "video-processing"
            if not run_root.is_dir():
                continue
            annotation_id = _annotation_revision(session_path)
            for run_directory in sorted(run_root.iterdir(), reverse=True):
                run = _json(run_directory / "run.json")
                if run is None:
                    continue
                quality = self.quality_checker.check(session_path, run_directory)
                metrics = dict(quality.metrics)
                phase = _json(run_directory / "phase-analysis.json") or {}
                phase_count = len(phase.get("segments", ()))
                vio = _vio_manifest(session_path, run.get("vio_run_id"))
                rows.append(
                    DatasetCandidateSummary(
                        session_id=session_path.name,
                        clip_id=(
                            str(run["clip_id"]) if isinstance(run.get("clip_id"), str) else None
                        ),
                        run_id=run_directory.name,
                        processing_state=str(run.get("state", "invalid")),
                        vio_state=_vio_state(vio),
                        hand_coverage=metrics.get("hand_coverage", 0.0),
                        object_coverage=metrics.get("object_triangulation_coverage", 0.0),
                        interpolation_ratio=metrics.get("interpolation_ratio", 0.0),
                        phase_count=phase_count,
                        quality_state=(
                            "ready"
                            if quality.publishable
                            else ("review" if not quality.hard_issues else "blocked")
                        ),
                        annotation_revision_id=annotation_id,
                        candidate=quality.publishable and annotation_id is not None,
                        task_profile_id=(
                            str(run["task_profile_id"])
                            if isinstance(run.get("task_profile_id"), str)
                            else None
                        ),
                        calibration_profile_id=(
                            str(vio["calibration_profile_id"])
                            if isinstance(vio, dict)
                            and isinstance(vio.get("calibration_profile_id"), str)
                            else None
                        ),
                        vio_coverage=metrics.get("vio_coverage", 0.0),
                    )
                )
        return tuple(rows)


def _json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _annotation_revision(session_path: Path) -> str | None:
    latest = _json(
        session_path / "annotations" / "episode-annotation-v1" / "latest.json"
    )
    value = latest.get("annotation_revision_id") if latest else None
    return str(value) if isinstance(value, str) and value else None


def _vio_manifest(session_path: Path, run_id: object) -> dict[str, object] | None:
    if not isinstance(run_id, str) or not run_id:
        return None
    return _json(session_path / "derived" / "vio" / "basalt" / run_id / "run.json")


def _vio_state(payload: dict[str, object] | None) -> str:
    if payload is None:
        return "missing"
    if payload.get("state") != "completed":
        return str(payload.get("state", "invalid"))
    return "verified" if payload.get("calibration_verified") is True else "unverified"
