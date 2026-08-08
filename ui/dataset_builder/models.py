from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from schemas import DatasetEpisode, DatasetManifest, QualityIssue


@dataclass(frozen=True, slots=True)
class DatasetQualityReport:
    """Deterministic quality decision for one processing run."""

    session_id: str
    run_id: str
    hard_issues: tuple[QualityIssue, ...] = ()
    soft_issues: tuple[QualityIssue, ...] = ()
    metrics: tuple[tuple[str, float], ...] = ()

    @property
    def publishable(self) -> bool:
        return not self.hard_issues and all(issue.restored for issue in self.soft_issues)

    @property
    def issue_count(self) -> int:
        return len(self.hard_issues) + len(self.soft_issues)

    def all_issues(self) -> tuple[QualityIssue, ...]:
        return self.hard_issues + self.soft_issues


@dataclass(frozen=True, slots=True)
class DatasetCandidate:
    session_id: str
    run_id: str
    session_path: Path
    run_directory: Path
    quality: DatasetQualityReport
    episodes: tuple[DatasetEpisode, ...]
    annotation_revision_id: str

    @property
    def publishable(self) -> bool:
        return self.quality.publishable and bool(self.episodes)


@dataclass(frozen=True, slots=True)
class DatasetBuildResult:
    dataset_id: str
    output_directory: Path
    manifest: DatasetManifest
    episode_count: int
    sample_count: int


@dataclass(frozen=True, slots=True)
class DatasetCandidateSummary:
    session_id: str
    clip_id: str | None
    run_id: str
    processing_state: str
    vio_state: str
    hand_coverage: float
    object_coverage: float
    interpolation_ratio: float
    phase_count: int
    quality_state: str
    annotation_revision_id: str | None
    candidate: bool
    task_profile_id: str | None = None
    calibration_profile_id: str | None = None
    vio_coverage: float = 0.0
