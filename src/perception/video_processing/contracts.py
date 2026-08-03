from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ProcessingJobState(StrEnum):
    QUEUED = "queued"
    PREPARING = "preparing"
    RUNNING = "running"
    CANCELING = "canceling"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELED = "canceled"


ACTIVE_JOB_STATES = frozenset(
    {
        ProcessingJobState.PREPARING,
        ProcessingJobState.RUNNING,
        ProcessingJobState.CANCELING,
    }
)
TERMINAL_JOB_STATES = frozenset(
    {
        ProcessingJobState.COMPLETED,
        ProcessingJobState.FAILED,
        ProcessingJobState.INTERRUPTED,
        ProcessingJobState.CANCELED,
    }
)


@dataclass(frozen=True, slots=True)
class ProcessingPreset:
    preset_id: str = "hand-tracking-quality"
    display_name: str = "手部追踪 · 质量优先"
    inference_stride_frames: int = 1

    def __post_init__(self) -> None:
        if not self.preset_id.strip() or not self.display_name.strip():
            raise ValueError("processing preset names cannot be empty")
        if self.inference_stride_frames < 1:
            raise ValueError("inference stride must be positive")


DEFAULT_PRESETS = (
    ProcessingPreset(),
    ProcessingPreset("hand-tracking-balanced", "手部追踪 · 均衡", 3),
    ProcessingPreset("hand-tracking-preview", "手部追踪 · 快速预览", 5),
)


@dataclass(frozen=True, slots=True)
class ProcessingJob:
    job_id: str
    session_id: str
    clip_id: str | None
    preset: ProcessingPreset
    state: ProcessingJobState
    created_at_unix_ns: int
    updated_at_unix_ns: int
    progress_current: int = 0
    progress_total: int = 0
    detail: str = ""
    run_id: str | None = None
    retry_of_job_id: str | None = None
    started_at_unix_ns: int | None = None
    finished_at_unix_ns: int | None = None

    @property
    def progress_fraction(self) -> float:
        if self.progress_total <= 0:
            return 0.0
        return min(1.0, self.progress_current / self.progress_total)

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at_unix_ns is None:
            return 0.0
        end_ns = self.finished_at_unix_ns or self.updated_at_unix_ns
        return max(0.0, (end_ns - self.started_at_unix_ns) / 1_000_000_000)


@dataclass(frozen=True, slots=True)
class ProcessingRunSummary:
    run_id: str
    session_id: str
    clip_id: str | None
    output_directory: Path
    input_frame_count: int
    inferred_frame_count: int
    detected_hand_count: int
    started_at_unix_ns: int
    completed_at_unix_ns: int


@dataclass(frozen=True, slots=True)
class ProcessingServiceSnapshot:
    revision: int
    active_job_id: str | None
    auto_enqueue_on_session_complete: bool
    jobs: tuple[ProcessingJob, ...]
