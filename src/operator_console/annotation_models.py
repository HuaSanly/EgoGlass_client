from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ID_PATTERN = r"^[0-9a-f]{32}$"


def _normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


class SegmentationStrategy(StrEnum):
    MANUAL = "manual"
    CLIP_AS_EPISODE = "clip_as_episode"
    FIXED_WINDOW = "fixed_window"
    EVENT_MARKER = "event_marker"
    MOTION_CHANGE = "motion_change"
    HAND_OBJECT_INTERACTION = "hand_object_interaction"
    VLM_SEMANTIC = "vlm_semantic"


class HandLabel(StrEnum):
    UNSPECIFIED = "unspecified"
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"
    NONE = "none"


class EpisodeOutcome(StrEnum):
    UNREVIEWED = "unreviewed"
    SUCCESS = "success"
    FAILURE = "failure"
    INTERRUPTED = "interrupted"
    INVALID = "invalid"


class PhaseKind(StrEnum):
    PREPARE = "prepare"
    APPROACH = "approach"
    CONTACT = "contact"
    MANIPULATE = "manipulate"
    RELEASE = "release"
    COMPLETE = "complete"
    OTHER = "other"


class QualityFlag(StrEnum):
    BLURRED = "blurred"
    OCCLUDED = "occluded"
    EXCESSIVE_CAMERA_MOTION = "excessive_camera_motion"
    INCOMPLETE_ACTION = "incomplete_action"
    METADATA_GAP = "metadata_gap"
    OTHER = "other"


class EpisodeLabels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str | None = Field(default=None, max_length=128)
    instruction: str = Field(default="", max_length=512)
    verb: str = Field(default="", max_length=128)
    object: str = Field(default="", max_length=128)
    target: str | None = Field(default=None, max_length=128)
    hand: HandLabel = HandLabel.UNSPECIFIED
    outcome: EpisodeOutcome = EpisodeOutcome.UNREVIEWED
    quality_flags: list[QualityFlag] = Field(default_factory=list, max_length=6)
    notes: str = Field(default="", max_length=2000)

    @field_validator("task_id", "target")
    @classmethod
    def normalize_optional(cls, value: str | None) -> str | None:
        return _normalize_optional(value)

    @field_validator("instruction", "verb", "object", "notes")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("quality_flags")
    @classmethod
    def unique_quality_flags(cls, value: list[QualityFlag]) -> list[QualityFlag]:
        if len(set(value)) != len(value):
            raise ValueError("quality_flags must be unique")
        return value


class PhaseDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase_id: str = Field(pattern=ID_PATTERN)
    start_frame_index: int = Field(ge=0)
    end_frame_index_exclusive: int = Field(ge=1)
    phase: PhaseKind
    action_verb: str | None = Field(default=None, max_length=128)
    active_hand: HandLabel = HandLabel.UNSPECIFIED
    object: str | None = Field(default=None, max_length=128)

    @field_validator("action_verb", "object")
    @classmethod
    def normalize_optional(cls, value: str | None) -> str | None:
        return _normalize_optional(value)

    @model_validator(mode="after")
    def ordered_interval(self) -> PhaseDraft:
        if self.start_frame_index >= self.end_frame_index_exclusive:
            raise ValueError("phase interval must be non-empty")
        return self


class EpisodeDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(pattern=ID_PATTERN)
    clip_id: str = Field(pattern=ID_PATTERN)
    start_frame_index: int = Field(ge=0)
    end_frame_index_exclusive: int = Field(ge=1)
    source_strategy: SegmentationStrategy = SegmentationStrategy.MANUAL
    labels: EpisodeLabels = Field(default_factory=EpisodeLabels)
    phases: list[PhaseDraft] = Field(default_factory=list)

    @model_validator(mode="after")
    def ordered_interval(self) -> EpisodeDraft:
        if self.start_frame_index >= self.end_frame_index_exclusive:
            raise ValueError("episode interval must be non-empty")
        return self


class AnnotationDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    contract_id: Literal["episode-annotation-draft-v1"] = "episode-annotation-draft-v1"
    session_id: str = Field(pattern=ID_PATTERN)
    taxonomy_version: Literal["egocentric-manipulation-v1"] = (
        "egocentric-manipulation-v1"
    )
    draft_revision: int = Field(ge=0)
    segmentation_strategy: SegmentationStrategy = SegmentationStrategy.MANUAL
    default_labels: EpisodeLabels = Field(default_factory=EpisodeLabels)
    episodes: list[EpisodeDraft] = Field(default_factory=list)
    updated_at_unix_ns: int | None = Field(default=None, ge=0)
    latest_published_revision_id: str | None = Field(default=None, pattern=ID_PATTERN)


class SaveDraftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_revision: int = Field(ge=0)
    segmentation_strategy: SegmentationStrategy
    default_labels: EpisodeLabels = Field(default_factory=EpisodeLabels)
    episodes: list[EpisodeDraft]


class PublishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    base_revision: int = Field(ge=1)


class ProposalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    strategy: Literal["clip_as_episode", "fixed_window"]
    clip_id: str | None = Field(default=None, pattern=ID_PATTERN)
    window_duration_ms: int = Field(default=8000, ge=1000, le=120000)
    stride_duration_ms: int = Field(default=8000, ge=250, le=120000)

    @model_validator(mode="after")
    def non_overlapping_windows(self) -> ProposalRequest:
        if (
            self.strategy == "fixed_window"
            and self.stride_duration_ms < self.window_duration_ms
        ):
            raise ValueError("episode window stride cannot be shorter than its duration")
        return self


class EpisodeProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(pattern=ID_PATTERN)
    clip_id: str = Field(pattern=ID_PATTERN)
    start_frame_index: int = Field(ge=0)
    end_frame_index_exclusive: int = Field(ge=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: list[str] = Field(min_length=1)


class ProposalBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    contract_id: Literal["episode-proposal-v1"] = "episode-proposal-v1"
    proposal_batch_id: str = Field(pattern=ID_PATTERN)
    session_id: str = Field(pattern=ID_PATTERN)
    strategy: Literal["clip_as_episode", "fixed_window"]
    generated_at_unix_ns: int = Field(ge=0)
    config: dict[str, int | str | None]
    proposals: list[EpisodeProposal]


class ClipSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_id: str = Field(pattern=ID_PATTERN)
    state: Literal["complete", "incomplete"]
    duration_ms: int = Field(ge=0)
    frame_count: int = Field(ge=1)
    fps: float = Field(gt=0, le=240)
    width: int = Field(gt=0, le=8192)
    height: int = Field(gt=0, le=8192)
    media_url: str
    exact_frame_index_available: bool


class AnnotationSessionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(pattern=ID_PATTERN)
    display_name: str = Field(min_length=1, max_length=128)
    state: Literal["active", "finalizing", "complete", "incomplete"]
    started_at_unix_ms: int = Field(ge=0)
    editable: bool
    annotation_status: Literal["unannotated", "draft", "published"]
    draft_revision: int = Field(ge=0)
    latest_published_revision_id: str | None = Field(default=None, pattern=ID_PATTERN)
    clips: list[ClipSummary]


class AnnotationWorkspace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    taxonomy_version: Literal["egocentric-manipulation-v1"] = (
        "egocentric-manipulation-v1"
    )
    project_name: Literal["EgoGlass 人类操作"] = "EgoGlass 人类操作"
    implemented_strategies: list[Literal["manual", "clip_as_episode", "fixed_window"]]
    planned_strategies: list[
        Literal[
            "event_marker",
            "motion_change",
            "hand_object_interaction",
            "vlm_semantic",
        ]
    ]
    skipped_session_count: int = Field(ge=0)
    skipped_session_reasons: dict[str, int]
    sessions: list[AnnotationSessionSummary]


class AnnotationSession(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: AnnotationSessionSummary
    draft: AnnotationDraft


class FrameBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_index: int = Field(ge=0)
    mp4_pts: int | None = None
    mp4_time_base_numerator: int | None = Field(default=None, ge=1)
    mp4_time_base_denominator: int | None = Field(default=None, ge=1)
    session_time_ns: int | None = Field(default=None, ge=0)
    timing_status: Literal["exact", "estimated", "unmapped"]


class PublishedEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(pattern=ID_PATTERN)
    clip_id: str = Field(pattern=ID_PATTERN)
    start: FrameBoundary
    end_exclusive: FrameBoundary
    source_strategy: SegmentationStrategy
    labels: EpisodeLabels
    phases: list[PhaseDraft] = Field(min_length=1)


class AnnotationQualityCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str = Field(min_length=1, max_length=128)
    status: Literal["pass", "warn", "fail"]
    evidence: str = Field(min_length=1, max_length=512)


class AnnotationQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["pass"] = "pass"
    episode_count: int = Field(ge=1)
    phase_count: int = Field(ge=1)
    checks: list[AnnotationQualityCheck] = Field(min_length=1)


class PublishedRevision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    contract_id: Literal["episode-annotation-v1"] = "episode-annotation-v1"
    annotation_revision_id: str = Field(pattern=ID_PATTERN)
    parent_draft_revision: int = Field(ge=1)
    session_id: str = Field(pattern=ID_PATTERN)
    taxonomy_version: Literal["egocentric-manipulation-v1"] = (
        "egocentric-manipulation-v1"
    )
    segmentation_strategy: SegmentationStrategy
    published_at_unix_ns: int = Field(ge=0)
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    episodes: list[PublishedEpisode] = Field(min_length=1)
    quality: AnnotationQuality
