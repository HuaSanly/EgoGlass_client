"""Immutable, frame-addressable dataset contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .object_tracking import ObjectMaskObservation, ObjectPose
from .phase import MotionPhase


class QualitySeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class QualityGate(StrEnum):
    HARD = "hard"
    SOFT = "soft"


class ArtifactReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    media_type: str = Field(min_length=1)


class VirtualVideoSpan(BaseModel):
    """A non-mutating frame interval in an immutable recorded clip."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    session_id: str = Field(min_length=1)
    clip_id: str = Field(min_length=1)
    start_frame_index: int = Field(ge=0)
    end_frame_index_exclusive: int = Field(ge=1)
    start_session_time_ns: int = Field(ge=0)
    end_session_time_ns: int = Field(ge=0)
    media: ArtifactReference

    @model_validator(mode="after")
    def validate_span(self) -> VirtualVideoSpan:
        if self.end_frame_index_exclusive <= self.start_frame_index:
            raise ValueError("virtual span must contain at least one frame")
        if self.end_session_time_ns < self.start_session_time_ns:
            raise ValueError("virtual span timestamps must be ordered")
        return self


class SensorSampleReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    telemetry_relative_path: str = Field(min_length=1)
    session_time_ns: int = Field(ge=0)
    imu_window_start_ns: int = Field(ge=0)
    imu_window_end_ns: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_window(self) -> SensorSampleReference:
        if self.imu_window_end_ns < self.imu_window_start_ns:
            raise ValueError("IMU window must be ordered")
        return self


class HandSample(BaseModel):
    """Structured final hand result embedded as JSON-safe data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    handedness: str = Field(min_length=1)
    final_confidence: float = Field(ge=0.0, le=1.0)
    grasp_ratio: float = Field(ge=0.0)
    is_grasping: bool
    keypoints_3d_camera_m: tuple[tuple[float, float, float], ...] = Field(
        min_length=21, max_length=21
    )
    keypoints_3d_world_m: tuple[tuple[float, float, float], ...] | None = None
    temporal_source: str | None = None


class ObjectObservation(BaseModel):
    """Frame-local object evidence referenced by a published sample."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    object_id: str = Field(min_length=1)
    mask: ObjectMaskObservation | None = None
    pose: ObjectPose | None = None
    track_visibility: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str = Field(min_length=1)


class QualityIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    issue_id: str = Field(min_length=1)
    gate: QualityGate
    severity: QualitySeverity
    message: str = Field(min_length=1)
    clip_id: str = Field(min_length=1)
    start_frame_index: int = Field(ge=0)
    end_frame_index_exclusive: int = Field(ge=1)
    restorable: bool = False
    restored: bool = False
    restored_by: str | None = None
    restored_at_unix_ns: int | None = Field(default=None, ge=0)
    restore_reason: str | None = None

    @model_validator(mode="after")
    def validate_restore(self) -> QualityIssue:
        if self.end_frame_index_exclusive <= self.start_frame_index:
            raise ValueError("quality issue interval must be non-empty")
        if self.gate is QualityGate.HARD and self.restorable:
            raise ValueError("hard quality issues cannot be restorable")
        if self.restored and (not self.restorable or not self.restore_reason):
            raise ValueError("restored issue requires a restorable gate and a reason")
        return self


class DatasetFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str = Field(min_length=1)
    clip_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    frame_index: int = Field(ge=0)
    session_time_ns: int = Field(ge=0)
    rgb_reference: ArtifactReference
    sensor: SensorSampleReference
    processing_run_id: str = Field(min_length=1)
    vio_run_id: str = Field(min_length=1)
    hand_result_reference: str = Field(min_length=1)
    object_result_reference: str = Field(min_length=1)
    annotation_revision_id: str = Field(min_length=1)
    configuration_revision: int = Field(ge=0)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    phase: MotionPhase
    hands: tuple[HandSample, ...]
    objects: tuple[ObjectObservation, ...]
    quality_state: str = Field(min_length=1)


class DatasetEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str = Field(min_length=1)
    source_episode_id: str | None = None
    span: VirtualVideoSpan
    processing_run_id: str = Field(min_length=1)
    vio_run_id: str = Field(min_length=1)
    annotation_revision_id: str = Field(min_length=1)
    labels: dict[str, object] = Field(default_factory=dict)
    phase_summary: tuple[MotionPhase, ...] = ()
    quality_issues: tuple[QualityIssue, ...] = ()


class DatasetSplit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(pattern=r"^(train|validation|test)$")
    session_ids: tuple[str, ...]
    episode_ids: tuple[str, ...]


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    dataset_id: str = Field(min_length=1)
    created_at_unix_ns: int = Field(ge=0)
    random_seed: int = Field(ge=0)
    episodes_artifact: ArtifactReference
    samples_artifact: ArtifactReference
    quality_report_artifact: ArtifactReference
    provenance_artifact: ArtifactReference
    splits_artifact: ArtifactReference
    splits: tuple[DatasetSplit, ...]
    source_session_ids: tuple[str, ...]
