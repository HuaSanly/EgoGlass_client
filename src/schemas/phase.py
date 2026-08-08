"""Stable phase-analysis contracts shared by offline processing and dataset export."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MotionPhase(StrEnum):
    """HumanEgo-inspired movement labels derived from VIO and hand kinematics."""

    STOP = "stop"
    MANIPULATION = "manipulation"
    FORWARD = "forward"
    ROTATE = "rotate"
    TRANSITION = "transition"
    FINISHED = "finished"


class PhaseFrame(BaseModel):
    """One time-aligned phase decision for a processed video frame."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    clip_id: str = Field(min_length=1)
    frame_index: int = Field(ge=0)
    session_time_ns: int = Field(ge=0)
    phase: MotionPhase
    confidence: float = Field(ge=0.0, le=1.0)
    head_linear_speed_m_s: float = Field(ge=0.0)
    head_angular_speed_rad_s: float = Field(ge=0.0)
    hand_linear_speed_m_s: float = Field(ge=0.0)
    grasping: bool = False


class PhaseSegment(BaseModel):
    """A contiguous phase interval in a single source clip."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    clip_id: str = Field(min_length=1)
    start_frame_index: int = Field(ge=0)
    end_frame_index_exclusive: int = Field(ge=1)
    start_session_time_ns: int = Field(ge=0)
    end_session_time_ns: int = Field(ge=0)
    phase: MotionPhase
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_interval(self) -> PhaseSegment:
        if self.end_frame_index_exclusive <= self.start_frame_index:
            raise ValueError("phase segment frame interval must be non-empty")
        if self.end_session_time_ns < self.start_session_time_ns:
            raise ValueError("phase segment timestamps must be ordered")
        return self


class ObjectCentricWindow(BaseModel):
    """Stable pre-contact to manipulation interval for object reconstruction."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    clip_id: str = Field(min_length=1)
    start_frame_index: int = Field(ge=0)
    reference_frame_index: int = Field(ge=0)
    end_frame_index_exclusive: int = Field(ge=1)
    start_session_time_ns: int = Field(ge=0)
    end_session_time_ns: int = Field(ge=0)
    evidence: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_window(self) -> ObjectCentricWindow:
        if not (
            self.start_frame_index <= self.reference_frame_index < self.end_frame_index_exclusive
        ):
            raise ValueError("reference frame must be inside the object-centric window")
        if self.end_session_time_ns < self.start_session_time_ns:
            raise ValueError("object-centric window timestamps must be ordered")
        return self


class PhaseAnalysisResult(BaseModel):
    """Immutable result of phase classification for one processing run."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str = "1.0"
    processing_run_id: str = Field(min_length=1)
    frames: tuple[PhaseFrame, ...]
    segments: tuple[PhaseSegment, ...]
    object_centric_windows: tuple[ObjectCentricWindow, ...]
