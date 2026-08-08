"""Input and configuration types for phase analysis."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PhaseAnalysisConfig(BaseModel):
    """Conservative thresholds for offline HumanEgo-style phase proposals."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    maximum_vio_pose_gap_ms: int = Field(default=100, ge=1, le=10_000)
    stop_linear_speed_m_s: float = Field(default=0.08, ge=0.0, le=10.0)
    forward_linear_speed_m_s: float = Field(default=0.15, ge=0.0, le=10.0)
    rotate_angular_speed_rad_s: float = Field(default=0.40, ge=0.0, le=20.0)
    manipulation_hand_speed_m_s: float = Field(default=0.04, ge=0.0, le=10.0)
    minimum_segment_frames: int = Field(default=5, ge=1, le=300)
    finished_trailing_frames: int = Field(default=15, ge=0, le=3000)
    precontact_window_frames: int = Field(default=30, ge=0, le=3000)
    minimum_object_window_frames: int = Field(default=30, ge=2, le=3000)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> PhaseAnalysisConfig:
        if self.forward_linear_speed_m_s < self.stop_linear_speed_m_s:
            raise ValueError("forward speed must be at least the stop speed")
        return self


@dataclass(frozen=True, slots=True)
class PhaseInputFrame:
    """One frame with VIO-derived head pose and finalized hand state."""

    clip_id: str
    frame_index: int
    session_time_ns: int
    head_position_m: tuple[float, float, float]
    head_quaternion_wxyz: tuple[float, float, float, float]
    hand_linear_speed_m_s: float
    grasping: bool

    def __post_init__(self) -> None:
        if not self.clip_id:
            raise ValueError("clip_id cannot be empty")
        if self.frame_index < 0 or self.session_time_ns < 0:
            raise ValueError("frame identity must be non-negative")
        if self.hand_linear_speed_m_s < 0.0:
            raise ValueError("hand speed cannot be negative")
