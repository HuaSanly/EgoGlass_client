"""Versioned task profiles and model settings for offline object processing."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ObjectTrackingError(RuntimeError):
    """Object processing cannot start because its profile or model is unavailable."""


class TaskProfile(BaseModel):
    """Task-local object IDs and prompts frozen into every processing run."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    profile_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    display_name: str = Field(min_length=1, max_length=128)
    object_prompts: dict[str, str] = Field(min_length=1)
    arm_prompt: str = Field(default="human arms . human hands .", min_length=1)
    orientation_method: Literal["pca1", "pca2"] = "pca1"

    @model_validator(mode="after")
    def validate_prompts(self) -> TaskProfile:
        for object_id, prompt in self.object_prompts.items():
            if not object_id.startswith("obj"):
                raise ValueError("object IDs must start with 'obj'")
            if not prompt.strip().endswith("."):
                raise ValueError("Grounding DINO prompts must end with a period")
        return self


class ObjectTrackingConfig(BaseModel):
    """Pinned model sources and conservative geometric tracking defaults."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str = "1.0"
    dino_model_id: str = "IDEA-Research/grounding-dino-tiny"
    dino_model_revision: str = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
    sam2_repo_id: str = "facebook/sam2-hiera-tiny"
    sam2_model_revision: str = "7c218beaf0bb87874785f32b582f640134fc1c09"
    sam2_checkpoint_name: str = "sam2_hiera_tiny.pt"
    sam2_config: str = "configs/sam2/sam2_hiera_t.yaml"
    sam2_code_revision: str = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
    cotracker_repository: str = "facebook/cotracker3"
    cotracker_model_revision: str = "bf55ea50d4390e1820a267f131cd6587240fb2c5"
    cotracker_checkpoint_name: str = "scaled_offline.pth"
    cotracker_code_revision: str = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
    device: Literal["cuda", "cpu"] = "cuda"
    require_cuda: bool = True
    box_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    mask_min_area_ratio: float = Field(default=0.001, gt=0.0, le=1.0)
    keypoint_count: int = Field(default=20, ge=3, le=200)
    morphology_close_kernel: int = Field(default=7, ge=1, le=101)
    morphology_erode_kernel: int = Field(default=5, ge=1, le=101)
    inner_edge_kernel: int = Field(default=5, ge=1, le=101)
    cotracker_resolution: int = Field(default=640, ge=128, le=2048)
    cotracker_chunk_size: int = Field(default=100, ge=4, le=2000)
    triangulation_frame_stride: int = Field(default=3, ge=1, le=100)
    triangulation_huber_scale_px: float = Field(default=3.0, gt=0.0, le=100.0)
    maximum_vio_pose_gap_ms: int = Field(default=100, ge=1, le=10_000)
    maximum_grasp_latch_distance_m: float = Field(default=0.35, gt=0.0, le=5.0)
    profiles: tuple[TaskProfile, ...] = ()

    @model_validator(mode="after")
    def validate_shape(self) -> ObjectTrackingConfig:
        if self.morphology_close_kernel % 2 == 0:
            raise ValueError("morphology_close_kernel must be odd")
        if self.morphology_erode_kernel % 2 == 0:
            raise ValueError("morphology_erode_kernel must be odd")
        if self.inner_edge_kernel % 2 == 0:
            raise ValueError("inner_edge_kernel must be odd")
        profile_ids = [profile.profile_id for profile in self.profiles]
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("task profile IDs must be unique")
        return self

    def profile(self, profile_id: str) -> TaskProfile:
        profile = next((item for item in self.profiles if item.profile_id == profile_id), None)
        if profile is None:
            raise ObjectTrackingError(f"unknown object task profile: {profile_id}")
        return profile


def load_object_tracking_config(path: str | Path) -> ObjectTrackingConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("YAML root must be a mapping")
        if isinstance(payload.get("profiles"), list):
            payload["profiles"] = tuple(payload["profiles"])
        return ObjectTrackingConfig.model_validate(payload)
    except (OSError, ValueError, ValidationError, yaml.YAMLError) as exc:
        raise ObjectTrackingError(f"invalid object tracking config: {config_path}") from exc
