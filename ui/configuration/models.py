from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ui.processing.models import DEFAULT_PRESETS


class ClientRuntimeConfig(BaseModel):
    """Restart-bound settings owned by the unified native client."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1.0"] = "1.0"
    host: str = Field(default="0.0.0.0", min_length=1)
    port: int = Field(default=8770, ge=1, le=65_535)
    discovery_port: int = Field(default=8771, ge=1, le=65_535)
    enable_discovery: bool = True
    recordings_root: Path = Path("local-data/recordings")

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("host cannot be empty")
        return normalized

    @classmethod
    def load(cls, path: str | Path) -> ClientRuntimeConfig:
        config_path = Path(path).expanduser().resolve()
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("client runtime config must be a mapping")
        recordings_root = payload.get("recordings_root")
        if not isinstance(recordings_root, str) or not recordings_root.strip():
            raise ValueError("recordings_root must be a path")
        resolved_root = Path(recordings_root).expanduser()
        if not resolved_root.is_absolute():
            resolved_root = config_path.parent / resolved_root
        return cls.model_validate(
            {**payload, "recordings_root": resolved_root.resolve()}
        )


class PerceptionRuntimeConfig(BaseModel):
    """Live orchestration values kept in perception-runtime.yaml."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1.0"] = "1.0"
    enabled: bool = False
    max_live_inference_fps: float = Field(default=6.0, gt=0.0, le=60.0)


class VideoProcessingConfig(BaseModel):
    """Defaults applied only when a new offline job is submitted."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1.0"] = "1.0"
    default_preset_id: str = DEFAULT_PRESETS[0].preset_id
    auto_enqueue_on_session_complete: bool = False
    default_output_result_type: Literal["structured_results"] = "structured_results"

    @field_validator("default_preset_id")
    @classmethod
    def validate_preset_id(cls, value: str) -> str:
        if not any(preset.preset_id == value for preset in DEFAULT_PRESETS):
            raise ValueError("unknown video-processing preset")
        return value
