"""Configuration for the external Basalt VIO executable."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class BasaltConfigError(ValueError):
    """Raised when the Basalt configuration cannot be loaded safely."""


class BasaltVioConfig(BaseModel):
    """Strict, reproducible settings for an offline Basalt invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1.0"]
    backend: Literal["native", "wsl"] = "native"
    executable: str = Field(default="basalt_vio", min_length=1)
    executable_args: tuple[str, ...] = ()
    basalt_revision: str = Field(
        default="0f3b2b52c807f70ff4e2973ce253c73329eea7bc",
        min_length=40,
        max_length=40,
        pattern=r"^[0-9a-f]{40}$",
    )
    wsl_distribution: str = Field(
        default="Nvidia_SDKM_Ubuntu_22.04_JetPack_7.2",
        min_length=1,
    )
    wsl_executable: str = Field(
        default="/home/nvidia/egoglass/tools/basalt-build/basalt_vio",
        min_length=1,
    )
    wsl_stage_root: str = Field(
        default="/home/nvidia/.cache/egoglass/basalt",
        min_length=1,
    )
    wsl_keep_stage_on_failure: bool = True
    wsl_timeout_seconds: int = Field(default=0, ge=0)
    dataset_type: Literal["euroc"] = "euroc"
    show_gui: bool = False
    use_imu: bool = True
    use_double: bool = False
    num_threads: int = Field(default=0, ge=0)
    max_frames: int = Field(default=0, ge=0)
    allow_unverified_calibration: bool = False
    input_is_rectified: bool = True
    config_path: Path | None = None

    @model_validator(mode="after")
    def validate_backend(self) -> BasaltVioConfig:
        """Reject WSL paths that cannot be staged or executed safely."""

        if self.backend != "wsl":
            return self
        if self.show_gui:
            raise ValueError("WSL Basalt runs must be headless")
        for field_name in ("wsl_executable", "wsl_stage_root"):
            value = PurePosixPath(getattr(self, field_name))
            if not value.is_absolute() or ".." in value.parts:
                raise ValueError(f"{field_name} must be an absolute normalized POSIX path")
        return self

    @classmethod
    def load(cls, path: str | Path) -> BasaltVioConfig:
        """Read YAML and resolve the optional Basalt JSON config path."""

        config_path = Path(path).resolve()
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise BasaltConfigError("Basalt configuration must be a mapping")
            executable_args = payload.get("executable_args")
            if isinstance(executable_args, list):
                payload = {**payload, "executable_args": tuple(executable_args)}
            selected = payload.get("config_path")
            if selected is not None:
                if not isinstance(selected, str) or not selected.strip():
                    raise BasaltConfigError("config_path must be a non-empty string")
                selected_path = Path(selected)
                if not selected_path.is_absolute():
                    selected_path = config_path.parent / selected_path
                payload = {**payload, "config_path": selected_path.resolve()}
            return cls.model_validate(payload)
        except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
            raise BasaltConfigError("invalid Basalt configuration") from exc
