"""Configuration for the external Basalt VIO executable."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class BasaltConfigError(ValueError):
    """Raised when the Basalt configuration cannot be loaded safely."""


class BasaltVioConfig(BaseModel):
    """Strict, reproducible settings for an offline Basalt invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1.0"]
    executable: str = Field(default="basalt_vio", min_length=1)
    executable_args: tuple[str, ...] = ()
    dataset_type: Literal["euroc"] = "euroc"
    show_gui: bool = False
    use_imu: bool = True
    use_double: bool = False
    num_threads: int = Field(default=0, ge=0)
    max_frames: int = Field(default=0, ge=0)
    allow_unverified_calibration: bool = False
    input_is_rectified: bool = True
    config_path: Path | None = None

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
