from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType


class ConfigImpact(StrEnum):
    """When a saved value may be consumed by its owning runtime."""

    IMMEDIATE = "immediate"
    NEXT_SESSION = "next_session"
    NEXT_TASK = "next_task"
    RESTART_CLIENT = "restart_client"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    module_id: str
    field_path: str
    message: str


@dataclass(frozen=True, slots=True)
class ConfigChange:
    module_id: str
    field_path: str
    impact: ConfigImpact


@dataclass(frozen=True, slots=True)
class ModuleConfigSnapshot:
    module_id: str
    display_name: str
    values: Mapping[str, object]
    source_paths: tuple[Path, ...]
    field_impacts: Mapping[str, ConfigImpact]


@dataclass(frozen=True, slots=True)
class ConfigSnapshot:
    modules: tuple[ModuleConfigSnapshot, ...]
    revision: int
    dirty: bool

    def require_module(self, module_id: str) -> ModuleConfigSnapshot:
        for module in self.modules:
            if module.module_id == module_id:
                return module
        raise KeyError(f"unknown configuration module {module_id!r}")


@dataclass(frozen=True, slots=True)
class ConfigurationProvenance:
    revision: int
    sha256_by_file: Mapping[str, str]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "revision": self.revision,
            "sha256_by_file": dict(self.sha256_by_file),
        }


@dataclass(frozen=True, slots=True)
class ConfigApplyResult:
    changed_modules: tuple[str, ...]
    immediate_applied: tuple[str, ...]
    pending_restart: tuple[str, ...]
    pending_next_task: tuple[str, ...]
    warnings: tuple[str, ...]
    pending_next_session: tuple[str, ...] = ()
    changes: tuple[ConfigChange, ...] = ()
    provenance: ConfigurationProvenance = field(
        default_factory=lambda: ConfigurationProvenance(
            revision=0,
            sha256_by_file=MappingProxyType({}),
        )
    )


@dataclass(frozen=True, slots=True)
class ConfigurationApplyRequest:
    revision: int
    changes: tuple[ConfigChange, ...]
    values: Mapping[str, Mapping[str, object]]


class ConfigurationError(RuntimeError):
    """Base error for invalid or unsafe configuration operations."""


class ConfigurationValidationError(ConfigurationError):
    def __init__(self, issues: tuple[ValidationIssue, ...]) -> None:
        self.issues = issues
        detail = "; ".join(
            f"{issue.module_id}.{issue.field_path}: {issue.message}" for issue in issues
        )
        super().__init__(detail or "configuration validation failed")
