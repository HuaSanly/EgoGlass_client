"""Small processing metadata shapes shared by CLI and UI result readers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AlgorithmRunMetadata:
    schema_version: str
    run_id: str
    session_id: str
    algorithm: str
    configuration_revision: int | None


@dataclass(frozen=True, slots=True)
class ProcessingArtifactRef:
    run_id: str
    output_directory: Path
    result_count: int
    readable: bool = True
