"""Boundary models used by the Basalt adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from schemas import VioTrajectory


class BasaltError(RuntimeError):
    """Base class for errors raised by the Basalt adapter.

    Backend metadata is carried on failures as well as successful results so
    the UI and headless workflows can persist enough evidence to diagnose a
    retained WSL staging directory without parsing human-readable logs.
    """

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        backend_metadata: tuple[tuple[str, str], ...] = (),
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.backend_metadata = backend_metadata


class BasaltExportError(BasaltError):
    """The prepared EgoGlass data cannot be represented as EuRoC input."""


class BasaltUnavailableError(BasaltError):
    """The configured Basalt executable is not available."""


class BasaltExecutionError(BasaltError):
    """Basalt returned a non-zero exit code or malformed output."""


@dataclass(frozen=True, slots=True)
class BasaltDataset:
    """Paths and deterministic counts for an exported EuRoC dataset."""

    root: Path
    camera_data_csv: Path
    imu_data_csv: Path
    frame_count: int
    imu_count: int
    interpolated_imu_count: int
    skipped_imu_timestamps: int


@dataclass(frozen=True, slots=True)
class BasaltRunResult:
    """A completed Basalt invocation and its parsed trajectory."""

    output_directory: Path
    dataset: BasaltDataset
    trajectory: VioTrajectory
    trajectory_path: Path
    command: tuple[str, ...]
    returncode: int
    stdout_path: Path
    stderr_path: Path
    backend: str = "native"
    backend_metadata: tuple[tuple[str, str], ...] = ()
