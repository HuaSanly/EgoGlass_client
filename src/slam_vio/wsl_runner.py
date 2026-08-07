"""Execute one staged Basalt run inside a configured WSL distribution."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .config import BasaltVioConfig
from .models import BasaltDataset, BasaltExecutionError, BasaltUnavailableError

_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")
_CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class WslExecutionResult:
    """Artifacts and provenance returned by a successful WSL invocation."""

    command: tuple[str, ...]
    returncode: int
    trajectory_path: Path
    stdout_path: Path
    stderr_path: Path
    metadata: tuple[tuple[str, str], ...]


def resolve_wsl_executable() -> Path | None:
    """Locate the Windows WSL launcher without assuming a fixed system directory."""

    discovered = shutil.which("wsl.exe") or shutil.which("wsl")
    return Path(discovered).resolve() if discovered else None


class WslBasaltExecutor:
    """Stage EuRoC input into WSL ext4 and copy only run artifacts back."""

    def __init__(
        self,
        config: BasaltVioConfig,
        *,
        command_runner: _CommandRunner = subprocess.run,
        wsl_executable: Path | None = None,
        helper_script: Path | None = None,
    ) -> None:
        if config.backend != "wsl":
            raise ValueError("WslBasaltExecutor requires the wsl backend")
        self.config = config
        self._run_command = command_runner
        self._wsl_executable = wsl_executable
        self._helper_script = helper_script or (
            Path(__file__).resolve().parents[2] / "scripts" / "wsl" / "run-basalt.sh"
        )

    def run(self, dataset: BasaltDataset, output_directory: Path) -> WslExecutionResult:
        """Copy one dataset to WSL, execute Basalt, and materialize its trajectory."""

        output = output_directory.resolve()
        output.mkdir(parents=True, exist_ok=True)
        stage_path = self._stage_path(output)
        base_metadata = self._metadata(stage_path, "not_started")
        launcher = self._wsl_executable or resolve_wsl_executable()
        if launcher is None or not launcher.is_file():
            raise BasaltUnavailableError(
                "WSL launcher was not found on Windows PATH",
                backend_metadata=base_metadata,
            )
        if not self._helper_script.is_file():
            raise BasaltUnavailableError(
                f"WSL Basalt helper is missing: {self._helper_script}",
                backend_metadata=base_metadata,
            )

        source_dataset = self._translate_path(launcher, dataset.root)
        host_output = self._translate_path(launcher, output)
        helper_script = self._translate_path(launcher, self._helper_script)
        source_config = (
            self._translate_path(launcher, self.config.config_path)
            if self.config.config_path is not None
            else "-"
        )
        stage_root = str(PurePosixPath(self.config.wsl_stage_root))
        basalt_args = [
            *self.config.executable_args,
            "--dataset-type",
            self.config.dataset_type,
            "--show-gui",
            "0",
            "--use-imu",
            "1" if self.config.use_imu else "0",
            "--use-double",
            "1" if self.config.use_double else "0",
            "--num-threads",
            str(self.config.num_threads),
            "--max-frames",
            str(self.config.max_frames),
            "--save-trajectory",
            "euroc",
        ]
        command = (
            str(launcher),
            "--distribution",
            self.config.wsl_distribution,
            "--exec",
            "/bin/bash",
            helper_script,
            stage_root,
            stage_path,
            source_dataset,
            source_config,
            host_output,
            self.config.wsl_executable,
            "1" if self.config.wsl_keep_stage_on_failure else "0",
            *basalt_args,
        )
        run_log = output / "run.log"
        run_log.write_text(
            "backend: wsl\ncommand_argv: " + repr(command) + "\n",
            encoding="utf-8",
        )
        timeout = self.config.wsl_timeout_seconds or None
        try:
            completed = self._run_command(
                list(command),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise BasaltUnavailableError(
                "failed to start wsl.exe",
                backend_metadata=base_metadata,
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise BasaltExecutionError(
                f"WSL Basalt exceeded {self.config.wsl_timeout_seconds} seconds",
                backend_metadata=self._metadata(stage_path, "retained_or_active"),
            ) from exc
        except OSError as exc:
            raise BasaltUnavailableError(
                "failed to execute WSL Basalt helper",
                backend_metadata=base_metadata,
            ) from exc

        stdout_path = output / "basalt.stdout.log"
        stderr_path = output / "basalt.stderr.log"
        if not stdout_path.is_file():
            stdout_path.write_text(completed.stdout, encoding="utf-8")
        if not stderr_path.is_file():
            stderr_path.write_text(completed.stderr, encoding="utf-8")
        with run_log.open("a", encoding="utf-8") as stream:
            stream.write(f"returncode: {completed.returncode}\n")
            stream.write(f"wsl_stage_path: {stage_path}\n")

        if completed.returncode != 0:
            detail = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
            if not detail:
                detail = completed.stderr.strip()
            raise BasaltExecutionError(
                f"WSL Basalt failed with exit code {completed.returncode}: {detail}",
                returncode=completed.returncode,
                backend_metadata=self._metadata(
                    stage_path,
                    "retained" if self.config.wsl_keep_stage_on_failure else "cleaned",
                ),
            )
        trajectory_path = output / "trajectory.csv"
        if not trajectory_path.is_file():
            raise BasaltExecutionError(
                "WSL Basalt did not copy trajectory.csv to Windows",
                backend_metadata=self._metadata(
                    stage_path,
                    "retained" if self.config.wsl_keep_stage_on_failure else "cleaned",
                ),
            )
        metadata = self._metadata(stage_path, "cleaned")
        return WslExecutionResult(
            command=command,
            returncode=completed.returncode,
            trajectory_path=trajectory_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            metadata=metadata,
        )

    def _metadata(self, stage_path: str, staging_state: str) -> tuple[tuple[str, str], ...]:
        return (
            ("backend", "wsl"),
            ("wsl_distribution", self.config.wsl_distribution),
            ("wsl_executable", self.config.wsl_executable),
            ("basalt_revision", self.config.basalt_revision),
            ("wsl_stage_path", stage_path),
            ("staging_state", staging_state),
        )

    def _translate_path(self, launcher: Path, path: Path) -> str:
        resolved = path.resolve()
        command = [
            str(launcher),
            "--distribution",
            self.config.wsl_distribution,
            "--exec",
            "wslpath",
            "-a",
            str(resolved),
        ]
        try:
            completed = self._run_command(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=30,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
            raise BasaltUnavailableError("failed to translate a Windows path through WSL") from exc
        translated = completed.stdout.strip()
        if completed.returncode != 0 or not translated.startswith("/"):
            raise BasaltUnavailableError(
                f"wslpath failed for {resolved}: {completed.stderr.strip()}"
            )
        return translated

    def _stage_path(self, output: Path) -> str:
        run_component = _safe_component(output.name, "run")
        if output.parent.name == "basalt" and len(output.parents) >= 4:
            session_component = _safe_component(output.parents[3].name, "session")
        else:
            digest = hashlib.sha256(str(output).encode("utf-8")).hexdigest()[:12]
            session_component = f"session-{digest}"
        return str(
            PurePosixPath(self.config.wsl_stage_root)
            / session_component
            / run_component
        )


def _safe_component(value: str, fallback: str) -> str:
    sanitized = _SAFE_COMPONENT.sub("-", value).strip(".-")
    return sanitized[:80] or fallback
