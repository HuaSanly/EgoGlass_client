"""Run Basalt as an isolated offline subprocess and parse its trajectory."""

from __future__ import annotations

import csv
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

from schemas import VioPose, VioTrajectory
from sensor_preprocessing import PreparedFrameBundle, SensorCalibration

from .calibration import calibration_is_verified
from .config import BasaltVioConfig
from .euroc_export import BasaltEuRoCExporter
from .models import (
    BasaltExecutionError,
    BasaltRunResult,
    BasaltUnavailableError,
)


def parse_euroc_trajectory(path: str | Path) -> VioTrajectory:
    """Parse Basalt's ``trajectory.csv`` output with strict column checks."""

    trajectory_path = Path(path)
    if not trajectory_path.is_file():
        raise BasaltExecutionError(f"Basalt did not produce trajectory: {trajectory_path}")
    poses: list[VioPose] = []
    try:
        with trajectory_path.open("r", encoding="utf-8", newline="") as stream:
            rows = csv.reader(stream)
            for row in rows:
                if not row or row[0].lstrip().startswith("#"):
                    continue
                if len(row) != 8:
                    raise ValueError("trajectory row must contain eight columns")
                values = [float(value.strip()) for value in row]
                timestamp = int(values[0])
                if values[0] != timestamp:
                    raise ValueError("trajectory timestamp must be an integer nanosecond")
                poses.append(
                    VioPose(
                        timestamp_ns=timestamp,
                        position_m=(values[1], values[2], values[3]),
                        quaternion_wxyz=(values[4], values[5], values[6], values[7]),
                    )
                )
    except (OSError, UnicodeError, ValueError) as exc:
        raise BasaltExecutionError(f"invalid Basalt trajectory: {trajectory_path}") from exc
    if not poses:
        raise BasaltExecutionError("Basalt trajectory is empty")
    try:
        return VioTrajectory(tuple(poses))
    except ValueError as exc:
        raise BasaltExecutionError("Basalt trajectory timestamps are not increasing") from exc


class BasaltVioRunner:
    """Export a prepared session, invoke Basalt, and return typed poses."""

    def __init__(
        self,
        config: BasaltVioConfig,
        *,
        exporter: BasaltEuRoCExporter | None = None,
    ) -> None:
        self.config = config
        self.exporter = exporter or BasaltEuRoCExporter()

    def run(
        self,
        bundles: Iterable[PreparedFrameBundle],
        output_directory: str | Path,
        *,
        calibration: SensorCalibration,
    ) -> BasaltRunResult:
        """Run one offline VIO job in ``output_directory``.

        The child process runs with that directory as its working directory
        because Basalt writes ``trajectory.csv`` relative to the current
        directory. No UI, gateway, or long-lived Basalt process is involved.
        """

        if not self.config.allow_unverified_calibration and not calibration_is_verified(
            calibration
        ):
            raise BasaltExecutionError(
                "refusing unverified sensor calibration; set "
                "allow_unverified_calibration explicitly"
            )
        output = Path(output_directory).resolve()
        output.mkdir(parents=True, exist_ok=True)
        dataset = self.exporter.export(
            bundles,
            output / "dataset",
            calibration=calibration,
            input_is_rectified=self.config.input_is_rectified,
        )
        executable = shutil.which(self.config.executable) or self.config.executable
        if not Path(executable).exists() and shutil.which(executable) is None:
            raise BasaltUnavailableError(f"Basalt executable not found: {self.config.executable}")
        command = [
            executable,
            *self.config.executable_args,
            "--dataset-path",
            str(dataset.root),
            "--cam-calib",
            str(dataset.root / "calibration.json"),
            "--dataset-type",
            self.config.dataset_type,
            "--show-gui",
            "1" if self.config.show_gui else "0",
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
        if self.config.config_path is not None:
            command.extend(["--config-path", str(self.config.config_path)])
        stdout_path = output / "basalt.stdout.log"
        stderr_path = output / "basalt.stderr.log"
        run_log_path = output / "run.log"
        run_log_path.write_text(
            "command: " + " ".join(command) + "\n",
            encoding="utf-8",
        )
        try:
            completed = subprocess.run(
                command,
                cwd=output,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise BasaltUnavailableError(
                f"failed to start Basalt: {self.config.executable}"
            ) from exc
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        with run_log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"returncode: {completed.returncode}\n")
        if completed.returncode != 0:
            raise BasaltExecutionError(
                f"Basalt failed with exit code {completed.returncode}: {completed.stderr.strip()}",
                returncode=completed.returncode,
            )
        trajectory_path = output / "trajectory.csv"
        trajectory = parse_euroc_trajectory(trajectory_path)
        return BasaltRunResult(
            output_directory=output,
            dataset=dataset,
            trajectory=trajectory,
            trajectory_path=trajectory_path,
            command=tuple(command),
            returncode=completed.returncode,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
