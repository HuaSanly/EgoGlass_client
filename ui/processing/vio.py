"""UI-owned orchestration for offline Basalt VIO runs."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from run_vio import run_session
from schemas import VioPose, VioTrajectory
from slam_vio import BasaltExecutionError, parse_euroc_trajectory


class VioRunState(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class VioRunInfo:
    """One UI-discoverable Basalt run and its optional parsed trajectory."""

    run_id: str
    session_id: str
    clip_id: str | None
    state: VioRunState
    output_directory: Path
    started_at_unix_ns: int
    completed_at_unix_ns: int | None
    trajectory: VioTrajectory | None
    error: str | None = None

    @property
    def is_viewable(self) -> bool:
        return self.state is VioRunState.COMPLETED and self.trajectory is not None

    @property
    def pose_count(self) -> int:
        return len(self.trajectory.poses) if self.trajectory is not None else 0

    def pose_at(self, session_time_ns: int) -> VioPose | None:
        if self.trajectory is None:
            return None
        return self.trajectory.pose_at(session_time_ns)


class OfflineVioService:
    """Run Basalt from the UI without joining the live inference path."""

    def __init__(
        self,
        recordings_root: str | Path,
        *,
        config_directory: str | Path = "config",
        allow_unverified_calibration: bool = True,
    ) -> None:
        self.recordings_root = Path(recordings_root).expanduser().resolve()
        self.config_directory = Path(config_directory).expanduser().resolve()
        self.allow_unverified_calibration = allow_unverified_calibration
        self._lock = threading.Lock()
        self._active = False

    def run(self, session_id: str, *, clip_id: str | None = None) -> VioRunInfo:
        """Run one complete session or clip and persist it under ``derived/vio``."""

        with self._lock:
            if self._active:
                raise RuntimeError("an offline VIO run is already active")
            self._active = True
        started = time.time_ns()
        run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
        session_path = self._session_path(session_id)
        output = session_path / "derived" / "vio" / "basalt" / run_id
        try:
            summary = run_session(
                session_path,
                output,
                sensor_config_path=self.config_directory / "sensor-preprocessing.yaml",
                basalt_config_path=self.config_directory / "basalt-vio.yaml",
                clip_id=clip_id,
                allow_unverified_calibration=self.allow_unverified_calibration,
            )
            return self._read_run(output, summary)
        except Exception as error:
            self._write_failure_manifest(
                output,
                run_id=run_id,
                session_id=session_id,
                clip_id=clip_id,
                started_at_unix_ns=started,
                error=str(error),
            )
            raise
        finally:
            with self._lock:
                self._active = False

    def list_runs(self, session_id: str) -> tuple[VioRunInfo, ...]:
        """List persisted Basalt runs, retaining failed runs for diagnostics."""

        root = self._session_path(session_id) / "derived" / "vio" / "basalt"
        if not root.is_dir():
            return ()
        runs: list[VioRunInfo] = []
        for directory in root.iterdir():
            if not directory.is_dir() or not (directory / "run.json").is_file():
                continue
            try:
                payload = json.loads((directory / "run.json").read_text(encoding="utf-8"))
                runs.append(self._read_run(directory, payload))
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
                BasaltExecutionError,
            ):
                continue
        runs.sort(key=lambda item: item.started_at_unix_ns, reverse=True)
        return tuple(runs)

    def close(self) -> None:
        """Reject no work and leave lifecycle ownership to the caller's future."""

    def _read_run(self, directory: Path, payload: dict[str, object]) -> VioRunInfo:
        run_id = _required_string(payload, "run_id")
        session_id = _required_string(payload, "session_id")
        if run_id != directory.name:
            raise ValueError("VIO run directory does not match run_id")
        state = VioRunState(_required_string(payload, "state"))
        clip_id = payload.get("clip_id")
        if clip_id is not None and not isinstance(clip_id, str):
            raise TypeError("VIO clip_id must be a string or null")
        started = _required_integer(payload, "started_at_unix_ns")
        completed = payload.get("completed_at_unix_ns")
        if completed is not None and (
            not isinstance(completed, int) or isinstance(completed, bool)
        ):
            raise TypeError("VIO completed timestamp must be an integer or null")
        error = payload.get("error")
        trajectory: VioTrajectory | None = None
        if state is VioRunState.COMPLETED:
            trajectory = parse_euroc_trajectory(directory / "trajectory.csv")
        return VioRunInfo(
            run_id,
            session_id,
            clip_id,
            state,
            directory,
            started,
            completed,
            trajectory,
            str(error) if error else None,
        )

    @staticmethod
    def _write_failure_manifest(
        directory: Path,
        *,
        run_id: str,
        session_id: str,
        clip_id: str | None,
        started_at_unix_ns: int,
        error: str,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0",
            "contract_id": "basalt-vio-run-v1",
            "run_id": run_id,
            "session_id": session_id,
            "clip_id": clip_id,
            "state": VioRunState.FAILED.value,
            "started_at_unix_ns": started_at_unix_ns,
            "completed_at_unix_ns": time.time_ns(),
            "error": error,
        }
        (directory / "run.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _session_path(self, session_id: str) -> Path:
        candidate = (self.recordings_root / session_id).resolve()
        if not candidate.is_relative_to(self.recordings_root) or not candidate.is_dir():
            raise FileNotFoundError("recording session is unavailable")
        return candidate


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"VIO manifest field {key!r} must be a non-empty string")
    return value


def _required_integer(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"VIO manifest field {key!r} must be an integer")
    return value
