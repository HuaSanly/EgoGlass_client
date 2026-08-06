from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path

from .export import ExportSummary, export_annotated_clip
from .job_store import ProcessingJobStore
from .models import (
    DEFAULT_PRESETS,
    ProcessingJob,
    ProcessingJobState,
    ProcessingPreset,
    ProcessingRunInfo,
    ProcessingRunState,
    ProcessingRunSummary,
    ProcessingServiceSnapshot,
)
from .results import ProcessingResultStore
from .runner import ProcessingCanceled, SessionProcessingRunner


class VideoProcessingService:
    """Single-process persistent worker for offline perception pipelines."""

    def __init__(
        self,
        recordings_root: str | Path,
        *,
        runner: SessionProcessingRunner | None = None,
        on_gpu_job_changed: Callable[[bool], None] | None = None,
        offline_vio_runner: Callable[..., object] | None = None,
        configuration_provenance_provider: (
            Callable[[], tuple[int, Mapping[str, str], str]] | None
        ) = None,
    ) -> None:
        self.recordings_root = Path(recordings_root).expanduser().resolve()
        self.recordings_root.mkdir(parents=True, exist_ok=True)
        self.store = ProcessingJobStore(
            self.recordings_root / ".processing" / "jobs.sqlite3"
        )
        self.runner = runner or SessionProcessingRunner()
        self.on_gpu_job_changed = on_gpu_job_changed
        self.offline_vio_runner = offline_vio_runner
        self.configuration_provenance_provider = configuration_provenance_provider
        self._condition = threading.Condition()
        self._stop_requested = False
        self._thread: threading.Thread | None = None
        self._active_job_id: str | None = None
        self._auto_enqueue = self.store.setting("auto_enqueue", "false") == "true"
        stored_preset = self.store.setting("default_preset_id", DEFAULT_PRESETS[0].preset_id)
        self._default_preset_id = (
            stored_preset
            if any(preset.preset_id == stored_preset for preset in DEFAULT_PRESETS)
            else DEFAULT_PRESETS[0].preset_id
        )
        self._default_output_result_type = "structured_results"
        self._revision = 0

    @property
    def presets(self) -> tuple[ProcessingPreset, ...]:
        return DEFAULT_PRESETS

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                return
            self._stop_requested = False
            self._thread = threading.Thread(
                target=self._worker,
                name="video-processing",
                daemon=False,
            )
            self._thread.start()

    def close(self, timeout_seconds: float = 30.0) -> None:
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout_seconds)
            if thread.is_alive():
                raise TimeoutError("video-processing worker did not stop")
        self.store.recover_interrupted()
        with self._condition:
            self._thread = None
            self._active_job_id = None
            self._revision += 1

    def enqueue(
        self,
        session_id: str,
        *,
        clip_id: str | None = None,
        preset_id: str | None = None,
    ) -> ProcessingJob:
        self._session_path(session_id)
        preset_id = preset_id or self._default_preset_id
        preset = next((item for item in self.presets if item.preset_id == preset_id), None)
        if preset is None:
            raise KeyError(f"unknown processing preset {preset_id!r}")
        configuration_revision, configuration_hashes, configuration_snapshot = (
            self._configuration_provenance()
        )
        job = self.store.enqueue(
            session_id,
            clip_id,
            preset,
            configuration_revision=configuration_revision,
            configuration_sha256_by_file=tuple(sorted(configuration_hashes.items())),
            configuration_snapshot_json=configuration_snapshot,
        )
        self._notify_changed()
        return job

    def cancel(self, job_id: str) -> ProcessingJob:
        job = self.store.request_cancel(job_id)
        self._notify_changed()
        return job

    def retry(self, job_id: str) -> ProcessingJob:
        configuration_revision, configuration_hashes, configuration_snapshot = (
            self._configuration_provenance()
        )
        job = self.store.retry(
            job_id,
            configuration_revision=configuration_revision,
            configuration_sha256_by_file=tuple(sorted(configuration_hashes.items())),
            configuration_snapshot_json=configuration_snapshot,
        )
        self._notify_changed()
        return job

    def set_auto_enqueue(self, enabled: bool) -> bool:
        with self._condition:
            self._auto_enqueue = bool(enabled)
            self.store.set_setting("auto_enqueue", "true" if enabled else "false")
            self._revision += 1
        return self._auto_enqueue

    def set_default_preset(self, preset_id: str) -> str:
        if not any(preset.preset_id == preset_id for preset in self.presets):
            raise KeyError(f"unknown processing preset {preset_id!r}")
        with self._condition:
            self._default_preset_id = preset_id
            self.store.set_setting("default_preset_id", preset_id)
            self._revision += 1
        return preset_id

    def set_configuration_defaults(
        self,
        *,
        default_preset_id: str,
        auto_enqueue_on_session_complete: bool,
        default_output_result_type: str,
    ) -> None:
        """Apply defaults to jobs submitted after this call."""

        if not any(preset.preset_id == default_preset_id for preset in self.presets):
            raise KeyError(f"unknown processing preset {default_preset_id!r}")
        if default_output_result_type != "structured_results":
            raise ValueError("unsupported default output result type")
        with self._condition:
            self._default_preset_id = default_preset_id
            self._auto_enqueue = bool(auto_enqueue_on_session_complete)
            self._default_output_result_type = default_output_result_type
            self._revision += 1

    def enqueue_completed_session(self, session_id: str) -> ProcessingJob | None:
        with self._condition:
            enabled = self._auto_enqueue
        if not enabled:
            return None
        if any(job.session_id == session_id for job in self.store.list_jobs()):
            return None
        return self.enqueue(session_id)

    def snapshot(self) -> ProcessingServiceSnapshot:
        with self._condition:
            revision = self._revision
            active_job_id = self._active_job_id
        return ProcessingServiceSnapshot(
            revision,
            active_job_id,
            self._auto_enqueue,
            self.store.list_jobs(),
            self._default_preset_id,
            self._default_output_result_type,
        )

    def _configuration_provenance(self) -> tuple[int, dict[str, str], str]:
        provider = self.configuration_provenance_provider
        if provider is None:
            return 0, {}, "{}"
        revision, values, snapshot = provider()
        return revision, dict(values), snapshot

    def result_for_frame(
        self,
        session_id: str,
        run_id: str,
        clip_id: str,
        frame_index: int,
        session_time_ns: int,
    ) -> dict[str, object] | None:
        path = self._run_directory(session_id, run_id) / "results.sqlite"
        manifest = self._read_run_manifest(session_id, run_id)
        stride = int(manifest["preset"]["inference_stride_frames"])
        return ProcessingResultStore(path, read_only=True).result_for_frame(
            clip_id,
            frame_index,
            session_time_ns,
            hold_previous_frames=max(0, stride - 1),
        )

    def list_runs(self, session_id: str) -> tuple[ProcessingRunInfo, ...]:
        root = self._session_path(session_id) / "derived" / "video-processing"
        if not root.is_dir():
            return ()
        runs: list[ProcessingRunInfo] = []
        for path in root.iterdir():
            manifest_path = path / "run.json"
            if not path.is_dir() or not manifest_path.is_file():
                continue
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            try:
                run = _processing_run_info(path, payload, session_id)
            except (KeyError, TypeError, ValueError):
                continue
            runs.append(run)
        runs.sort(key=lambda item: item.started_at_unix_ns, reverse=True)
        return tuple(runs)

    def session_directory(self, session_id: str) -> Path:
        """Return a validated capture-session directory for the local player."""

        return self._session_path(session_id)

    def export_annotated_clip(
        self,
        session_id: str,
        run_id: str,
        clip_id: str,
    ) -> ExportSummary:
        manifest = self._read_run_manifest(session_id, run_id)
        if manifest.get("state") not in {"completed", "partial"}:
            raise ValueError("only completed or partial processing runs can be exported")
        preset = manifest.get("preset")
        if not isinstance(preset, dict):
            raise ValueError("processing run has no valid preset")
        stride = int(preset.get("inference_stride_frames", 1))
        run_clip_id = manifest.get("clip_id")
        if run_clip_id is not None and run_clip_id != clip_id:
            raise ValueError("processing run does not contain the selected clip")
        return export_annotated_clip(
            self._session_path(session_id),
            self._run_directory(session_id, run_id),
            clip_id,
            hold_previous_frames=max(0, stride - 1),
        )

    def _worker(self) -> None:
        while True:
            with self._condition:
                if self._stop_requested:
                    return
            job = self.store.claim_next()
            if job is None:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._stop_requested
                        or any(
                            item.state is ProcessingJobState.QUEUED
                            for item in self.store.list_jobs()
                        ),
                        timeout=1.0,
                    )
                continue
            self._process(job)

    def _process(self, job: ProcessingJob) -> None:
        run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
        output = self._session_path(job.session_id) / "derived" / "video-processing" / run_id
        with self._condition:
            self._active_job_id = job.job_id
            self._revision += 1
        gpu_claim_started = self.on_gpu_job_changed is not None
        try:
            if self.on_gpu_job_changed is not None:
                self.on_gpu_job_changed(True)
            total = self.runner.inspect(self._session_path(job.session_id), job.clip_id)
            if self._is_canceled(job.job_id):
                raise ProcessingCanceled()
            try:
                job = self.store.start_run(job.job_id, run_id, total)
            except RuntimeError:
                if self.store.require(job.job_id).state is ProcessingJobState.CANCELING:
                    raise ProcessingCanceled() from None
                raise
            try:
                vio_run_info = None
                vio_error = None
                if self.offline_vio_runner is not None:
                    if self._is_canceled(job.job_id):
                        raise ProcessingCanceled("processing canceled")
                    self.store.update_progress(
                        job.job_id,
                        0,
                        total,
                        "Running offline SLAM/VIO",
                    )
                    try:
                        vio_run_info = self.offline_vio_runner(
                            job.session_id,
                            clip_id=job.clip_id,
                        )
                    except Exception as error:
                        vio_error = str(error)
                summary = self.runner.run(
                    job,
                    self._session_path(job.session_id),
                    output,
                    progress=lambda current, count: self._progress(
                        job.job_id, current, count
                    ),
                    is_canceled=lambda: self._is_canceled(job.job_id),
                    vio_run_info=vio_run_info,
                    vio_error=vio_error,
                )
            finally:
                release_gpu = getattr(self.runner, "release_gpu", None)
                if callable(release_gpu):
                    release_gpu()
            if vio_error is not None and not summary.partial:
                self._mark_run_terminal(output, "partial", vio_error)
        except ProcessingCanceled as error:
            self._mark_run_terminal(output, "canceled", str(error) or "processing canceled")
            current = self.store.require(job.job_id)
            if current.state is ProcessingJobState.CANCELING:
                self.store.mark_canceled(job.job_id)
        except Exception as error:
            current = self.store.require(job.job_id)
            terminal_state = (
                "canceled" if current.state is ProcessingJobState.CANCELING else "failed"
            )
            self._mark_run_terminal(output, terminal_state, str(error))
            if current.state is ProcessingJobState.CANCELING:
                self.store.mark_canceled(job.job_id)
            elif current.state in {ProcessingJobState.PREPARING, ProcessingJobState.RUNNING}:
                self.store.fail(job.job_id, str(error))
        else:
            self._complete(job.job_id, summary, vio_error=vio_error)
        finally:
            try:
                if gpu_claim_started and self.on_gpu_job_changed is not None:
                    self.on_gpu_job_changed(False)
            finally:
                with self._condition:
                    self._active_job_id = None
                    self._revision += 1
                    self._condition.notify_all()

    def _progress(self, job_id: str, current: int, total: int) -> None:
        self.store.update_progress(job_id, current, total, f"正在处理 {current}/{total}")
        if current == total or current % 10 == 0:
            self._notify_changed()

    def _complete(
        self,
        job_id: str,
        summary: ProcessingRunSummary,
        *,
        vio_error: str | None = None,
    ) -> None:
        if summary.partial or vio_error is not None:
            reason = vio_error or "部分手帧缺少 100 ms 内的 VIO 位姿"
            self.store.partial(job_id, f"部分完成: {reason}")
            return
        detail = f"处理完成 {summary.inferred_frame_count} 帧手部追踪"
        if self.offline_vio_runner is not None:
            detail += "与 SLAM/VIO"
        self.store.complete(
            job_id,
            detail,
        )

    def _is_canceled(self, job_id: str) -> bool:
        with self._condition:
            stopping = self._stop_requested
        return stopping or self.store.require(job_id).state is ProcessingJobState.CANCELING

    @staticmethod
    def _mark_run_terminal(output: Path, state: str, error: str) -> None:
        """Keep the persisted run in sync when a later pipeline stage fails."""

        manifest = output / "run.json"
        if not manifest.is_file():
            return
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return
            payload["state"] = state
            payload["completed_at_unix_ns"] = time.time_ns()
            payload["error"] = error
            temporary = manifest.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, manifest)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            return

    def _notify_changed(self) -> None:
        with self._condition:
            self._revision += 1
            self._condition.notify_all()

    def _session_path(self, session_id: str) -> Path:
        candidate = (self.recordings_root / session_id).resolve()
        if not candidate.is_relative_to(self.recordings_root):
            raise ValueError("session path escapes recordings root")
        if not candidate.is_dir():
            raise FileNotFoundError("recording session is unavailable")
        return candidate

    def _run_directory(self, session_id: str, run_id: str) -> Path:
        candidate = (
            self._session_path(session_id) / "derived" / "video-processing" / run_id
        ).resolve()
        if not candidate.is_relative_to(self._session_path(session_id)) or not candidate.is_dir():
            raise FileNotFoundError("processing run is unavailable")
        return candidate

    def _read_run_manifest(self, session_id: str, run_id: str) -> dict[str, object]:
        payload = json.loads(
            (self._run_directory(session_id, run_id) / "run.json").read_text(encoding="utf-8")
        )
        if payload.get("session_id") != session_id or payload.get("run_id") != run_id:
            raise ValueError("processing run identity mismatch")
        return payload


def _processing_run_info(
    run_directory: Path,
    payload: dict[str, object],
    expected_session_id: str,
) -> ProcessingRunInfo:
    run_id = _required_string(payload, "run_id")
    session_id = _required_string(payload, "session_id")
    if run_id != run_directory.name or session_id != expected_session_id:
        raise ValueError("processing run identity mismatch")
    clip_id = payload.get("clip_id")
    if clip_id is not None and not isinstance(clip_id, str):
        raise TypeError("processing run clip_id must be a string or null")
    preset_payload = payload.get("preset")
    if not isinstance(preset_payload, dict):
        raise TypeError("processing run preset must be an object")
    preset = ProcessingPreset(
        _required_string(preset_payload, "preset_id"),
        _required_string(preset_payload, "display_name"),
        _required_integer(preset_payload, "inference_stride_frames"),
    )
    state = ProcessingRunState(_required_string(payload, "state"))
    results_path = run_directory / "results.sqlite"
    unavailable_reason: str | None = None
    if state not in {ProcessingRunState.COMPLETED, ProcessingRunState.PARTIAL}:
        error = payload.get("error")
        unavailable_reason = str(error) if error else f"运行状态为 {state.value}"
    else:
        try:
            ProcessingResultStore(results_path, read_only=True)
        except Exception as error:
            unavailable_reason = str(error)
    completed_at = payload.get("completed_at_unix_ns")
    if completed_at is not None and (
        not isinstance(completed_at, int) or isinstance(completed_at, bool) or completed_at < 0
    ):
        raise TypeError("completed_at_unix_ns must be a non-negative integer or null")
    return ProcessingRunInfo(
        run_id,
        session_id,
        clip_id,
        preset,
        state,
        _required_integer(payload, "input_frame_count"),
        _required_integer(payload, "inferred_frame_count"),
        _required_integer(payload, "detected_hand_count"),
        _required_integer(payload, "started_at_unix_ns"),
        completed_at,
        results_path,
        unavailable_reason is None,
        unavailable_reason,
        str(payload["vio_run_id"])
        if isinstance(payload.get("vio_run_id"), str)
        else None,
    )


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{key} must be a non-empty string")
    return value


def _required_integer(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError(f"{key} must be a non-negative integer")
    return value
