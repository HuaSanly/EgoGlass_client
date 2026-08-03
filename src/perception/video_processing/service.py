from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from .contracts import (
    DEFAULT_PRESETS,
    ProcessingJob,
    ProcessingJobState,
    ProcessingPreset,
    ProcessingRunSummary,
    ProcessingServiceSnapshot,
)
from .export import ExportSummary, export_annotated_clip
from .job_store import ProcessingJobStore
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
    ) -> None:
        self.recordings_root = Path(recordings_root).expanduser().resolve()
        self.recordings_root.mkdir(parents=True, exist_ok=True)
        self.store = ProcessingJobStore(
            self.recordings_root / ".processing" / "jobs.sqlite3"
        )
        self.runner = runner or SessionProcessingRunner()
        self.on_gpu_job_changed = on_gpu_job_changed
        self._condition = threading.Condition()
        self._stop_requested = False
        self._thread: threading.Thread | None = None
        self._active_job_id: str | None = None
        self._auto_enqueue = self.store.setting("auto_enqueue", "false") == "true"
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
        preset_id: str = DEFAULT_PRESETS[0].preset_id,
    ) -> ProcessingJob:
        self._session_path(session_id)
        preset = next((item for item in self.presets if item.preset_id == preset_id), None)
        if preset is None:
            raise KeyError(f"unknown processing preset {preset_id!r}")
        job = self.store.enqueue(session_id, clip_id, preset)
        self._notify_changed()
        return job

    def cancel(self, job_id: str) -> ProcessingJob:
        job = self.store.request_cancel(job_id)
        self._notify_changed()
        return job

    def retry(self, job_id: str) -> ProcessingJob:
        job = self.store.retry(job_id)
        self._notify_changed()
        return job

    def set_auto_enqueue(self, enabled: bool) -> bool:
        with self._condition:
            self._auto_enqueue = bool(enabled)
            self.store.set_setting("auto_enqueue", "true" if enabled else "false")
            self._revision += 1
        return self._auto_enqueue

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
        )

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

    def list_runs(self, session_id: str) -> tuple[dict[str, object], ...]:
        root = self._session_path(session_id) / "derived" / "video-processing"
        if not root.is_dir():
            return ()
        manifests: list[dict[str, object]] = []
        for path in root.iterdir():
            manifest_path = path / "run.json"
            if not path.is_dir() or not manifest_path.is_file():
                continue
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("session_id") == session_id:
                manifests.append(payload)
        manifests.sort(key=lambda item: int(item.get("started_at_unix_ns", 0)), reverse=True)
        return tuple(manifests)

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
        if manifest.get("state") != "completed":
            raise ValueError("only completed processing runs can be exported")
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
                summary = self.runner.run(
                    job,
                    self._session_path(job.session_id),
                    output,
                    progress=lambda current, count: self._progress(job.job_id, current, count),
                    is_canceled=lambda: self._is_canceled(job.job_id),
                )
            finally:
                release_gpu = getattr(self.runner, "release_gpu", None)
                if callable(release_gpu):
                    release_gpu()
        except ProcessingCanceled:
            current = self.store.require(job.job_id)
            if current.state is ProcessingJobState.CANCELING:
                self.store.mark_canceled(job.job_id)
        except Exception as error:
            current = self.store.require(job.job_id)
            if current.state is ProcessingJobState.CANCELING:
                self.store.mark_canceled(job.job_id)
            elif current.state in {ProcessingJobState.PREPARING, ProcessingJobState.RUNNING}:
                self.store.fail(job.job_id, str(error))
        else:
            self._complete(job.job_id, summary)
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

    def _complete(self, job_id: str, summary: ProcessingRunSummary) -> None:
        self.store.complete(
            job_id,
            f"完成 {summary.inferred_frame_count} 次推理，检测 {summary.detected_hand_count} 只手",
        )

    def _is_canceled(self, job_id: str) -> bool:
        with self._condition:
            stopping = self._stop_requested
        return stopping or self.store.require(job_id).state is ProcessingJobState.CANCELING

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
