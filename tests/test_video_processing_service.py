from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from perception.video_processing import (
    ProcessingCanceled,
    ProcessingJob,
    ProcessingJobState,
    ProcessingRunSummary,
    VideoProcessingService,
)


class FakeRunner:
    def __init__(self, release: threading.Event | None = None) -> None:
        self.release = release
        self.started = threading.Event()

    def inspect(self, _session: Path, _clip_id: str | None) -> int:
        return 3

    def run(self, job, _session, output, *, progress, is_canceled):
        output.mkdir(parents=True)
        self.started.set()
        for index in range(1, 4):
            if self.release is not None:
                self.release.wait(0.02)
            if is_canceled():
                raise ProcessingCanceled()
            progress(index, 3)
        now = time.time_ns()
        (output / "run.json").write_text(
            json.dumps(
                {
                    "run_id": output.name,
                    "session_id": job.session_id,
                    "preset": {"inference_stride_frames": 1},
                    "started_at_unix_ns": now,
                }
            ),
            encoding="utf-8",
        )
        return ProcessingRunSummary(
            output.name, job.session_id, job.clip_id, output, 3, 3, 2, now, now
        )


def _wait_for(service: VideoProcessingService, state: ProcessingJobState) -> ProcessingJob:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        jobs = service.snapshot().jobs
        if jobs and jobs[0].state is state:
            return jobs[0]
        time.sleep(0.01)
    raise AssertionError(f"job did not reach {state.value}")


def test_service_runs_one_persistent_job_and_keeps_run_history(tmp_path: Path) -> None:
    session = tmp_path / ("a" * 32)
    session.mkdir()
    runner = FakeRunner()
    service = VideoProcessingService(tmp_path, runner=runner)  # type: ignore[arg-type]
    service.start()
    try:
        service.enqueue("a" * 32)
        completed = _wait_for(service, ProcessingJobState.COMPLETED)
        assert completed.progress_current == completed.progress_total == 3
        assert completed.run_id is not None
        assert service.list_runs("a" * 32)[0]["run_id"] == completed.run_id
    finally:
        service.close()


def test_service_cancels_active_job_without_deleting_its_run(tmp_path: Path) -> None:
    session = tmp_path / ("a" * 32)
    session.mkdir()
    release = threading.Event()
    runner = FakeRunner(release)
    service = VideoProcessingService(tmp_path, runner=runner)  # type: ignore[arg-type]
    service.start()
    try:
        job = service.enqueue("a" * 32)
        assert runner.started.wait(2)
        service.cancel(job.job_id)
        release.set()
        canceled = _wait_for(service, ProcessingJobState.CANCELED)
        assert canceled.run_id is not None
        assert (session / "derived" / "video-processing" / canceled.run_id).is_dir()
    finally:
        service.close()
