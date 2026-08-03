from __future__ import annotations

from perception.video_processing import (
    ProcessingJobState,
    ProcessingJobStore,
    ProcessingPreset,
    ProcessingResultStore,
)


def test_job_queue_persists_history_and_recovers_active_jobs(tmp_path) -> None:
    path = tmp_path / ".processing" / "jobs.sqlite3"
    store = ProcessingJobStore(path)
    first = store.enqueue("a" * 32, None, ProcessingPreset())
    claimed = store.claim_next()
    assert claimed is not None and claimed.job_id == first.job_id
    store.start_run(first.job_id, "run-1", 10)
    store.update_progress(first.job_id, 3, 10, "3/10")

    recovered = ProcessingJobStore(path)
    interrupted = recovered.require(first.job_id)
    assert interrupted.state is ProcessingJobState.INTERRUPTED
    assert interrupted.progress_current == 3

    retry = recovered.retry(first.job_id)
    assert retry.retry_of_job_id == first.job_id
    assert retry.state is ProcessingJobState.QUEUED
    assert len(recovered.list_jobs()) == 2


def test_queue_cancel_distinguishes_waiting_and_running_jobs(tmp_path) -> None:
    store = ProcessingJobStore(tmp_path / "jobs.sqlite3")
    waiting = store.enqueue("a" * 32, None, ProcessingPreset())
    assert store.request_cancel(waiting.job_id).state is ProcessingJobState.CANCELED

    running = store.enqueue("b" * 32, None, ProcessingPreset())
    store.claim_next()
    store.start_run(running.job_id, "run-2", 5)
    assert store.request_cancel(running.job_id).state is ProcessingJobState.CANCELING
    assert store.mark_canceled(running.job_id).state is ProcessingJobState.CANCELED


def test_results_are_addressed_by_clip_frame_and_session_time(tmp_path) -> None:
    path = tmp_path / "results.sqlite"
    store = ProcessingResultStore(path)
    result = {
        "session_id": "a" * 32,
        "sequence_id": "b" * 32,
        "frame_index": 6,
        "session_time_ns": 200,
        "hands": [],
    }
    store.put(result)

    reader = ProcessingResultStore(path, read_only=True)
    assert reader.result_for_frame("b" * 32, 6, 200) == result
    assert reader.result_for_frame("b" * 32, 7, 250) is None
    assert reader.result_for_frame("b" * 32, 7, 250, hold_previous_frames=1) == result
    assert reader.result_for_frame("b" * 32, 7, 199, hold_previous_frames=1) is None
