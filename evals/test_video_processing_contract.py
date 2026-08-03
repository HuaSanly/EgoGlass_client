from pathlib import Path

from perception.video_processing import ProcessingJobState, ProcessingJobStore, ProcessingPreset


def test_case_vp_001_restart_never_silently_resumes_gpu_work(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    store = ProcessingJobStore(database)
    job = store.enqueue("a" * 32, None, ProcessingPreset())
    store.claim_next()
    store.start_run(job.job_id, "run-a", 100)
    store.update_progress(job.job_id, 41, 100, "processing")

    restarted = ProcessingJobStore(database)
    recovered = restarted.require(job.job_id)
    assert recovered.state is ProcessingJobState.INTERRUPTED
    assert recovered.progress_current == 41
    assert restarted.claim_next() is None
