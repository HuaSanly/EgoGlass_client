import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path

from av import VideoFrame

from hand_tracking.runtime import HandTrackingRuntime, LiveHandTrackingFrame
from tests.test_video_processing_service import BlockingInspectRunner
from ui.processing import (
    ProcessingJobState,
    ProcessingJobStore,
    ProcessingPreset,
    ProcessingResultStore,
    VideoProcessingService,
    cleanup_legacy_hand_tracking,
)


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


def test_case_vp_002_legacy_cleanup_preserves_raw_media_bytes(tmp_path: Path) -> None:
    session = tmp_path / "session"
    legacy = session / "perception" / "hand-tracking" / "run"
    media = session / "media" / "clip.mp4"
    legacy.mkdir(parents=True)
    media.parent.mkdir(parents=True)
    legacy.joinpath("result.jsonl").write_text("{}\n", encoding="utf-8")
    media.write_bytes(b"immutable-media-evidence")

    report = cleanup_legacy_hand_tracking(
        tmp_path,
        tmp_path / ".processing" / "audit.json",
        apply=True,
    )

    assert report.media_unchanged
    assert media.read_bytes() == b"immutable-media-evidence"
    assert not legacy.parent.exists()


def test_case_vp_003_long_job_cancel_and_retry_preserve_attempt_history(
    tmp_path: Path,
) -> None:
    store = ProcessingJobStore(tmp_path / "jobs.sqlite3")
    original = store.enqueue("a" * 32, None, ProcessingPreset())
    store.claim_next()
    store.start_run(original.job_id, "run-long", 100_000)
    store.update_progress(original.job_id, 47_500, 100_000, "processing")
    assert store.request_cancel(original.job_id).state is ProcessingJobState.CANCELING
    canceled = store.mark_canceled(original.job_id)
    retry = store.retry(original.job_id)

    assert canceled.progress_current == 47_500
    assert canceled.state is ProcessingJobState.CANCELED
    assert retry.job_id != canceled.job_id
    assert retry.retry_of_job_id == canceled.job_id
    assert retry.state is ProcessingJobState.QUEUED


def test_case_vp_004_offline_gpu_ownership_has_no_live_inference_overlap(
    tmp_path: Path,
) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text(
        'schema_version: "1.0"\nenabled: true\nmax_live_inference_fps: 60.0\n',
        encoding="utf-8",
    )
    live_started = threading.Event()
    live_release = threading.Event()

    async def scenario() -> None:
        runtime = HandTrackingRuntime(runtime_config_path=config)

        def process(_frame: LiveHandTrackingFrame) -> object:
            live_started.set()
            assert live_release.wait(timeout=2)
            return _EvalResult()

        runtime._process_live_frame = process  # type: ignore[method-assign]
        await runtime.submit_live_frame(
            LiveHandTrackingFrame(
                session_id="a" * 32,
                connection_session_id="b" * 32,
                frame_index=1,
                received_at_client_monotonic_ns=1,
                decoded_frame=VideoFrame(8, 6, "yuv420p"),
            )
        )
        assert await asyncio.to_thread(live_started.wait, 2)
        offline_claim = asyncio.create_task(runtime.set_offline_processing(True))
        await asyncio.sleep(0)
        assert not offline_claim.done()
        live_release.set()
        await asyncio.wait_for(offline_claim, timeout=2)
        assert (await runtime.status())["offline_processing"] is True
        await runtime.close()

    asyncio.run(scenario())


class _EvalResult:
    def to_json_dict(self) -> dict[str, object]:
        return {"schema_version": "1.0", "frame_index": 1, "hands": []}


def test_case_vp_006_cancel_during_preparation_never_sticks_canceling(
    tmp_path: Path,
) -> None:
    (tmp_path / ("a" * 32)).mkdir()
    runner = BlockingInspectRunner()
    service = VideoProcessingService(tmp_path, runner=runner)  # type: ignore[arg-type]
    service.start()
    try:
        job = service.enqueue("a" * 32)
        assert runner.inspect_started.wait(timeout=2)
        service.cancel(job.job_id)
        runner.inspect_release.set()
        deadline = time.monotonic() + 2
        while service.store.require(job.job_id).state is not ProcessingJobState.CANCELED:
            if time.monotonic() >= deadline:
                raise AssertionError("preparing cancellation did not become terminal")
            time.sleep(0.005)

        assert service.store.require(job.job_id).state is ProcessingJobState.CANCELED
        assert not runner.run_called
    finally:
        runner.inspect_release.set()
        service.close()


def test_case_vp_007_completed_artifact_wins_last_moment_cancel(tmp_path: Path) -> None:
    store = ProcessingJobStore(tmp_path / "jobs.sqlite3")
    job = store.enqueue("a" * 32, None, ProcessingPreset())
    store.claim_next()
    store.start_run(job.job_id, "run-finished", 10)
    store.update_progress(job.job_id, 10, 10, "run artifact committed")
    store.request_cancel(job.job_id)

    completed = store.complete(job.job_id)

    assert completed.state is ProcessingJobState.COMPLETED
    assert completed.progress_fraction == 1.0


def test_case_vp_008_only_valid_completed_runs_are_viewable(tmp_path: Path) -> None:
    session_id = "a" * 32
    clip_id = "b" * 32
    session = tmp_path / session_id
    good = session / "derived" / "video-processing" / "good-run"
    bad = session / "derived" / "video-processing" / "bad-run"
    good.mkdir(parents=True)
    bad.mkdir()
    now = time.time_ns()
    base = {
        "session_id": session_id,
        "clip_id": clip_id,
        "state": "completed",
        "preset": {
            "preset_id": "hand-tracking-quality",
            "display_name": "手部追踪 · 质量优先",
            "inference_stride_frames": 1,
        },
        "started_at_unix_ns": now,
        "completed_at_unix_ns": now,
        "input_frame_count": 1,
        "inferred_frame_count": 1,
        "detected_hand_count": 0,
        "error": None,
    }
    good.joinpath("run.json").write_text(
        json.dumps({**base, "run_id": "good-run"}), encoding="utf-8"
    )
    bad.joinpath("run.json").write_text(
        json.dumps({**base, "run_id": "bad-run"}), encoding="utf-8"
    )
    ProcessingResultStore(good / "results.sqlite")

    runs = VideoProcessingService(tmp_path).list_runs(session_id)
    by_id = {run.run_id: run for run in runs}

    assert by_id["good-run"].covers_clip(clip_id)
    assert not by_id["bad-run"].is_viewable


def test_case_vp_009_startup_recovers_mistagged_queue_without_data_loss(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    store = ProcessingJobStore(database)
    job = store.enqueue(
        "a" * 32,
        None,
        ProcessingPreset(),
        configuration_revision=3,
        configuration_snapshot_json='{"offline_hand_tracking": {"detector": "vitpose"}}',
    )
    store.set_setting("auto_enqueue", "false")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE metadata SET value = '6' WHERE key = 'schema_version'"
        )

    reopened = ProcessingJobStore(database)
    assert reopened.require(job.job_id).configuration_revision == 3
    assert reopened.setting("auto_enqueue", "missing") == "false"
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone() == ("5",)
