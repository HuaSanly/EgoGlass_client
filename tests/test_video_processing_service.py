from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from ui.processing import (
    ProcessingCanceled,
    ProcessingJob,
    ProcessingJobState,
    ProcessingJobStore,
    ProcessingPreset,
    ProcessingResultStore,
    ProcessingRunState,
    ProcessingRunSummary,
    VideoProcessingService,
)


class FakeRunner:
    def __init__(self, release: threading.Event | None = None) -> None:
        self.release = release
        self.started = threading.Event()
        self.release_gpu_calls = 0
        self.vio_inputs: list[tuple[object | None, str | None]] = []

    def inspect(self, _session: Path, _clip_id: str | None) -> int:
        return 3

    def run(
        self,
        job,
        _session,
        output,
        *,
        progress,
        is_canceled,
        vio_run_info=None,
        vio_error=None,
    ):
        self.vio_inputs.append((vio_run_info, vio_error))
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
                    "clip_id": job.clip_id,
                    "state": "completed",
                    "preset": {
                        "preset_id": job.preset.preset_id,
                        "display_name": job.preset.display_name,
                        "inference_stride_frames": 1,
                    },
                    "started_at_unix_ns": now,
                    "completed_at_unix_ns": now,
                    "input_frame_count": 3,
                    "inferred_frame_count": 3,
                    "detected_hand_count": 2,
                    "error": None,
                }
            ),
            encoding="utf-8",
        )
        ProcessingResultStore(output / "results.sqlite")
        return ProcessingRunSummary(
            output.name, job.session_id, job.clip_id, output, 3, 3, 2, now, now
        )

    def release_gpu(self) -> None:
        self.release_gpu_calls += 1


class BlockingInspectRunner(FakeRunner):
    def __init__(self) -> None:
        super().__init__()
        self.inspect_started = threading.Event()
        self.inspect_release = threading.Event()
        self.run_called = False

    def inspect(self, _session: Path, _clip_id: str | None) -> int:
        self.inspect_started.set()
        assert self.inspect_release.wait(timeout=2)
        return 3

    def run(self, *args, **kwargs):
        self.run_called = True
        return super().run(*args, **kwargs)


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
        run = service.list_runs("a" * 32)[0]
        assert run.run_id == completed.run_id
        assert run.is_viewable
        assert runner.release_gpu_calls == 1
    finally:
        service.close()


def test_service_runs_vio_as_part_of_the_same_processing_job(tmp_path: Path) -> None:
    session_id = "a" * 32
    (tmp_path / session_id).mkdir()
    runner = FakeRunner()
    vio_calls: list[tuple[str, str | None]] = []

    vio_run = object()

    def run_vio(session: str, *, clip_id: str | None = None) -> object:
        vio_calls.append((session, clip_id))
        return vio_run

    service = VideoProcessingService(
        tmp_path,
        runner=runner,  # type: ignore[arg-type]
        offline_vio_runner=run_vio,
    )
    service.start()
    try:
        job = service.enqueue(session_id, clip_id="b" * 32)
        completed = _wait_for(service, ProcessingJobState.COMPLETED)
        assert completed.job_id == job.job_id
        assert vio_calls == [(session_id, "b" * 32)]
        assert runner.vio_inputs == [(vio_run, None)]
        assert completed.detail == "处理完成 3 帧手部追踪与 SLAM/VIO"
    finally:
        service.close()


def test_service_keeps_viewable_partial_result_when_vio_fails(tmp_path: Path) -> None:
    session_id = "a" * 32
    (tmp_path / session_id).mkdir()

    def run_vio(_session: str, *, clip_id: str | None = None) -> None:
        raise RuntimeError(f"VIO failed for {clip_id}")

    service = VideoProcessingService(
        tmp_path,
        runner=FakeRunner(),  # type: ignore[arg-type]
        offline_vio_runner=run_vio,
    )
    service.start()
    try:
        service.enqueue(session_id, clip_id="b" * 32)
        partial = _wait_for(service, ProcessingJobState.PARTIAL)
        assert partial.detail == "部分完成: VIO failed for " + "b" * 32
        run = service.list_runs(session_id)[0]
        assert run.is_viewable
        assert run.state is ProcessingRunState.PARTIAL
        assert service.runner.vio_inputs == [
            (None, "VIO failed for " + "b" * 32)
        ]
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


def test_service_cancel_during_preparation_reaches_terminal_state(tmp_path: Path) -> None:
    (tmp_path / ("a" * 32)).mkdir()
    runner = BlockingInspectRunner()
    service = VideoProcessingService(tmp_path, runner=runner)  # type: ignore[arg-type]
    service.start()
    try:
        job = service.enqueue("a" * 32)
        assert runner.inspect_started.wait(timeout=2)
        assert service.cancel(job.job_id).state is ProcessingJobState.CANCELING
        runner.inspect_release.set()
        canceled = _wait_for(service, ProcessingJobState.CANCELED)

        assert canceled.run_id is None
        assert not runner.run_called
        assert service._thread is not None and service._thread.is_alive()
    finally:
        runner.inspect_release.set()
        service.close()


def test_auto_enqueue_is_persistent_and_idempotent(tmp_path: Path) -> None:
    (tmp_path / ("a" * 32)).mkdir()
    service = VideoProcessingService(tmp_path, runner=FakeRunner())  # type: ignore[arg-type]
    assert not service.snapshot().auto_enqueue_on_session_complete
    service.set_auto_enqueue(True)
    first = service.enqueue_completed_session("a" * 32)
    second = service.enqueue_completed_session("a" * 32)

    assert first is not None
    assert second is None
    restarted = VideoProcessingService(tmp_path, runner=FakeRunner())  # type: ignore[arg-type]
    assert restarted.snapshot().auto_enqueue_on_session_complete


def test_new_jobs_only_accept_and_use_the_per_frame_quality_preset(tmp_path: Path) -> None:
    (tmp_path / ("a" * 32)).mkdir()
    service = VideoProcessingService(tmp_path, runner=FakeRunner())  # type: ignore[arg-type]

    assert service.set_default_preset("hand-tracking-quality") == "hand-tracking-quality"
    job = service.enqueue("a" * 32)
    assert job.preset.preset_id == "hand-tracking-quality"
    assert job.preset.inference_stride_frames == 1
    assert service.snapshot().default_preset_id == "hand-tracking-quality"

    restarted = VideoProcessingService(tmp_path, runner=FakeRunner())  # type: ignore[arg-type]
    assert restarted.snapshot().default_preset_id == "hand-tracking-quality"
    try:
        restarted.set_default_preset("hand-tracking-preview")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown preset should be rejected")


def test_object_model_configuration_is_frozen_when_job_is_submitted(tmp_path: Path) -> None:
    session_id = "a" * 32
    (tmp_path / session_id).mkdir()
    runner = FakeRunner()
    runner.object_tracking_config_path = Path("config/object-tracking.yaml").resolve()
    service = VideoProcessingService(tmp_path, runner=runner)  # type: ignore[arg-type]

    job = service.enqueue(session_id, task_profile_id="default-manipulation")

    snapshot = json.loads(job.configuration_snapshot_json)
    assert snapshot["object_tracking"]["dino_model_revision"]
    assert snapshot["object_tracking"]["cotracker_code_revision"]
    assert dict(job.configuration_sha256_by_file)["object-tracking.yaml"]
    assert json.loads(job.task_profile_snapshot_json)["profile_id"] == "default-manipulation"


def test_object_model_snapshot_round_trips_strict_profiles() -> None:
    from object_tracking import load_object_tracking_config
    from object_tracking.config import ObjectTrackingConfig

    config = load_object_tracking_config(Path("config/object-tracking.yaml"))
    snapshot = json.dumps(config.model_dump(mode="json"))

    restored = ObjectTrackingConfig.model_validate_json(snapshot)

    assert tuple(profile.profile_id for profile in restored.profiles) == tuple(
        profile.profile_id for profile in config.profiles
    )


def test_historical_non_quality_preset_remains_readable_and_retryable(
    tmp_path: Path,
) -> None:
    session_id = "a" * 32
    (tmp_path / session_id).mkdir()
    store = ProcessingJobStore(tmp_path / ".processing" / "jobs.sqlite3")
    historical_preset = ProcessingPreset(
        preset_id="hand-tracking-preview",
        display_name="Historical preview",
        inference_stride_frames=3,
    )
    historical = store.enqueue(session_id, None, historical_preset)
    store.request_cancel(historical.job_id)

    service = VideoProcessingService(tmp_path, runner=FakeRunner())  # type: ignore[arg-type]
    loaded = service.store.require(historical.job_id)
    retried = service.retry(historical.job_id)

    assert loaded.preset == historical_preset
    assert retried.preset == historical_preset
    assert retried.retry_of_job_id == historical.job_id
    with pytest.raises(KeyError):
        service.enqueue(session_id, preset_id="hand-tracking-preview")


def test_completed_run_with_invalid_result_store_is_not_viewable(tmp_path: Path) -> None:
    session = tmp_path / ("a" * 32)
    run = session / "derived" / "video-processing" / "run-a"
    run.mkdir(parents=True)
    now = time.time_ns()
    run.joinpath("run.json").write_text(
        json.dumps(
            {
                "run_id": "run-a",
                "session_id": "a" * 32,
                "clip_id": "b" * 32,
                "state": "completed",
                "preset": {
                    "preset_id": "hand-tracking-quality",
                    "display_name": "手部追踪 · 质量优先",
                    "inference_stride_frames": 1,
                },
                "started_at_unix_ns": now,
                "completed_at_unix_ns": now,
                "input_frame_count": 10,
                "inferred_frame_count": 10,
                "detected_hand_count": 2,
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    run.joinpath("results.sqlite").write_bytes(b"not sqlite")

    info = VideoProcessingService(tmp_path, runner=FakeRunner()).list_runs("a" * 32)[0]

    assert not info.is_viewable
    assert info.unavailable_reason
    assert not info.covers_clip("b" * 32)


def test_result_lookup_adds_validated_object_overlays(tmp_path: Path) -> None:
    session_id = "a" * 32
    clip_id = "b" * 32
    run = tmp_path / session_id / "derived" / "video-processing" / "run-objects"
    objects = run / "objects"
    masks = objects / "masks"
    masks.mkdir(parents=True)
    mask_path = masks / "obj1-000003.png"
    mask_path.write_bytes(b"mask")
    outside = run / "outside.png"
    outside.write_bytes(b"outside")
    now = time.time_ns()
    (run / "run.json").write_text(
        json.dumps(
            {
                "run_id": run.name,
                "session_id": session_id,
                "clip_id": clip_id,
                "state": "completed",
                "preset": {
                    "preset_id": "hand-tracking-quality",
                    "display_name": "quality",
                    "inference_stride_frames": 1,
                },
                "started_at_unix_ns": now,
                "completed_at_unix_ns": now,
                "input_frame_count": 1,
                "inferred_frame_count": 1,
                "detected_hand_count": 0,
            }
        ),
        encoding="utf-8",
    )
    store = ProcessingResultStore(run / "results.sqlite")
    store.put_final(
        {
            "session_id": session_id,
            "sequence_id": clip_id,
            "frame_index": 3,
            "session_time_ns": 100,
            "hands": [],
        }
    )
    store.put_raw(
        {
            "session_id": session_id,
            "sequence_id": clip_id,
            "frame_index": 3,
            "session_time_ns": 100,
            "hands": [{"handedness": "right", "confidence": 0.25}],
        }
    )
    (objects / "object-result.json").write_text(
        json.dumps(
            {
                "masks": [
                    {
                        "object_id": "obj1",
                        "clip_id": clip_id,
                        "frame_index": 3,
                        "mask_relative_path": "masks/obj1-000003.png",
                    },
                    {
                        "object_id": "obj2",
                        "clip_id": clip_id,
                        "frame_index": 3,
                        "mask_relative_path": "../outside.png",
                    },
                ],
                "tracks": [
                    {
                        "object_id": "obj1",
                        "clip_id": clip_id,
                        "frame_indices": [3],
                        "points_xy_px": [[[10.0, 20.0]]],
                        "visibility": [[1.0]],
                    }
                ],
                "triangulations": [{"object_id": "obj1", "valid_point_count": 3}],
                "poses": [{"object_id": "obj1", "clip_id": clip_id, "frame_index": 3}],
            }
        ),
        encoding="utf-8",
    )
    service = VideoProcessingService(tmp_path, runner=FakeRunner())  # type: ignore[arg-type]

    result = service.result_for_frame(session_id, run.name, clip_id, 3, 100)

    assert result is not None
    overlays = result["object_overlays"]
    assert isinstance(overlays, list) and len(overlays) == 2
    first = next(item for item in overlays if item["object_id"] == "obj1")
    second = next(item for item in overlays if item["object_id"] == "obj2")
    assert first["mask_path"] == str(mask_path.resolve())
    assert first["track"] == {"points_xy_px": [[10.0, 20.0]], "visibility": [1.0]}
    assert second["mask_path"] is None
    raw = service.result_for_frame(
        session_id,
        run.name,
        clip_id,
        3,
        100,
        hand_result_kind="raw",
    )
    assert raw is not None and raw["hands"][0]["confidence"] == 0.25
    assert raw["object_overlays"] == overlays
    with pytest.raises(ValueError, match="hand_result_kind"):
        service.result_for_frame(
            session_id,
            run.name,
            clip_id,
            3,
            100,
            hand_result_kind="unsupported",
        )

    (objects / "object-result.json").write_text("{broken", encoding="utf-8")
    restarted = VideoProcessingService(tmp_path, runner=FakeRunner())  # type: ignore[arg-type]
    assert "object_overlays" not in restarted.result_for_frame(
        session_id, run.name, clip_id, 3, 100
    )


def test_gpu_claim_failure_marks_job_failed_and_releases_ownership(tmp_path: Path) -> None:
    (tmp_path / ("a" * 32)).mkdir()
    ownership_changes: list[bool] = []
    ownership_released = threading.Event()

    def change_ownership(active: bool) -> None:
        ownership_changes.append(active)
        if active:
            raise RuntimeError("GPU ownership unavailable")
        ownership_released.set()

    service = VideoProcessingService(
        tmp_path,
        runner=FakeRunner(),  # type: ignore[arg-type]
        on_gpu_job_changed=change_ownership,
    )
    service.start()
    try:
        service.enqueue("a" * 32)
        failed = _wait_for(service, ProcessingJobState.FAILED)

        assert failed.detail == "GPU ownership unavailable"
        assert ownership_released.wait(timeout=2)
        assert ownership_changes == [True, False]
        assert service._thread is not None and service._thread.is_alive()
    finally:
        service.close()
