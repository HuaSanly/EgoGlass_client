from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock

from perception.configuration import ConfigurationService
from perception.runtime import HandTrackingRuntime, HandTrackingRuntimeConfig
from ui.runtime import RuntimeConfig, UnifiedRuntimeHost

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config_copy(tmp_path: Path) -> Path:
    destination = tmp_path / "config"
    destination.mkdir()
    for name in (
        "client-runtime.yaml",
        "perception-runtime.yaml",
        "sensor-preprocessing.yaml",
        "live-hand-tracking.yaml",
        "offline-hand-tracking.yaml",
        "video-processing.yaml",
        "sensor-calibration-640x480-sample.json",
    ):
        shutil.copy2(PROJECT_ROOT / "config" / name, destination / name)
    return destination


def test_runtime_applies_live_and_future_job_configuration(tmp_path: Path) -> None:
    configuration = ConfigurationService(
        _config_copy(tmp_path),
        recordings_root=tmp_path / "recordings",
    )
    host = UnifiedRuntimeHost(
        RuntimeConfig(recordings_root=tmp_path / "recordings"),
        configuration_service=configuration,
    )
    host.perception.apply_runtime_config = AsyncMock()  # type: ignore[method-assign]
    host.perception.reload_tracker_configuration = AsyncMock()  # type: ignore[method-assign]
    host.perception.status = AsyncMock(  # type: ignore[method-assign]
        return_value={"offline_processing": False}
    )
    try:
        saved = configuration.save(
            {
                "live_hand_tracking": {
                    "runtime": {"enabled": True, "max_live_inference_fps": 9.0},
                    "algorithm": {"minimum_hand_confidence": 0.45},
                },
                "offline_hand_tracking": {"minimum_hand_confidence": 0.65},
                "video_processing": {
                    "default_preset_id": "hand-tracking-quality",
                    "auto_enqueue_on_session_complete": True,
                },
            }
        )

        result = asyncio.run(
            host._apply_configuration(configuration.apply_request(saved))
        )

        applied = host.perception.apply_runtime_config.await_args.args[0]
        assert applied.enabled
        assert applied.max_live_inference_fps == 9.0
        host.perception.reload_tracker_configuration.assert_awaited_once()
        processing = host.video_processing.snapshot()
        assert processing.default_preset_id == "hand-tracking-quality"
        assert processing.auto_enqueue_on_session_complete
        assert result.immediate_applied == ("live_hand_tracking", "video_processing")
        assert result.pending_next_task == ("offline_hand_tracking",)
        session_id = "a" * 32
        (tmp_path / "recordings" / session_id).mkdir()
        job = host.video_processing.enqueue(session_id)
        assert job.configuration_revision == saved.provenance.revision
        assert dict(job.configuration_sha256_by_file) == dict(
            saved.provenance.sha256_by_file
        )
        assert '"sensor_calibration"' in job.configuration_snapshot_json
        assert '"offline_hand_tracking"' in job.configuration_snapshot_json
        assert '"live_hand_tracking"' not in job.configuration_snapshot_json
        assert job.preset.inference_stride_frames == 1
    finally:
        host.stop()


def test_runtime_wires_live_and_offline_profiles_to_separate_consumers(
    tmp_path: Path,
) -> None:
    config = _config_copy(tmp_path)
    configuration = ConfigurationService(
        config,
        recordings_root=tmp_path / "recordings",
    )
    host = UnifiedRuntimeHost(
        RuntimeConfig(recordings_root=tmp_path / "recordings"),
        configuration_service=configuration,
    )
    try:
        assert host.perception.hand_tracking_config_path == (
            config / "live-hand-tracking.yaml"
        ).resolve()
        assert host.video_processing.runner.offline_hand_tracking_config_path == (
            config / "offline-hand-tracking.yaml"
        ).resolve()
    finally:
        host.stop()


def test_profile_changes_do_not_cross_runtime_or_queued_task_boundaries(
    tmp_path: Path,
) -> None:
    config = _config_copy(tmp_path)
    recordings = tmp_path / "recordings"
    session_id = "a" * 32
    (recordings / session_id).mkdir(parents=True)
    configuration = ConfigurationService(config, recordings_root=recordings)
    host = UnifiedRuntimeHost(
        RuntimeConfig(recordings_root=recordings),
        configuration_service=configuration,
    )
    host.perception.apply_runtime_config = AsyncMock()  # type: ignore[method-assign]
    host.perception.reload_tracker_configuration = AsyncMock()  # type: ignore[method-assign]
    host.perception.status = AsyncMock(  # type: ignore[method-assign]
        return_value={"offline_processing": False}
    )
    try:
        initial = host.video_processing.enqueue(session_id)
        initial_snapshot = json.loads(initial.configuration_snapshot_json)

        offline_saved = configuration.save(
            {"offline_hand_tracking": {"minimum_hand_confidence": 0.65}}
        )
        offline_result = asyncio.run(
            host._apply_configuration(configuration.apply_request(offline_saved))
        )
        host.perception.apply_runtime_config.assert_not_awaited()
        host.perception.reload_tracker_configuration.assert_not_awaited()
        assert offline_result.pending_next_task == ("offline_hand_tracking",)
        assert json.loads(initial.configuration_snapshot_json) == initial_snapshot
        assert initial_snapshot["offline_hand_tracking"]["minimum_hand_confidence"] == 0.3

        changed_offline = host.video_processing.enqueue(session_id)
        changed_offline_snapshot = json.loads(
            changed_offline.configuration_snapshot_json
        )
        assert (
            changed_offline_snapshot["offline_hand_tracking"][
                "minimum_hand_confidence"
            ]
            == 0.65
        )

        live_saved = configuration.save(
            {"live_hand_tracking": {"algorithm": {"minimum_hand_confidence": 0.7}}}
        )
        asyncio.run(host._apply_configuration(configuration.apply_request(live_saved)))
        host.perception.reload_tracker_configuration.assert_awaited_once()
        after_live = host.video_processing.enqueue(session_id)
        assert json.loads(after_live.configuration_snapshot_json) == changed_offline_snapshot
    finally:
        host.stop()


def test_runtime_does_not_interrupt_offline_gpu_job_for_model_reload(
    tmp_path: Path,
) -> None:
    configuration = ConfigurationService(
        _config_copy(tmp_path),
        recordings_root=tmp_path / "recordings",
    )
    host = UnifiedRuntimeHost(
        RuntimeConfig(recordings_root=tmp_path / "recordings"),
        configuration_service=configuration,
    )
    host.perception.apply_runtime_config = AsyncMock()  # type: ignore[method-assign]
    host.perception.reload_tracker_configuration = AsyncMock()  # type: ignore[method-assign]
    host.perception.status = AsyncMock(  # type: ignore[method-assign]
        return_value={"offline_processing": True}
    )
    try:
        saved = configuration.save(
            {
                "live_hand_tracking": {
                    "algorithm": {"minimum_hand_confidence": 0.5}
                }
            }
        )

        result = asyncio.run(
            host._apply_configuration(configuration.apply_request(saved))
        )

        host.perception.reload_tracker_configuration.assert_not_awaited()
        assert result.warnings == (
            "离线任务正在占用 GPU，实时手部模型配置将在实时推理恢复时加载",
        )
    finally:
        host.stop()


def test_hand_tracking_runtime_updates_live_fps_without_restarting(
    tmp_path: Path,
) -> None:
    runtime_path = tmp_path / "perception-runtime.yaml"
    shutil.copy2(PROJECT_ROOT / "config" / "perception-runtime.yaml", runtime_path)
    runtime = HandTrackingRuntime(runtime_config_path=runtime_path)
    async def exercise() -> dict[str, object]:
        try:
            await runtime.apply_runtime_config(
                HandTrackingRuntimeConfig(
                    schema_version="1.0",
                    enabled=True,
                    max_live_inference_fps=11.0,
                )
            )
            return await runtime.status()
        finally:
            await runtime.close()

    status = asyncio.run(exercise())
    assert status["live_enabled"] is True
    assert runtime.config.max_live_inference_fps == 11.0
