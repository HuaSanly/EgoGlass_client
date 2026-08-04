from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
import yaml

from perception.configuration import (
    ClientRuntimeConfig,
    ConfigImpact,
    ConfigurationError,
    ConfigurationService,
    ConfigurationValidationError,
)
from perception.video_processing import ProcessingJobStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config_copy(tmp_path: Path) -> Path:
    destination = tmp_path / "config"
    destination.mkdir()
    for name in (
        "client-runtime.yaml",
        "perception-runtime.yaml",
        "sensor-preprocessing.yaml",
        "hand-tracking.yaml",
        "video-processing.yaml",
        "sensor-calibration-640x480-sample.json",
    ):
        shutil.copy2(PROJECT_ROOT / "config" / name, destination / name)
    return destination


def test_service_exposes_four_typed_modules_backed_by_five_files(tmp_path: Path) -> None:
    config = _config_copy(tmp_path)
    service = ConfigurationService(config, recordings_root=tmp_path / "recordings")

    snapshot = service.snapshot()

    assert tuple(module.module_id for module in snapshot.modules) == (
        "client_runtime",
        "sensor_preprocessing",
        "hand_tracking",
        "video_processing",
    )
    assert not snapshot.dirty
    hand = snapshot.require_module("hand_tracking")
    assert tuple(path.name for path in hand.source_paths) == (
        "perception-runtime.yaml",
        "hand-tracking.yaml",
    )
    assert hand.values["runtime"]["max_live_inference_fps"] == 6.0  # type: ignore[index]
    assert hand.values["algorithm"]["enable_cuda_amp"] is True  # type: ignore[index]
    assert set(service.provenance().sha256_by_file) == {
        "client-runtime.yaml",
        "perception-runtime.yaml",
        "sensor-preprocessing.yaml",
        "hand-tracking.yaml",
        "video-processing.yaml",
    }


def test_client_runtime_loader_resolves_recordings_root_from_config_directory(
    tmp_path: Path,
) -> None:
    config = _config_copy(tmp_path)

    loaded = ClientRuntimeConfig.load(config / "client-runtime.yaml")

    assert loaded.recordings_root == (tmp_path / "local-data" / "recordings").resolve()
    assert ClientRuntimeConfig.load(config / "client-runtime.yaml").recordings_root == (
        tmp_path / "local-data" / "recordings"
    ).resolve()


def test_validation_rejects_unknown_fields_and_invalid_calibration(tmp_path: Path) -> None:
    config = _config_copy(tmp_path)
    service = ConfigurationService(config, recordings_root=tmp_path / "recordings")

    issues = service.validate({"client_runtime": {"unknown": True}})
    assert len(issues) == 1
    assert issues[0].module_id == "client_runtime"
    assert issues[0].field_path == "unknown"

    broken = config / "broken-calibration.json"
    broken.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")
    issues = service.validate(
        {"sensor_preprocessing": {"calibration_file": broken.name}}
    )
    assert len(issues) == 1
    assert issues[0].module_id == "sensor_preprocessing"
    assert issues[0].field_path == "calibration_file"

    with pytest.raises(ConfigurationValidationError):
        service.save({"client_runtime": {"port": 70_000}})
    assert yaml.safe_load((config / "client-runtime.yaml").read_text(encoding="utf-8"))[
        "port"
    ] == 8770


def test_stage_discard_and_restore_defaults_do_not_write_early(tmp_path: Path) -> None:
    config = _config_copy(tmp_path)
    service = ConfigurationService(config, recordings_root=tmp_path / "recordings")
    original = (config / "client-runtime.yaml").read_bytes()

    service.stage({"client_runtime": {"port": 9000}})
    assert service.snapshot().dirty
    assert service.snapshot().require_module("client_runtime").values["port"] == 9000
    assert (config / "client-runtime.yaml").read_bytes() == original

    service.discard()
    assert not service.snapshot().dirty
    service.stage({"client_runtime": {"port": 9000}})
    service.restore_defaults("client_runtime")
    assert service.snapshot().require_module("client_runtime").values["port"] == 8770


def test_save_creates_backup_and_persists_revision_and_hashes(tmp_path: Path) -> None:
    config = _config_copy(tmp_path)
    recordings = tmp_path / "recordings"
    service = ConfigurationService(config, recordings_root=recordings)
    original = (config / "client-runtime.yaml").read_bytes()
    old_hash = service.provenance().sha256_by_file["client-runtime.yaml"]

    result = service.save({"client_runtime": {"port": 8870}})

    assert result.changed_modules == ("client_runtime",)
    assert result.pending_restart == ("client_runtime",)
    assert result.provenance.revision == 1
    assert result.provenance.sha256_by_file["client-runtime.yaml"] != old_hash
    assert (config / "client-runtime.yaml.bak").read_bytes() == original
    assert yaml.safe_load((config / "client-runtime.yaml").read_text(encoding="utf-8"))[
        "port"
    ] == 8870
    assert ConfigurationService(config, recordings_root=recordings).snapshot().revision == 1


def test_failed_atomic_replace_preserves_file_and_dirty_form(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config_copy(tmp_path)
    service = ConfigurationService(config, recordings_root=tmp_path / "recordings")
    client_path = config / "client-runtime.yaml"
    original = client_path.read_bytes()
    real_replace = os.replace

    def fail_target(source: str | bytes | Path, target: str | bytes | Path) -> None:
        if Path(target) == client_path:
            raise OSError("simulated replace failure")
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_target)

    with pytest.raises(ConfigurationError):
        service.save({"client_runtime": {"port": 8870}})

    assert client_path.read_bytes() == original
    assert service.snapshot().dirty
    assert service.snapshot().require_module("client_runtime").values["port"] == 8870


def test_multi_file_save_rolls_back_files_committed_before_later_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config_copy(tmp_path)
    service = ConfigurationService(config, recordings_root=tmp_path / "recordings")
    client_path = config / "client-runtime.yaml"
    sensor_path = config / "sensor-preprocessing.yaml"
    originals = {
        client_path: client_path.read_bytes(),
        sensor_path: sensor_path.read_bytes(),
    }
    real_replace = os.replace

    def fail_sensor(source: str | bytes | Path, target: str | bytes | Path) -> None:
        if Path(target) == sensor_path:
            raise OSError("simulated second-file replace failure")
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_sensor)
    with pytest.raises(ConfigurationError):
        service.save(
            {
                "client_runtime": {"port": 8870},
                "sensor_preprocessing": {"image": {"undistort": False}},
            }
        )

    assert client_path.read_bytes() == originals[client_path]
    assert sensor_path.read_bytes() == originals[sensor_path]
    assert service.snapshot().dirty


def test_legacy_processing_metadata_migrates_once_and_stays_compatible(
    tmp_path: Path,
) -> None:
    config = _config_copy(tmp_path)
    database = tmp_path / "recordings" / ".processing" / "jobs.sqlite3"
    store = ProcessingJobStore(database)
    store.set_setting("auto_enqueue", "true")
    store.set_setting("default_preset_id", "hand-tracking-preview")

    service = ConfigurationService(config, jobs_database_path=database)
    values = service.snapshot().require_module("video_processing").values
    assert values["auto_enqueue_on_session_complete"] is True
    assert values["default_preset_id"] == "hand-tracking-preview"

    service.save(
        {
            "video_processing": {
                "auto_enqueue_on_session_complete": False,
                "default_preset_id": "hand-tracking-balanced",
            }
        }
    )
    assert store.setting("auto_enqueue", "missing") == "false"
    assert store.setting("default_preset_id", "missing") == "hand-tracking-balanced"

    store.set_setting("default_preset_id", "hand-tracking-preview")
    restarted = ConfigurationService(config, jobs_database_path=database)
    assert (
        restarted.snapshot().require_module("video_processing").values[
            "default_preset_id"
        ]
        == "hand-tracking-balanced"
    )


def test_change_impacts_and_runtime_apply_request_are_explicit(tmp_path: Path) -> None:
    config = _config_copy(tmp_path)
    service = ConfigurationService(config, recordings_root=tmp_path / "recordings")

    result = service.save(
        {
            "sensor_preprocessing": {"image": {"undistort": False}},
            "hand_tracking": {
                "runtime": {"enabled": True},
                "algorithm": {"minimum_hand_confidence": 0.4},
            },
        }
    )

    assert result.immediate_applied == ("hand_tracking",)
    assert result.pending_next_task == ("hand_tracking",)
    assert result.pending_next_session == ("sensor_preprocessing",)
    assert {(change.field_path, change.impact) for change in result.changes} == {
        ("image.undistort", ConfigImpact.NEXT_SESSION),
        ("runtime.enabled", ConfigImpact.IMMEDIATE),
        ("algorithm.minimum_hand_confidence", ConfigImpact.NEXT_TASK),
    }
    request = service.apply_request(result)
    assert request.revision == 1
    assert set(request.values) == {"sensor_preprocessing", "hand_tracking"}
