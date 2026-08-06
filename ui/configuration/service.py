from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml
from pydantic import ValidationError

from hand_tracking import HandTrackingConfig
from sensor_preprocessing import (
    SensorCalibration,
    SensorPreprocessingConfig,
)

from .contracts import (
    ConfigApplyResult,
    ConfigChange,
    ConfigImpact,
    ConfigSnapshot,
    ConfigurationApplyRequest,
    ConfigurationError,
    ConfigurationProvenance,
    ConfigurationValidationError,
    ModuleConfigSnapshot,
    ValidationIssue,
)
from .models import ClientRuntimeConfig, PerceptionRuntimeConfig, VideoProcessingConfig

_MODULE_ORDER = (
    "client_runtime",
    "sensor_preprocessing",
    "live_hand_tracking",
    "offline_hand_tracking",
    "video_processing",
)
_DISPLAY_NAMES = {
    "client_runtime": "客户端与网关",
    "sensor_preprocessing": "传感器预处理",
    "live_hand_tracking": "实时手部追踪",
    "offline_hand_tracking": "离线手部追踪",
    "video_processing": "离线视频处理",
}
_FILE_NAMES = {
    "client_runtime": "client-runtime.yaml",
    "perception_runtime": "perception-runtime.yaml",
    "sensor_preprocessing": "sensor-preprocessing.yaml",
    "live_hand_tracking": "live-hand-tracking.yaml",
    "offline_hand_tracking": "offline-hand-tracking.yaml",
    "video_processing": "video-processing.yaml",
}
_LEGACY_HAND_TRACKING_FILE = "hand-tracking.yaml"

_CLIENT_DEFAULTS: dict[str, object] = {
    "schema_version": "1.0",
    "host": "0.0.0.0",
    "port": 8770,
    "discovery_port": 8771,
    "enable_discovery": True,
    "recordings_root": "../local-data/recordings",
}
_PERCEPTION_DEFAULTS: dict[str, object] = {
    "schema_version": "1.0",
    "enabled": False,
    "max_live_inference_fps": 6.0,
}
_SENSOR_DEFAULTS: dict[str, object] = {
    "schema_version": "1.0",
    "calibration_file": "sensor-calibration-640x480-sample.json",
    "recorded": {"verify_media_hashes": True, "decode_threads": 0},
    "image": {
        "undistort": True,
        "interpolation": "linear",
        "border_mode": "constant",
    },
    "live": {"max_pending_imu_samples": 2048},
}
_HAND_DEFAULTS: dict[str, object] = {
    "schema_version": "1.0",
    "model_directory": "../local-data/models/hand-tracking",
    "device": "cuda",
    "require_cuda": True,
    "enable_cuda_amp": True,
    "require_hamer": True,
    "download_models": True,
    "detector": "mediapipe",
    "fallback_detector": "none",
    "allow_mediapipe_reconstruction_fallback": True,
    "vitpose_variant": "s",
    "detector_keypoint_confidence": 0.3,
    "detector_min_valid_keypoints": 3,
    "detector_bbox_padding_ratio": 0.3,
    "detector_min_bbox_dimension_ratio": 0.05,
    "detector_max_bbox_area_ratio": 0.35,
    "minimum_hand_confidence": 0.3,
    "physical_wrist_to_middle_mcp_m": 0.085,
    "minimum_depth_m": 0.05,
    "maximum_depth_m": 3.0,
    "grasp_ratio_threshold": 1.0,
    "sources": {
        "hamer_code_revision": "3a01849f4148352e9260b69bf28b65d1671a4905",
        "hamer_weights_revision": "f64df318d49e7e014a6a7f5d0547cba87d6e4317",
        "vitpose_code_revision": "bb9860359e55b099a507c8000e360d48a27cc36d",
        "vitpose_weights_revision": "e83805274e89428969355ec4afffcbc413e79188",
        "mediapipe_weights_revision": "6ed55322affd263688b9dfffda68a10545f15b95",
        "mano_weights_revision": "b00adea9a6843bbb4c9042109c5eb29ab2a59dea",
    },
}
_OFFLINE_HAND_DEFAULTS: dict[str, object] = {
    **_HAND_DEFAULTS,
    "device": "cuda",
    "require_cuda": True,
    "enable_cuda_amp": False,
    "require_hamer": True,
    "detector": "vitpose",
    "fallback_detector": "none",
    "allow_mediapipe_reconstruction_fallback": False,
    "vitpose_variant": "h",
    "temporal_processing": {
        "enabled": True,
        "confidence_threshold": 0.3,
        "interpolation_max_gap_frames": 20,
        "minimum_segment_frames": 10,
        "grasp_smoothing_window_frames": 5,
        "grasp_flicker_max_frames": 5,
        "sg_window_frames": 21,
        "sg_polyorder": 2,
        "orientation_ema_alpha": 0.15,
        "minimum_smoothing_frames": 6,
        "smoothing_fill_max_gap_frames": 10,
        "maximum_vio_pose_gap_ms": 100,
    },
}
_OFFLINE_HAND_INVARIANTS: dict[str, object] = {
    "device": "cuda",
    "require_cuda": True,
    "enable_cuda_amp": False,
    "require_hamer": True,
    "detector": "vitpose",
    "fallback_detector": "none",
    "allow_mediapipe_reconstruction_fallback": False,
    "vitpose_variant": "h",
}
_VIDEO_DEFAULTS: dict[str, object] = {
    "schema_version": "1.0",
    "default_preset_id": "hand-tracking-quality",
    "auto_enqueue_on_session_complete": False,
    "default_output_result_type": "structured_results",
}
_MODULE_DEFAULTS: dict[str, dict[str, object]] = {
    "client_runtime": _CLIENT_DEFAULTS,
    "sensor_preprocessing": _SENSOR_DEFAULTS,
    "live_hand_tracking": {
        "runtime": _PERCEPTION_DEFAULTS,
        "algorithm": _HAND_DEFAULTS,
    },
    "offline_hand_tracking": _OFFLINE_HAND_DEFAULTS,
    "video_processing": _VIDEO_DEFAULTS,
}

_FIELD_IMPACTS: dict[str, dict[str, ConfigImpact]] = {
    "client_runtime": {"*": ConfigImpact.RESTART_CLIENT},
    "sensor_preprocessing": {"*": ConfigImpact.NEXT_SESSION},
    "live_hand_tracking": {
        "runtime.enabled": ConfigImpact.IMMEDIATE,
        "runtime.max_live_inference_fps": ConfigImpact.IMMEDIATE,
        "algorithm.*": ConfigImpact.IMMEDIATE,
    },
    "offline_hand_tracking": {"*": ConfigImpact.NEXT_TASK},
    "video_processing": {
        "auto_enqueue_on_session_complete": ConfigImpact.IMMEDIATE,
        "default_preset_id": ConfigImpact.IMMEDIATE,
        "default_output_result_type": ConfigImpact.NEXT_TASK,
    },
}


class ConfigurationService:
    """Typed owner for the managed YAML files exposed as five UI modules."""

    def __init__(
        self,
        config_directory: str | Path = "config",
        *,
        recordings_root: str | Path | None = None,
        jobs_database_path: str | Path | None = None,
    ) -> None:
        self.config_directory = Path(config_directory).expanduser().resolve()
        self.config_directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._migration_warnings: list[str] = []
        self._legacy_hand_tracking_migrated = False
        self._ensure_bootstrap_files()

        client_payload = self._read_yaml(self._path("client_runtime"))
        client_values = self._validate_client(client_payload)
        configured_recordings_root = self._resolve_path(
            client_values["recordings_root"]
        )
        self.recordings_root = (
            Path(recordings_root).expanduser().resolve()
            if recordings_root is not None
            else configured_recordings_root
        )
        self.jobs_database_path = (
            Path(jobs_database_path).expanduser().resolve()
            if jobs_database_path is not None
            else self.recordings_root / ".processing" / "jobs.sqlite3"
        )
        self._state_path = self.jobs_database_path.parent / "configuration-state.json"
        self._migrate_legacy_processing_settings()
        self._saved_values = self._load_saved_values()
        self._working_values = copy.deepcopy(self._saved_values)
        self._revision = self._read_revision()
        if self._legacy_hand_tracking_migrated:
            self._write_state()

    @property
    def module_ids(self) -> tuple[str, ...]:
        return _MODULE_ORDER

    def snapshot(self) -> ConfigSnapshot:
        with self._lock:
            modules = tuple(
                ModuleConfigSnapshot(
                    module_id=module_id,
                    display_name=_DISPLAY_NAMES[module_id],
                    values=_freeze_mapping(self._working_values[module_id]),
                    source_paths=self._source_paths(module_id),
                    field_impacts=MappingProxyType(dict(_FIELD_IMPACTS[module_id])),
                )
                for module_id in _MODULE_ORDER
            )
            return ConfigSnapshot(
                modules=modules,
                revision=self._revision,
                dirty=self._working_values != self._saved_values,
            )

    def stage(
        self,
        values: Mapping[str, object],
    ) -> ConfigSnapshot:
        """Merge form values into memory without writing or applying them."""

        with self._lock:
            unknown = set(values) - set(_MODULE_ORDER)
            if unknown:
                name = sorted(unknown)[0]
                raise KeyError(f"unknown configuration module {name!r}")
            for module_id, module_values in values.items():
                if not isinstance(module_values, Mapping):
                    raise TypeError(f"{module_id} configuration must be a mapping")
                current = self._working_values[module_id]
                self._working_values[module_id] = _deep_merge(current, module_values)
            return self.snapshot()

    def validate(
        self,
        values: Mapping[str, object] | None = None,
    ) -> tuple[ValidationIssue, ...]:
        with self._lock:
            candidate = copy.deepcopy(self._working_values)
            if values is not None:
                try:
                    candidate = self._merged_values(candidate, values)
                except (KeyError, TypeError) as error:
                    return (ValidationIssue("configuration", "", str(error)),)
            _, issues = self._validated_values(candidate)
            return issues

    def save(
        self,
        values: Mapping[str, object] | None = None,
    ) -> ConfigApplyResult:
        """Validate and atomically persist changed YAML files.

        Runtime application is deliberately separate. The returned change list lets
        ``UnifiedRuntimeHost`` apply immediate fields and report deferred fields.
        """

        with self._lock:
            if values is not None:
                self.stage(values)
            normalized, issues = self._validated_values(self._working_values)
            if issues:
                raise ConfigurationValidationError(issues)
            changes = self._changes_between(self._saved_values, normalized)
            if not changes:
                return self._apply_result((), self.provenance())

            current_physical = self._physical_payloads(self._saved_values)
            next_physical = self._physical_payloads(normalized)
            changed_files = tuple(
                file_id
                for file_id in _FILE_NAMES
                if current_physical[file_id] != next_physical[file_id]
            )
            original_contents = {
                file_id: self._path(file_id).read_bytes() for file_id in changed_files
            }
            committed_files: list[str] = []
            try:
                for file_id in changed_files:
                    self._write_yaml_atomic(
                        self._path(file_id),
                        next_physical[file_id],
                        create_backup=True,
                    )
                    committed_files.append(file_id)
            except (OSError, UnicodeError, yaml.YAMLError) as error:
                rollback_errors: list[OSError] = []
                for file_id in reversed(committed_files):
                    try:
                        self._write_bytes_atomic(
                            self._path(file_id),
                            original_contents[file_id],
                            create_backup=False,
                        )
                    except OSError as rollback_error:
                        rollback_errors.append(rollback_error)
                self._saved_values = self._load_saved_values()
                message = "could not save configuration atomically"
                if rollback_errors:
                    message += "; rollback failed"
                raise ConfigurationError(message) from error

            self._saved_values = copy.deepcopy(normalized)
            self._working_values = copy.deepcopy(normalized)
            self._revision += 1
            compatibility_warning = self._sync_legacy_processing_settings()
            self._write_state()
            warnings = list(self._migration_warnings)
            if compatibility_warning is not None:
                warnings.append(compatibility_warning)
            result = self._apply_result(changes, self.provenance(), tuple(warnings))
            self._migration_warnings.clear()
            return result

    def discard(self) -> None:
        with self._lock:
            self._working_values = copy.deepcopy(self._saved_values)

    def restore_defaults(self, module_id: str) -> None:
        with self._lock:
            if module_id not in _MODULE_DEFAULTS:
                raise KeyError(f"unknown configuration module {module_id!r}")
            self._working_values[module_id] = copy.deepcopy(_MODULE_DEFAULTS[module_id])

    def reload(self) -> ConfigSnapshot:
        with self._lock:
            self._saved_values = self._load_saved_values()
            self._working_values = copy.deepcopy(self._saved_values)
            self._revision = self._read_revision()
            return self.snapshot()

    def provenance(self) -> ConfigurationProvenance:
        with self._lock:
            hashes = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (self._path(file_id) for file_id in _FILE_NAMES)
            }
            return ConfigurationProvenance(
                revision=self._revision,
                sha256_by_file=MappingProxyType(hashes),
            )

    def apply_request(self, result: ConfigApplyResult) -> ConfigurationApplyRequest:
        """Build the immutable boundary object consumed by a runtime host."""

        with self._lock:
            changed = set(result.changed_modules)
            return ConfigurationApplyRequest(
                revision=result.provenance.revision,
                changes=result.changes,
                values=MappingProxyType(
                    {
                        module_id: _freeze_mapping(self._saved_values[module_id])
                        for module_id in _MODULE_ORDER
                        if module_id in changed
                    }
                ),
            )

    def _ensure_bootstrap_files(self) -> None:
        for file_id, payload in (
            ("client_runtime", _CLIENT_DEFAULTS),
            ("video_processing", _VIDEO_DEFAULTS),
        ):
            path = self._path(file_id)
            if not path.exists():
                self._write_yaml_atomic(path, payload, create_backup=False)

        video_path = self._path("video_processing")
        video_payload = self._read_yaml(video_path)
        if video_payload.pop("default_inference_stride_frames", None) is not None:
            self._write_yaml_atomic(
                video_path,
                video_payload,
                create_backup=True,
            )
            self._migration_warnings.append(
                "已移除离线视频处理的旧推理步长设置，新任务固定逐帧推理"
            )

        live_path = self._path("live_hand_tracking")
        offline_path = self._path("offline_hand_tracking")
        legacy_path = self._path("legacy_hand_tracking")
        if not live_path.exists():
            if legacy_path.is_file():
                self._write_yaml_atomic(
                    live_path,
                    self._read_yaml(legacy_path),
                    create_backup=False,
                )
                self._migration_warnings.append(
                    "已将旧 hand-tracking.yaml 迁移为实时手部追踪配置"
                )
                self._legacy_hand_tracking_migrated = True
            else:
                self._write_yaml_atomic(
                    live_path,
                    _HAND_DEFAULTS,
                    create_backup=False,
                )
        if not offline_path.exists():
            self._write_yaml_atomic(
                offline_path,
                _OFFLINE_HAND_DEFAULTS,
                create_backup=False,
            )

    def _load_saved_values(self) -> dict[str, dict[str, object]]:
        candidates = {
            "client_runtime": self._read_yaml(self._path("client_runtime")),
            "sensor_preprocessing": self._read_yaml(
                self._path("sensor_preprocessing")
            ),
            "live_hand_tracking": {
                "runtime": self._read_yaml(self._path("perception_runtime")),
                "algorithm": self._read_yaml(
                    self._path("live_hand_tracking")
                ),
            },
            "offline_hand_tracking": self._read_yaml(
                self._path("offline_hand_tracking")
            ),
            "video_processing": self._read_yaml(self._path("video_processing")),
        }
        normalized, issues = self._validated_values(candidates)
        if issues:
            raise ConfigurationValidationError(issues)
        return normalized

    def _validated_values(
        self,
        values: Mapping[str, object],
    ) -> tuple[dict[str, dict[str, object]], tuple[ValidationIssue, ...]]:
        normalized: dict[str, dict[str, object]] = {}
        issues: list[ValidationIssue] = []
        for module_id in _MODULE_ORDER:
            raw = values.get(module_id)
            if not isinstance(raw, Mapping):
                issues.append(
                    ValidationIssue(module_id, "", "module configuration must be a mapping")
                )
                continue
            try:
                normalized[module_id] = self._validate_module(module_id, raw)
            except ValidationError as error:
                issues.extend(_pydantic_issues(module_id, error))
            except (OSError, UnicodeError, ValueError, TypeError) as error:
                field_path = "calibration_file" if module_id == "sensor_preprocessing" else ""
                issues.append(ValidationIssue(module_id, field_path, str(error)))
            except Exception as error:
                field_path = "calibration_file" if module_id == "sensor_preprocessing" else ""
                issues.append(ValidationIssue(module_id, field_path, str(error)))
        return normalized, tuple(issues)

    def _validate_module(
        self,
        module_id: str,
        values: Mapping[str, object],
    ) -> dict[str, object]:
        if module_id == "client_runtime":
            return self._validate_client(values)
        if module_id == "sensor_preprocessing":
            return self._validate_sensor(values)
        if module_id == "live_hand_tracking":
            if set(values) != {"runtime", "algorithm"}:
                raise ValueError("live hand tracking requires runtime and algorithm groups")
            runtime = values["runtime"]
            algorithm = values["algorithm"]
            if not isinstance(runtime, Mapping) or not isinstance(algorithm, Mapping):
                raise TypeError("live hand tracking groups must be mappings")
            return {
                "runtime": PerceptionRuntimeConfig.model_validate(runtime).model_dump(
                    mode="json"
                ),
                "algorithm": self._validate_hand_algorithm(algorithm),
            }
        if module_id == "offline_hand_tracking":
            normalized = self._validate_hand_algorithm(values)
            for field_name, required_value in _OFFLINE_HAND_INVARIANTS.items():
                if normalized[field_name] != required_value:
                    raise ValueError(
                        f"offline hand tracking requires {field_name}={required_value!r}"
                    )
            return normalized
        if module_id == "video_processing":
            return VideoProcessingConfig.model_validate(values).model_dump(mode="json")
        raise KeyError(f"unknown configuration module {module_id!r}")

    def _validate_client(self, values: Mapping[str, object]) -> dict[str, object]:
        payload = dict(values)
        path_value = payload.get("recordings_root")
        if not isinstance(path_value, (str, Path)):
            raise TypeError("recordings_root must be a path")
        resolved = self._resolve_path(path_value)
        payload["recordings_root"] = resolved
        model = ClientRuntimeConfig.model_validate(payload)
        normalized = model.model_dump(mode="json")
        normalized["recordings_root"] = self._stored_path(resolved)
        return normalized

    def _validate_sensor(self, values: Mapping[str, object]) -> dict[str, object]:
        payload = dict(values)
        path_value = payload.get("calibration_file")
        if not isinstance(path_value, (str, Path)):
            raise TypeError("calibration_file must be a path")
        calibration_path = self._resolve_path(path_value)
        payload["calibration_file"] = calibration_path
        model = SensorPreprocessingConfig.model_validate(payload)
        SensorCalibration.load(model.calibration_file)
        normalized = model.model_dump(mode="json")
        normalized["calibration_file"] = self._stored_path(model.calibration_file)
        return normalized

    def _validate_hand_algorithm(
        self,
        values: Mapping[str, object],
    ) -> dict[str, object]:
        payload = dict(values)
        path_value = payload.get("model_directory")
        if not isinstance(path_value, (str, Path)):
            raise TypeError("model_directory must be a path")
        model_path = self._resolve_path(path_value)
        payload["model_directory"] = model_path
        model = HandTrackingConfig.model_validate(payload)
        normalized = model.model_dump(mode="json")
        normalized["model_directory"] = self._stored_path(model.model_directory)
        return normalized

    def _merged_values(
        self,
        base: dict[str, dict[str, object]],
        values: Mapping[str, object],
    ) -> dict[str, dict[str, object]]:
        unknown = set(values) - set(_MODULE_ORDER)
        if unknown:
            raise KeyError(f"unknown configuration module {sorted(unknown)[0]!r}")
        result = copy.deepcopy(base)
        for module_id, module_values in values.items():
            if not isinstance(module_values, Mapping):
                raise TypeError(f"{module_id} configuration must be a mapping")
            result[module_id] = _deep_merge(result[module_id], module_values)
        return result

    def _changes_between(
        self,
        before: Mapping[str, Mapping[str, object]],
        after: Mapping[str, Mapping[str, object]],
    ) -> tuple[ConfigChange, ...]:
        changes: list[ConfigChange] = []
        for module_id in _MODULE_ORDER:
            for field_path in _diff_paths(before[module_id], after[module_id]):
                changes.append(
                    ConfigChange(
                        module_id,
                        field_path,
                        _impact_for(module_id, field_path),
                    )
                )
        return tuple(changes)

    def _physical_payloads(
        self,
        values: Mapping[str, Mapping[str, object]],
    ) -> dict[str, dict[str, object]]:
        live_hand = values["live_hand_tracking"]
        offline_hand = values["offline_hand_tracking"]
        runtime = live_hand["runtime"]
        live_algorithm = live_hand["algorithm"]
        assert isinstance(runtime, Mapping) and isinstance(live_algorithm, Mapping)
        assert isinstance(offline_hand, Mapping)
        return {
            "client_runtime": copy.deepcopy(dict(values["client_runtime"])),
            "perception_runtime": copy.deepcopy(dict(runtime)),
            "sensor_preprocessing": copy.deepcopy(
                dict(values["sensor_preprocessing"])
            ),
            "live_hand_tracking": copy.deepcopy(dict(live_algorithm)),
            "offline_hand_tracking": copy.deepcopy(dict(offline_hand)),
            "video_processing": copy.deepcopy(dict(values["video_processing"])),
        }

    def _apply_result(
        self,
        changes: tuple[ConfigChange, ...],
        provenance: ConfigurationProvenance,
        warnings: tuple[str, ...] = (),
    ) -> ConfigApplyResult:
        changed_modules = _ordered_modules(change.module_id for change in changes)
        immediate = _ordered_modules(
            change.module_id
            for change in changes
            if change.impact is ConfigImpact.IMMEDIATE
        )
        pending_restart = _ordered_modules(
            change.module_id
            for change in changes
            if change.impact is ConfigImpact.RESTART_CLIENT
        )
        pending_next_task = _ordered_modules(
            change.module_id
            for change in changes
            if change.impact is ConfigImpact.NEXT_TASK
        )
        pending_next_session = _ordered_modules(
            change.module_id
            for change in changes
            if change.impact is ConfigImpact.NEXT_SESSION
        )
        if pending_next_session:
            warnings = (
                *warnings,
                "传感器预处理配置将在下次连接或离线任务中生效",
            )
        return ConfigApplyResult(
            changed_modules=changed_modules,
            immediate_applied=immediate,
            pending_restart=pending_restart,
            pending_next_task=pending_next_task,
            warnings=warnings,
            pending_next_session=pending_next_session,
            changes=changes,
            provenance=provenance,
        )

    def _source_paths(self, module_id: str) -> tuple[Path, ...]:
        if module_id == "live_hand_tracking":
            return (
                self._path("perception_runtime"),
                self._path("live_hand_tracking"),
            )
        return (self._path(module_id),)

    def _path(self, file_id: str) -> Path:
        if file_id == "legacy_hand_tracking":
            return self.config_directory / _LEGACY_HAND_TRACKING_FILE
        return self.config_directory / _FILE_NAMES[file_id]

    def _resolve_path(self, value: object) -> Path:
        path = Path(value).expanduser()  # type: ignore[arg-type]
        if not path.is_absolute():
            path = self.config_directory / path
        return path.resolve()

    def _stored_path(self, value: str | Path) -> str:
        resolved = Path(value).resolve()
        try:
            relative = Path(os.path.relpath(resolved, self.config_directory))
        except ValueError:
            return resolved.as_posix()
        return relative.as_posix()

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, object]:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise ConfigurationError(f"cannot read configuration file {path.name}") from error
        if not isinstance(payload, dict):
            raise ConfigurationError(f"configuration file {path.name} must be a mapping")
        return payload

    @staticmethod
    def _write_yaml_atomic(
        path: Path,
        payload: Mapping[str, object],
        *,
        create_backup: bool,
    ) -> None:
        content = yaml.safe_dump(
            dict(payload),
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        ).encode("utf-8")
        ConfigurationService._write_bytes_atomic(
            path,
            content,
            create_backup=create_backup,
        )

    @staticmethod
    def _write_bytes_atomic(
        path: Path,
        content: bytes,
        *,
        create_backup: bool,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        backup_temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
                temporary = Path(stream.name)
            if create_backup and path.is_file():
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=path.parent,
                    prefix=f".{path.name}.bak.",
                    suffix=".tmp",
                    delete=False,
                ) as stream:
                    stream.write(path.read_bytes())
                    stream.flush()
                    os.fsync(stream.fileno())
                    backup_temporary = Path(stream.name)
                os.replace(backup_temporary, path.with_name(f"{path.name}.bak"))
                backup_temporary = None
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if backup_temporary is not None:
                backup_temporary.unlink(missing_ok=True)

    def _read_revision(self) -> int:
        if not self._state_path.is_file():
            return 0
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            revision = payload.get("revision")
            if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
                raise ValueError("invalid configuration revision")
            return revision
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            self._migration_warnings.append("配置版本状态损坏，已从版本 0 重新计数")
            return 0

    def _write_state(self) -> None:
        payload = {
            "schema_version": "1.0",
            "revision": self._revision,
            "updated_at_unix_ns": time.time_ns(),
            "sha256_by_file": dict(self.provenance().sha256_by_file),
            "migrations": {"hand_tracking_profiles_v2": True},
        }
        self._write_bytes_atomic(
            self._state_path,
            (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
            create_backup=False,
        )

    def _migrate_legacy_processing_settings(self) -> None:
        if not self.jobs_database_path.is_file():
            return
        try:
            with sqlite3.connect(self.jobs_database_path, timeout=5.0) as connection:
                rows = dict(
                    connection.execute(
                        "SELECT key, value FROM metadata WHERE key IN (?, ?, ?)",
                        (
                            "setting:auto_enqueue",
                            "setting:default_preset_id",
                            "setting:configuration_migrated_v1",
                        ),
                    ).fetchall()
                )
                if rows.get("setting:configuration_migrated_v1") == "true":
                    return
                payload = self._read_yaml(self._path("video_processing"))
                payload.pop("default_inference_stride_frames", None)
                auto_enqueue = rows.get("setting:auto_enqueue")
                preset_id = rows.get("setting:default_preset_id")
                if auto_enqueue in {"true", "false"}:
                    payload["auto_enqueue_on_session_complete"] = auto_enqueue == "true"
                elif auto_enqueue is not None:
                    self._migration_warnings.append(
                        "旧版自动入队设置无效，已保留统一配置中的值"
                    )
                if preset_id is not None:
                    candidate = {**payload, "default_preset_id": preset_id}
                    try:
                        VideoProcessingConfig.model_validate(candidate)
                    except ValidationError:
                        self._migration_warnings.append(
                            "旧版默认处理方案无效，已保留统一配置中的值"
                        )
                    else:
                        payload = candidate
                normalized = VideoProcessingConfig.model_validate(payload).model_dump(
                    mode="json"
                )
                self._write_yaml_atomic(
                    self._path("video_processing"), normalized, create_backup=True
                )
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    ("setting:configuration_migrated_v1", "true"),
                )
        except (OSError, sqlite3.Error, ValidationError, ConfigurationError) as error:
            self._migration_warnings.append(f"旧版处理设置迁移失败: {error}")

    def _sync_legacy_processing_settings(self) -> str | None:
        if not self.jobs_database_path.is_file():
            return None
        video = self._saved_values["video_processing"]
        try:
            with sqlite3.connect(self.jobs_database_path, timeout=5.0) as connection:
                for key, value in (
                    (
                        "setting:auto_enqueue",
                        "true"
                        if video["auto_enqueue_on_session_complete"]
                        else "false",
                    ),
                    ("setting:default_preset_id", str(video["default_preset_id"])),
                    ("setting:configuration_migrated_v1", "true"),
                ):
                    connection.execute(
                        "INSERT INTO metadata(key, value) VALUES (?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (key, value),
                    )
        except sqlite3.Error as error:
            return f"统一配置已保存，但旧版任务设置同步失败: {error}"
        return None


def _deep_merge(
    before: Mapping[str, object],
    updates: Mapping[str, object],
) -> dict[str, object]:
    result = copy.deepcopy(dict(before))
    for key, value in updates.items():
        previous = result.get(key)
        if isinstance(previous, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(previous, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _freeze_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    def freeze(value: object) -> object:
        if isinstance(value, Mapping):
            return MappingProxyType({str(key): freeze(item) for key, item in value.items()})
        if isinstance(value, list | tuple):
            return tuple(freeze(item) for item in value)
        return value

    return MappingProxyType({str(key): freeze(value) for key, value in values.items()})


def _diff_paths(
    before: Mapping[str, object],
    after: Mapping[str, object],
    prefix: str = "",
) -> tuple[str, ...]:
    paths: list[str] = []
    for key in sorted(set(before) | set(after)):
        path = f"{prefix}.{key}" if prefix else str(key)
        old = before.get(key)
        new = after.get(key)
        if isinstance(old, Mapping) and isinstance(new, Mapping):
            paths.extend(_diff_paths(old, new, path))
        elif old != new:
            paths.append(path)
    return tuple(paths)


def _impact_for(module_id: str, field_path: str) -> ConfigImpact:
    rules = _FIELD_IMPACTS[module_id]
    exact = rules.get(field_path)
    if exact is not None:
        return exact
    matches = (
        (prefix[:-2], impact)
        for prefix, impact in rules.items()
        if prefix.endswith(".*") and field_path.startswith(prefix[:-1])
    )
    best = max(matches, key=lambda item: len(item[0]), default=None)
    if best is not None:
        return best[1]
    return rules["*"]


def _ordered_modules(module_ids: Any) -> tuple[str, ...]:
    selected = set(module_ids)
    return tuple(module_id for module_id in _MODULE_ORDER if module_id in selected)


def _pydantic_issues(
    module_id: str,
    error: ValidationError,
) -> tuple[ValidationIssue, ...]:
    return tuple(
        ValidationIssue(
            module_id=module_id,
            field_path=".".join(str(item) for item in detail["loc"]),
            message=str(detail["msg"]),
        )
        for detail in error.errors(include_url=False)
    )
