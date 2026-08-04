from __future__ import annotations

import shutil
from pathlib import Path

from perception.configuration import ConfigurationService
from perception.spatial_perception.hand_tracking import HandTrackingConfig
from perception.video_processing import DEFAULT_PRESETS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_configuration_roundtrip_preserves_values_and_provenance(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.mkdir()
    for name in (
        "client-runtime.yaml",
        "perception-runtime.yaml",
        "sensor-preprocessing.yaml",
        "live-hand-tracking.yaml",
        "offline-hand-tracking.yaml",
        "video-processing.yaml",
        "sensor-calibration-640x480-sample.json",
    ):
        shutil.copy2(PROJECT_ROOT / "config" / name, config / name)
    recordings = tmp_path / "recordings"

    service = ConfigurationService(config, recordings_root=recordings)
    result = service.save(
        {
            "live_hand_tracking": {
                "runtime": {"max_live_inference_fps": 8.0},
                "algorithm": {"detector_keypoint_confidence": 0.45},
            },
            "offline_hand_tracking": {"detector_keypoint_confidence": 0.55},
            "video_processing": {
                "default_preset_id": "hand-tracking-quality",
                "auto_enqueue_on_session_complete": True,
            },
        }
    )
    expected = {
        module.module_id: dict(module.values) for module in service.snapshot().modules
    }

    restarted = ConfigurationService(config, recordings_root=recordings)
    actual = {
        module.module_id: dict(module.values) for module in restarted.snapshot().modules
    }

    assert actual == expected
    assert restarted.provenance().to_json_dict() == result.provenance.to_json_dict()
    assert restarted.validate() == ()


def test_live_and_offline_profiles_keep_distinct_execution_policies() -> None:
    live = HandTrackingConfig.load(
        PROJECT_ROOT / "config" / "live-hand-tracking.yaml"
    )
    offline = HandTrackingConfig.load(
        PROJECT_ROOT / "config" / "offline-hand-tracking.yaml"
    )

    assert live.detector == "mediapipe"
    assert live.enable_cuda_amp is True
    assert live.allow_mediapipe_reconstruction_fallback is True

    assert offline.detector == "vitpose"
    assert offline.vitpose_variant == "h"
    assert offline.require_hamer is True
    assert offline.require_cuda is True
    assert offline.enable_cuda_amp is False
    assert offline.fallback_detector == "none"
    assert offline.allow_mediapipe_reconstruction_fallback is False

    assert len(DEFAULT_PRESETS) == 1
    preset = DEFAULT_PRESETS[0]
    frame_indices = range(90)
    inferred = [
        frame_index
        for frame_index in frame_indices
        if frame_index % preset.inference_stride_frames == 0
    ]
    assert inferred == list(range(90))
