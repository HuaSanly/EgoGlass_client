from __future__ import annotations

import shutil
from pathlib import Path

from perception.configuration import ConfigurationService

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_configuration_roundtrip_preserves_values_and_provenance(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.mkdir()
    for name in (
        "client-runtime.yaml",
        "perception-runtime.yaml",
        "sensor-preprocessing.yaml",
        "hand-tracking.yaml",
        "video-processing.yaml",
        "sensor-calibration-640x480-sample.json",
    ):
        shutil.copy2(PROJECT_ROOT / "config" / name, config / name)
    recordings = tmp_path / "recordings"

    service = ConfigurationService(config, recordings_root=recordings)
    result = service.save(
        {
            "hand_tracking": {
                "runtime": {"max_live_inference_fps": 8.0},
                "algorithm": {"detector_keypoint_confidence": 0.45},
            },
            "video_processing": {
                "default_preset_id": "hand-tracking-balanced",
                "default_inference_stride_frames": 3,
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
