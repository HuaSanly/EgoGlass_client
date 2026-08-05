from __future__ import annotations

import ast
import inspect
from pathlib import Path

from sensor_preprocessing import (
    CaptureSessionReader,
    RawFrameRef,
    RawImuSample,
)

SOURCE_ROOT = (
    Path(__file__).parents[1] / "src" / "sensor_preprocessing"
)
FORBIDDEN_IMPORT_PREFIXES = (
    "ui.gateway",
    "annotation",
    "hand_tracking",
    "ui",
)


def test_sensor_preprocessing_does_not_depend_on_runtime_consumers() -> None:
    violations: list[str] = []
    for source_path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            imported_names: list[str] = []
            if isinstance(node, ast.Import):
                imported_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names = [node.module]
            for imported_name in imported_names:
                if imported_name.startswith(FORBIDDEN_IMPORT_PREFIXES):
                    violations.append(f"{source_path.name}:{node.lineno}:{imported_name}")

    assert violations == []


def test_capture_reader_keeps_the_raw_replay_contract_public_and_read_only() -> None:
    open_parameters = inspect.signature(CaptureSessionReader.open).parameters

    assert open_parameters["verify_media_hashes"].default is True
    assert inspect.signature(CaptureSessionReader.iter_frames).return_annotation == (
        "Iterator[RawFrameRef]"
    )
    assert inspect.signature(CaptureSessionReader.iter_imu_samples).return_annotation == (
        "Iterator[RawImuSample]"
    )
    assert RawFrameRef.__dataclass_params__.frozen is True
    assert RawImuSample.__dataclass_params__.frozen is True

    reader_source = (SOURCE_ROOT / "capture_reader.py").read_text(encoding="utf-8")
    assert "mode=ro&immutable=1" in reader_source
    assert "PRAGMA query_only = ON" in reader_source
