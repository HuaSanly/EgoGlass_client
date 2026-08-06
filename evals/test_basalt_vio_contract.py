from __future__ import annotations

import ast
import os
import shutil
import subprocess
from pathlib import Path

import pytest


def test_basalt_cli_is_ui_independent() -> None:
    path = Path(__file__).parents[1] / "src" / "run_vio.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name == "ui" or name.startswith("ui.") for name in imports)
    assert not any(name in {"PyQt6", "qfluentwidgets", "aiortc"} for name in imports)


def test_basalt_config_defaults_to_quality_safe_flags() -> None:
    path = Path(__file__).parents[1] / "config" / "basalt-vio.yaml"
    text = path.read_text(encoding="utf-8")
    assert "allow_unverified_calibration: false" in text
    assert "input_is_rectified: true" in text
    assert "use_imu: true" in text


def test_native_basalt_help_when_configured() -> None:
    """Smoke-check the optional native executable without requiring it in CI."""

    executable = os.environ.get("EGOGLASS_BASALT_EXE") or shutil.which("basalt_vio")
    if not executable:
        pytest.skip("Basalt native executable is not configured")
    completed = subprocess.run(
        [executable, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--dataset-path" in completed.stdout


def test_vio_route_is_offline_workbench_only() -> None:
    repository = Path(__file__).parents[1]
    workbench_source = (repository / "ui" / "views" / "video_processing.py").read_text(
        encoding="utf-8"
    )
    live_source = (repository / "ui" / "views" / "home.py").read_text(encoding="utf-8")
    assert "request_vio" in workbench_source
    assert "request_vio" not in live_source
