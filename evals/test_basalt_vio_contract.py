from __future__ import annotations

import ast
from pathlib import Path


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
