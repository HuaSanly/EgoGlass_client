from __future__ import annotations

from importlib import import_module

import pytest

MODULES = (
    "calibration",
    "export",
    "hand_tracking",
    "models",
    "pipeline",
    "quality",
    "session_input",
    "time_association",
    "vio",
)


@pytest.mark.parametrize("module_name", MODULES)
def test_planned_spatial_perception_module_is_importable(module_name: str) -> None:
    module = import_module(f"egoglass_spatial_perception.{module_name}")
    assert module.__doc__
