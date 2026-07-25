from __future__ import annotations

from importlib import import_module

import pytest

MODULES = (
    "arm_inpainting",
    "export",
    "interaction_trajectory",
    "keypoint_selection",
    "keypoint_tracking",
    "object_pose_estimation",
    "object_segmentation",
    "phase_segmentation",
    "pipeline",
    "quality",
    "visualization",
    "window_selection",
)


@pytest.mark.parametrize("module_name", MODULES)
def test_planned_interaction_processing_module_is_importable(module_name: str) -> None:
    module = import_module(f"interaction_processing.{module_name}")
    assert module.__doc__
