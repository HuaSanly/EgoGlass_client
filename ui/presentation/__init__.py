"""Presentation-only adapters for algorithm results."""

from .hand_overlay import render_hand_tracking_overlay
from .spatial_scene import SpatialReferenceFrame, SpatialSceneState, build_spatial_scene_state

__all__ = [
    "SpatialReferenceFrame",
    "SpatialSceneState",
    "build_spatial_scene_state",
    "render_hand_tracking_overlay",
]
