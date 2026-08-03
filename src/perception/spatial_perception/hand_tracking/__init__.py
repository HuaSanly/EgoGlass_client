"""Public hand-tracking contracts and pipeline."""

from .models import (
    HUMANEGO_ARIA_JOINT_NAMES,
    DetectedHand,
    Handedness,
    HandReconstruction,
    HandTrackingConfig,
    HandTrackingError,
    HandTrackingResult,
    MetricDepthStatus,
    ReconstructionBackend,
    TrackedHand,
    remap_hamer_to_humanego_aria,
    rotated_image_bbox_to_source,
    rotated_image_points_to_source,
    source_image_dimensions,
)
from .pipeline import HumanEgoHandTrackingPipeline, release_pipeline_resources
from .visualization import render_hand_tracking_overlay

__all__ = [
    "HUMANEGO_ARIA_JOINT_NAMES",
    "DetectedHand",
    "Handedness",
    "HandReconstruction",
    "HandTrackingConfig",
    "HandTrackingError",
    "HandTrackingResult",
    "HumanEgoHandTrackingPipeline",
    "MetricDepthStatus",
    "ReconstructionBackend",
    "TrackedHand",
    "remap_hamer_to_humanego_aria",
    "release_pipeline_resources",
    "render_hand_tracking_overlay",
    "rotated_image_bbox_to_source",
    "rotated_image_points_to_source",
    "source_image_dimensions",
]
