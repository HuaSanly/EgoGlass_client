"""Public hand-tracking contracts and pipeline."""

from .models import (
    HUMANEGO_ARIA_JOINT_NAMES,
    DetectedHand,
    Handedness,
    HandKinematics,
    HandReconstruction,
    HandTemporalMetadata,
    HandTrackingConfig,
    HandTrackingError,
    HandTrackingResult,
    MetricDepthStatus,
    OfflineHandTemporalConfig,
    ReconstructionBackend,
    TemporalSource,
    TrackedHand,
    remap_hamer_to_humanego_aria,
    rotated_image_bbox_to_source,
    rotated_image_points_to_source,
    source_image_dimensions,
)
from .pipeline import HumanEgoHandTrackingPipeline, release_pipeline_resources
from .temporal import (
    OfflineHandTemporalProcessor,
    TemporalProcessingOutput,
    TemporalProcessingStats,
)

__all__ = [
    "HUMANEGO_ARIA_JOINT_NAMES",
    "DetectedHand",
    "Handedness",
    "HandReconstruction",
    "HandKinematics",
    "HandTemporalMetadata",
    "HandTrackingConfig",
    "HandTrackingError",
    "HandTrackingResult",
    "HumanEgoHandTrackingPipeline",
    "MetricDepthStatus",
    "OfflineHandTemporalConfig",
    "OfflineHandTemporalProcessor",
    "ReconstructionBackend",
    "TrackedHand",
    "TemporalProcessingOutput",
    "TemporalProcessingStats",
    "TemporalSource",
    "remap_hamer_to_humanego_aria",
    "release_pipeline_resources",
    "rotated_image_bbox_to_source",
    "rotated_image_points_to_source",
    "source_image_dimensions",
]
