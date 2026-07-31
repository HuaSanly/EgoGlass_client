"""Custom paint surfaces that have no Fluent component equivalent."""

from .spatial_sync_canvas import SpatialSyncCanvas, SpatialSyncCanvasStatus
from .video_canvas import VideoCanvas, VideoCanvasStatus, fit_image_geometry

__all__ = [
    "SpatialSyncCanvas",
    "SpatialSyncCanvasStatus",
    "VideoCanvas",
    "VideoCanvasStatus",
    "fit_image_geometry",
]
