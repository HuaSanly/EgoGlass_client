"""Stable data shapes shared by the client host and algorithm services."""

from .frame import FramePacket
from .hand_tracking import (
    Handedness,
    HandTrackingConfig,
    HandTrackingError,
    HandTrackingResult,
    TrackedHand,
)
from .imu import ImuPacket, ImuSensor
from .playback import PlaybackFrame
from .processing import AlgorithmRunMetadata, ProcessingArtifactRef
from .trajectory import VioPose, VioTrajectory

__all__ = [
    "AlgorithmRunMetadata",
    "FramePacket",
    "HandTrackingConfig",
    "HandTrackingError",
    "HandTrackingResult",
    "Handedness",
    "ImuPacket",
    "ImuSensor",
    "PlaybackFrame",
    "ProcessingArtifactRef",
    "TrackedHand",
    "VioPose",
    "VioTrajectory",
]
