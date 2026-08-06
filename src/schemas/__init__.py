"""Stable data shapes shared by the client host and algorithm services."""

from .frame import FramePacket
from .hand_tracking import (
    Handedness,
    HandKinematics,
    HandTemporalMetadata,
    HandTrackingConfig,
    HandTrackingError,
    HandTrackingResult,
    OfflineHandTemporalConfig,
    TemporalSource,
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
    "OfflineHandTemporalConfig",
    "HandTrackingError",
    "HandTrackingResult",
    "Handedness",
    "ImuPacket",
    "ImuSensor",
    "PlaybackFrame",
    "ProcessingArtifactRef",
    "TrackedHand",
    "HandKinematics",
    "HandTemporalMetadata",
    "TemporalSource",
    "VioPose",
    "VioTrajectory",
]
