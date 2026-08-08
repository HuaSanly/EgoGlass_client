"""Stable data shapes shared by the client host and algorithm services."""

from .dataset import (
    ArtifactReference,
    DatasetEpisode,
    DatasetFrame,
    DatasetManifest,
    DatasetSplit,
    HandSample,
    ObjectObservation,
    QualityGate,
    QualityIssue,
    QualitySeverity,
    SensorSampleReference,
    VirtualVideoSpan,
)
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
from .object_tracking import (
    BoundingBox,
    ObjectKeypointTrack,
    ObjectMaskObservation,
    ObjectPose,
    ObjectTrackingResult,
    ObjectTriangulation,
)
from .phase import (
    MotionPhase,
    ObjectCentricWindow,
    PhaseAnalysisResult,
    PhaseFrame,
    PhaseSegment,
)
from .playback import PlaybackFrame
from .processing import AlgorithmRunMetadata, ProcessingArtifactRef
from .trajectory import VioPose, VioTrajectory

__all__ = [
    "AlgorithmRunMetadata",
    "ArtifactReference",
    "BoundingBox",
    "DatasetEpisode",
    "DatasetFrame",
    "DatasetManifest",
    "DatasetSplit",
    "FramePacket",
    "HandTrackingConfig",
    "OfflineHandTemporalConfig",
    "HandTrackingError",
    "HandTrackingResult",
    "Handedness",
    "ImuPacket",
    "ImuSensor",
    "HandSample",
    "MotionPhase",
    "ObjectCentricWindow",
    "ObjectObservation",
    "ObjectKeypointTrack",
    "ObjectMaskObservation",
    "ObjectPose",
    "ObjectTrackingResult",
    "ObjectTriangulation",
    "PlaybackFrame",
    "ProcessingArtifactRef",
    "PhaseAnalysisResult",
    "PhaseFrame",
    "PhaseSegment",
    "QualityGate",
    "QualityIssue",
    "QualitySeverity",
    "SensorSampleReference",
    "TrackedHand",
    "HandKinematics",
    "HandTemporalMetadata",
    "TemporalSource",
    "VioPose",
    "VioTrajectory",
    "VirtualVideoSpan",
]
