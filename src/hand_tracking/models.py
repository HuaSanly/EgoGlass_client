"""Typed configuration and boundary models for hand tracking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

FloatArray = NDArray[np.float32]

HUMANEGO_ARIA_JOINT_NAMES = (
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
    "wrist",
    "thumb_mcp",
    "thumb_ip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "palm_center",
)

_MP_TO_HUMANEGO_ARIA = (
    4,
    8,
    12,
    16,
    20,
    0,
    2,
    3,
    5,
    6,
    7,
    9,
    10,
    11,
    13,
    14,
    15,
    17,
    18,
    19,
)


class HandTrackingError(RuntimeError):
    """Hand tracking cannot initialize or produce a valid result."""


class Handedness(StrEnum):
    """The wearer's physical hand side in a non-mirrored egocentric image."""

    LEFT = "left"
    RIGHT = "right"


class ReconstructionBackend(StrEnum):
    """The implementation that produced a hand's 3D structure."""

    HAMER = "hamer"
    MEDIAPIPE = "mediapipe"


class MetricDepthStatus(StrEnum):
    """How the absolute camera-space depth was obtained."""

    MODEL_ESTIMATED = "model_estimated"
    PHYSICAL_SIZE_ESTIMATED = "physical_size_estimated"


class TemporalSource(StrEnum):
    """Whether a finalized hand came from inference or temporal interpolation."""

    OBSERVED = "observed"
    INTERPOLATED = "interpolated"


class OfflineHandTemporalConfig(BaseModel):
    """HumanEgo-derived sequence cleanup used only by offline processing."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    enabled: bool = True
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    interpolation_max_gap_frames: int = Field(default=20, ge=0, le=300)
    minimum_segment_frames: int = Field(default=10, ge=1, le=3000)
    grasp_smoothing_window_frames: int = Field(default=5, ge=1, le=301)
    grasp_flicker_max_frames: int = Field(default=5, ge=0, le=300)
    sg_window_frames: int = Field(default=21, ge=5, le=1001)
    sg_polyorder: int = Field(default=2, ge=1, le=10)
    orientation_ema_alpha: float = Field(default=0.15, gt=0.0, le=1.0)
    minimum_smoothing_frames: int = Field(default=6, ge=2, le=3000)
    smoothing_fill_max_gap_frames: int = Field(default=10, ge=0, le=300)
    maximum_vio_pose_gap_ms: int = Field(default=100, ge=1, le=10_000)

    @model_validator(mode="after")
    def validate_filter_shape(self) -> OfflineHandTemporalConfig:
        if self.sg_window_frames % 2 == 0:
            raise ValueError("sg_window_frames must be odd")
        if self.sg_polyorder >= self.sg_window_frames:
            raise ValueError("sg_polyorder must be smaller than sg_window_frames")
        return self


class ModelSourceConfig(BaseModel):
    """Pinned upstream revisions used by the HumanEgo reproduction."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    hamer_code_revision: str = "3a01849f4148352e9260b69bf28b65d1671a4905"
    hamer_weights_revision: str = "f64df318d49e7e014a6a7f5d0547cba87d6e4317"
    vitpose_code_revision: str = "bb9860359e55b099a507c8000e360d48a27cc36d"
    vitpose_weights_revision: str = "e83805274e89428969355ec4afffcbc413e79188"
    mediapipe_weights_revision: str = "6ed55322affd263688b9dfffda68a10545f15b95"
    mano_weights_revision: str = "b00adea9a6843bbb4c9042109c5eb29ab2a59dea"


class HandTrackingConfig(BaseModel):
    """Runtime settings for HumanEgo-compatible HaMeR hand tracking."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1.0"]
    model_directory: Path
    device: Literal["cuda", "cpu"] = "cuda"
    require_cuda: bool = True
    enable_cuda_amp: bool = True
    require_hamer: bool = True
    download_models: bool = True
    detector: Literal["vitpose", "mediapipe"] = "mediapipe"
    fallback_detector: Literal["mediapipe", "none"] = "none"
    allow_mediapipe_reconstruction_fallback: bool = True
    vitpose_variant: Literal["h", "l", "b", "s"] = "s"
    detector_keypoint_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    detector_min_valid_keypoints: int = Field(default=3, ge=1, le=21)
    detector_bbox_padding_ratio: float = Field(default=0.3, ge=0.0, le=2.0)
    detector_min_bbox_dimension_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    detector_max_bbox_area_ratio: float = Field(default=0.35, gt=0.0, le=1.0)
    minimum_hand_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    physical_wrist_to_middle_mcp_m: float = Field(default=0.085, gt=0.0)
    minimum_depth_m: float = Field(default=0.05, gt=0.0)
    maximum_depth_m: float = Field(default=3.0, gt=0.0)
    grasp_ratio_threshold: float = Field(default=1.0, gt=0.0)
    temporal_processing: OfflineHandTemporalConfig | None = None
    sources: ModelSourceConfig = Field(default_factory=ModelSourceConfig)

    @model_validator(mode="after")
    def validate_detector_fallback(self) -> HandTrackingConfig:
        """Reject a fallback that repeats the selected primary detector."""

        if self.detector == self.fallback_detector:
            raise ValueError("fallback_detector must differ from detector")
        return self

    @classmethod
    def load(cls, path: str | Path) -> HandTrackingConfig:
        """Load YAML and resolve the model directory relative to that YAML."""

        try:
            config_path = Path(path).resolve()
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
            raise HandTrackingError("invalid hand tracking config file") from exc
        if not isinstance(payload, dict):
            raise HandTrackingError("invalid hand tracking config file")
        model_directory = payload.get("model_directory")
        if not isinstance(model_directory, str) or not model_directory.strip():
            raise HandTrackingError("invalid hand tracking config file")
        try:
            resolved_model_directory = Path(model_directory)
            if not resolved_model_directory.is_absolute():
                resolved_model_directory = config_path.parent / resolved_model_directory
            return cls.model_validate(
                {**payload, "model_directory": resolved_model_directory.resolve()}
            )
        except (OSError, ValueError, ValidationError) as exc:
            raise HandTrackingError("invalid hand tracking config file") from exc


@dataclass(frozen=True, slots=True)
class DetectedHand:
    """A detector crop and optional MediaPipe relative 3D landmarks."""

    bbox_xyxy_px: tuple[float, float, float, float]
    handedness: Handedness
    confidence: float
    keypoints_2d_px: FloatArray
    relative_keypoints_3d_m: FloatArray | None = None
    egocentric_fallback: bool = False

    def __post_init__(self) -> None:
        _validate_array(self.keypoints_2d_px, (21, 2), "detector 2D keypoints")
        if self.relative_keypoints_3d_m is not None:
            _validate_array(
                self.relative_keypoints_3d_m,
                (21, 3),
                "detector relative 3D keypoints",
            )
        x1, y1, x2, y2 = self.bbox_xyxy_px
        if not all(np.isfinite(value) for value in self.bbox_xyxy_px):
            raise ValueError("detector bounding box must be finite")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("detector bounding box must have positive area")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("detector confidence must be between zero and one")


@dataclass(frozen=True, slots=True)
class HandReconstruction:
    """HaMeR-order camera-space reconstruction for one detector crop."""

    keypoints_3d_m: FloatArray
    keypoints_2d_px: FloatArray
    confidence: float
    depth_score: float
    coverage_score: float
    compactness_score: float

    def __post_init__(self) -> None:
        _validate_array(self.keypoints_3d_m, (21, 3), "reconstructed 3D keypoints")
        _validate_array(self.keypoints_2d_px, (21, 2), "reconstructed 2D keypoints")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("reconstruction confidence must be between zero and one")
        for name, value in (
            ("depth score", self.depth_score),
            ("coverage score", self.coverage_score),
            ("compactness score", self.compactness_score),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")


@dataclass(frozen=True, slots=True)
class HandTemporalMetadata:
    """Per-hand provenance added by the offline temporal processor."""

    source: TemporalSource
    interpolation_source_frames: tuple[int, int] | None = None
    vio_pose_timestamp_ns: int | None = None
    vio_time_difference_ns: int | None = None
    world_kinematics_available: bool = False
    kinematics_optimized: bool = False

    def __post_init__(self) -> None:
        if self.source is TemporalSource.INTERPOLATED:
            if self.interpolation_source_frames is None:
                raise ValueError("interpolated hands require source frame indices")
            start, end = self.interpolation_source_frames
            if start < 0 or end <= start:
                raise ValueError("interpolation source frames are invalid")
        elif self.interpolation_source_frames is not None:
            raise ValueError("observed hands cannot have interpolation source frames")
        timing = (self.vio_pose_timestamp_ns, self.vio_time_difference_ns)
        if self.world_kinematics_available and any(value is None for value in timing):
            raise ValueError("world kinematics require VIO timing provenance")
        if not self.world_kinematics_available and any(value is not None for value in timing):
            raise ValueError("unavailable world kinematics cannot contain VIO timing")
        if any(value is not None and value < 0 for value in timing):
            raise ValueError("VIO timing values must be non-negative")
        if self.kinematics_optimized and not self.world_kinematics_available:
            raise ValueError("optimized kinematics require world kinematics")


@dataclass(frozen=True, slots=True)
class HandKinematics:
    """HumanEgo-style raw and optimized world-space hand kinematics."""

    keypoints_3d_world_m: FloatArray
    wrist_pose_raw_world: FloatArray
    wrist_pose_optimized_world: FloatArray
    midpoint_pose_raw_world: FloatArray
    midpoint_pose_optimized_world: FloatArray
    thumb_tip_raw_world_m: FloatArray
    thumb_tip_optimized_world_m: FloatArray
    index_tip_raw_world_m: FloatArray
    index_tip_optimized_world_m: FloatArray
    thumb_base_raw_world_m: FloatArray
    thumb_base_optimized_world_m: FloatArray
    index_base_raw_world_m: FloatArray
    index_base_optimized_world_m: FloatArray
    wrist_linear_velocity_raw_m_s: FloatArray
    wrist_linear_velocity_optimized_m_s: FloatArray
    wrist_angular_velocity_raw_rad_s: FloatArray
    wrist_angular_velocity_optimized_rad_s: FloatArray
    midpoint_linear_velocity_raw_m_s: FloatArray
    midpoint_linear_velocity_optimized_m_s: FloatArray
    midpoint_angular_velocity_raw_rad_s: FloatArray
    midpoint_angular_velocity_optimized_rad_s: FloatArray

    def __post_init__(self) -> None:
        _validate_array(self.keypoints_3d_world_m, (21, 3), "world hand keypoints")
        for name in (
            "wrist_pose_raw_world",
            "wrist_pose_optimized_world",
            "midpoint_pose_raw_world",
            "midpoint_pose_optimized_world",
        ):
            _validate_array(getattr(self, name), (4, 4), name)
        for name in (
            "thumb_tip_raw_world_m",
            "thumb_tip_optimized_world_m",
            "index_tip_raw_world_m",
            "index_tip_optimized_world_m",
            "thumb_base_raw_world_m",
            "thumb_base_optimized_world_m",
            "index_base_raw_world_m",
            "index_base_optimized_world_m",
            "wrist_linear_velocity_raw_m_s",
            "wrist_linear_velocity_optimized_m_s",
            "wrist_angular_velocity_raw_rad_s",
            "wrist_angular_velocity_optimized_rad_s",
            "midpoint_linear_velocity_raw_m_s",
            "midpoint_linear_velocity_optimized_m_s",
            "midpoint_angular_velocity_raw_rad_s",
            "midpoint_angular_velocity_optimized_rad_s",
        ):
            _validate_array(getattr(self, name), (3,), name)


@dataclass(frozen=True, slots=True)
class TrackedHand:
    """One hand in HumanEgo's Aria-compatible 21-joint order."""

    handedness: Handedness
    confidence: float
    detector_confidence: float
    reconstruction_quality: float | None
    depth_score: float | None
    coverage_score: float | None
    compactness_score: float | None
    reconstruction_backend: ReconstructionBackend
    metric_depth_status: MetricDepthStatus
    bbox_xyxy_px: tuple[float, float, float, float]
    keypoints_2d_px: FloatArray
    keypoints_3d_camera_m: FloatArray
    joint_angles_degrees: Mapping[str, float]
    grasp_ratio: float
    is_grasping: bool
    temporal: HandTemporalMetadata | None = None
    kinematics: HandKinematics | None = None

    def __post_init__(self) -> None:
        _validate_array(self.keypoints_2d_px, (21, 2), "tracked 2D keypoints")
        _validate_array(self.keypoints_3d_camera_m, (21, 3), "tracked 3D keypoints")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("tracked hand confidence must be between zero and one")
        if not 0.0 <= self.detector_confidence <= 1.0:
            raise ValueError("detector confidence must be between zero and one")
        for name, value in (
            ("reconstruction quality", self.reconstruction_quality),
            ("depth score", self.depth_score),
            ("coverage score", self.coverage_score),
            ("compactness score", self.compactness_score),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if not np.isfinite(self.grasp_ratio) or self.grasp_ratio < 0.0:
            raise ValueError("grasp ratio must be finite and non-negative")
        if not all(np.isfinite(value) for value in self.joint_angles_degrees.values()):
            raise ValueError("joint angles must be finite")
        if self.kinematics is not None and (
            self.temporal is None or not self.temporal.world_kinematics_available
        ):
            raise ValueError("kinematics require matching temporal provenance")


@dataclass(frozen=True, slots=True)
class HandTrackingResult:
    """Per-frame result passed to visualization, storage, or later perception stages."""

    schema_version: Literal["1.0"]
    session_id: str
    sequence_id: str
    frame_index: int
    session_time_ns: int
    timestamp_uncertainty_ns: int
    image_width_px: int
    image_height_px: int
    source_rotation_degrees: int
    detector_backend: str
    requested_device: str
    execution_device: str
    hamer_loaded: bool
    inference_duration_ns: int
    hands: tuple[TrackedHand, ...]
    frame_preparation_duration_ns: int = 0
    detector_duration_ns: int = 0
    reconstruction_duration_ns: int = 0
    postprocessing_duration_ns: int = 0
    reconstruction_batch_size: int = 0
    amp_enabled: bool = False

    def __post_init__(self) -> None:
        if self.image_width_px < 1 or self.image_height_px < 1:
            raise ValueError("result image dimensions must be positive")
        if self.source_rotation_degrees not in {0, 90, 180, 270}:
            raise ValueError("result source rotation must be valid")
        stage_duration_ns = (
            self.frame_preparation_duration_ns
            + self.detector_duration_ns
            + self.reconstruction_duration_ns
            + self.postprocessing_duration_ns
        )
        if stage_duration_ns != self.inference_duration_ns:
            raise ValueError("stage durations must sum to inference duration")
        if self.reconstruction_batch_size < 0:
            raise ValueError("reconstruction batch size cannot be negative")

    def to_json_dict(self) -> dict[str, object]:
        """Convert the immutable result to a JSON-compatible boundary payload."""

        source_width_px, source_height_px = source_image_dimensions(
            self.image_width_px,
            self.image_height_px,
            self.source_rotation_degrees,
        )
        hands_payload: list[dict[str, object]] = []
        for hand in self.hands:
            source_keypoints = rotated_image_points_to_source(
                hand.keypoints_2d_px,
                image_width_px=self.image_width_px,
                image_height_px=self.image_height_px,
                rotation_degrees=self.source_rotation_degrees,
            )
            source_bbox = rotated_image_bbox_to_source(
                hand.bbox_xyxy_px,
                image_width_px=self.image_width_px,
                image_height_px=self.image_height_px,
                rotation_degrees=self.source_rotation_degrees,
            )
            hands_payload.append(
                {
                    "handedness": hand.handedness.value,
                    "confidence": hand.confidence,
                    "detector_confidence": hand.detector_confidence,
                    "reconstruction_quality": hand.reconstruction_quality,
                    "depth_score": hand.depth_score,
                    "coverage_score": hand.coverage_score,
                    "compactness_score": hand.compactness_score,
                    "final_confidence": hand.confidence,
                    "reconstruction_backend": hand.reconstruction_backend.value,
                    "metric_depth_status": hand.metric_depth_status.value,
                    "bbox_xyxy_px": list(hand.bbox_xyxy_px),
                    "keypoints_2d_px": hand.keypoints_2d_px.tolist(),
                    "source_bbox_xyxy_px": list(source_bbox),
                    "source_keypoints_2d_px": source_keypoints.tolist(),
                    "keypoints_3d_camera_m": hand.keypoints_3d_camera_m.tolist(),
                    "joint_angles_degrees": dict(hand.joint_angles_degrees),
                    "grasp_ratio": hand.grasp_ratio,
                    "is_grasping": hand.is_grasping,
                    "temporal": _temporal_json(hand.temporal),
                    "kinematics": _kinematics_json(hand.kinematics),
                }
            )
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "sequence_id": self.sequence_id,
            "frame_index": self.frame_index,
            "session_time_ns": self.session_time_ns,
            "timestamp_uncertainty_ns": self.timestamp_uncertainty_ns,
            "image_width_px": self.image_width_px,
            "image_height_px": self.image_height_px,
            "source_rotation_degrees": self.source_rotation_degrees,
            "source_image_width_px": source_width_px,
            "source_image_height_px": source_height_px,
            "detector_backend": self.detector_backend,
            "requested_device": self.requested_device,
            "execution_device": self.execution_device,
            "hamer_loaded": self.hamer_loaded,
            "inference_duration_ns": self.inference_duration_ns,
            "frame_preparation_duration_ns": self.frame_preparation_duration_ns,
            "detector_duration_ns": self.detector_duration_ns,
            "reconstruction_duration_ns": self.reconstruction_duration_ns,
            "postprocessing_duration_ns": self.postprocessing_duration_ns,
            "reconstruction_batch_size": self.reconstruction_batch_size,
            "amp_enabled": self.amp_enabled,
            "joint_order": list(HUMANEGO_ARIA_JOINT_NAMES),
            "hands": hands_payload,
        }


def source_image_dimensions(
    image_width_px: int,
    image_height_px: int,
    rotation_degrees: int,
) -> tuple[int, int]:
    """Return decoded source dimensions before the preprocessing rotation."""

    if image_width_px < 1 or image_height_px < 1:
        raise ValueError("image dimensions must be positive")
    if rotation_degrees not in {0, 90, 180, 270}:
        raise ValueError("rotation must be 0, 90, 180, or 270 degrees")
    if rotation_degrees in {90, 270}:
        return image_height_px, image_width_px
    return image_width_px, image_height_px


def rotated_image_points_to_source(
    points: NDArray[np.floating],
    *,
    image_width_px: int,
    image_height_px: int,
    rotation_degrees: int,
) -> FloatArray:
    """Map rotated preprocessing pixel centers back to decoded source coordinates."""

    source_width_px, source_height_px = source_image_dimensions(
        image_width_px,
        image_height_px,
        rotation_degrees,
    )
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("points must be a finite Nx2 array")
    mapped = np.empty_like(values)
    if rotation_degrees == 0:
        mapped[:] = values
    elif rotation_degrees == 90:
        mapped[:, 0] = values[:, 1]
        mapped[:, 1] = source_height_px - 1 - values[:, 0]
    elif rotation_degrees == 180:
        mapped[:, 0] = source_width_px - 1 - values[:, 0]
        mapped[:, 1] = source_height_px - 1 - values[:, 1]
    else:
        mapped[:, 0] = source_width_px - 1 - values[:, 1]
        mapped[:, 1] = values[:, 0]
    mapped = np.ascontiguousarray(mapped, dtype=np.float32)
    mapped.setflags(write=False)
    return mapped


def rotated_image_bbox_to_source(
    bbox_xyxy_px: tuple[float, float, float, float],
    *,
    image_width_px: int,
    image_height_px: int,
    rotation_degrees: int,
) -> tuple[float, float, float, float]:
    """Map a rotated image-edge bounding box back to the decoded source image."""

    source_width_px, source_height_px = source_image_dimensions(
        image_width_px,
        image_height_px,
        rotation_degrees,
    )
    x1, y1, x2, y2 = bbox_xyxy_px
    if not all(np.isfinite(value) for value in bbox_xyxy_px) or x2 <= x1 or y2 <= y1:
        raise ValueError("bounding box must be finite and have positive area")
    if rotation_degrees == 0:
        return bbox_xyxy_px
    if rotation_degrees == 90:
        return y1, source_height_px - x2, y2, source_height_px - x1
    if rotation_degrees == 180:
        return (
            source_width_px - x2,
            source_height_px - y2,
            source_width_px - x1,
            source_height_px - y1,
        )
    return source_width_px - y2, x1, source_width_px - y1, x2


def _temporal_json(metadata: HandTemporalMetadata | None) -> dict[str, object] | None:
    """Serialize offline provenance without exposing NumPy implementation details."""

    if metadata is None:
        return None
    return {
        "temporal_source": metadata.source.value,
        "interpolation_source_frames": (
            list(metadata.interpolation_source_frames)
            if metadata.interpolation_source_frames is not None
            else None
        ),
        "vio_pose_timestamp_ns": metadata.vio_pose_timestamp_ns,
        "vio_time_difference_ns": metadata.vio_time_difference_ns,
        "world_kinematics_available": metadata.world_kinematics_available,
        "kinematics_optimized": metadata.kinematics_optimized,
    }


def _kinematics_json(kinematics: HandKinematics | None) -> dict[str, object] | None:
    """Serialize all raw and optimized world-space values for offline replay."""

    if kinematics is None:
        return None
    return {
        "keypoints_3d_world_m": kinematics.keypoints_3d_world_m.tolist(),
        "wrist_pose_raw_world": kinematics.wrist_pose_raw_world.tolist(),
        "wrist_pose_optimized_world": kinematics.wrist_pose_optimized_world.tolist(),
        "midpoint_pose_raw_world": kinematics.midpoint_pose_raw_world.tolist(),
        "midpoint_pose_optimized_world": kinematics.midpoint_pose_optimized_world.tolist(),
        "thumb_tip_raw_world_m": kinematics.thumb_tip_raw_world_m.tolist(),
        "thumb_tip_optimized_world_m": kinematics.thumb_tip_optimized_world_m.tolist(),
        "index_tip_raw_world_m": kinematics.index_tip_raw_world_m.tolist(),
        "index_tip_optimized_world_m": kinematics.index_tip_optimized_world_m.tolist(),
        "thumb_base_raw_world_m": kinematics.thumb_base_raw_world_m.tolist(),
        "thumb_base_optimized_world_m": kinematics.thumb_base_optimized_world_m.tolist(),
        "index_base_raw_world_m": kinematics.index_base_raw_world_m.tolist(),
        "index_base_optimized_world_m": kinematics.index_base_optimized_world_m.tolist(),
        "wrist_linear_velocity_raw_m_s": kinematics.wrist_linear_velocity_raw_m_s.tolist(),
        "wrist_linear_velocity_optimized_m_s": (
            kinematics.wrist_linear_velocity_optimized_m_s.tolist()
        ),
        "wrist_angular_velocity_raw_rad_s": kinematics.wrist_angular_velocity_raw_rad_s.tolist(),
        "wrist_angular_velocity_optimized_rad_s": (
            kinematics.wrist_angular_velocity_optimized_rad_s.tolist()
        ),
        "midpoint_linear_velocity_raw_m_s": kinematics.midpoint_linear_velocity_raw_m_s.tolist(),
        "midpoint_linear_velocity_optimized_m_s": (
            kinematics.midpoint_linear_velocity_optimized_m_s.tolist()
        ),
        "midpoint_angular_velocity_raw_rad_s": (
            kinematics.midpoint_angular_velocity_raw_rad_s.tolist()
        ),
        "midpoint_angular_velocity_optimized_rad_s": (
            kinematics.midpoint_angular_velocity_optimized_rad_s.tolist()
        ),
    }


class HandDetector(Protocol):
    """Detector boundary implemented by ViTPose and MediaPipe adapters."""

    @property
    def name(self) -> str: ...

    def detect(self, image_rgb: NDArray[np.uint8]) -> Sequence[DetectedHand]: ...


class HandReconstructor(Protocol):
    """3D reconstruction boundary implemented by the HaMeR adapter."""

    @property
    def name(self) -> str: ...

    @property
    def execution_device(self) -> str: ...

    @property
    def is_available(self) -> bool: ...

    @property
    def amp_enabled(self) -> bool: ...

    def predict_batch(
        self,
        image_rgb: NDArray[np.uint8],
        detections: Sequence[DetectedHand],
    ) -> Sequence[HandReconstruction | None]: ...


def readonly_float_array(values: object, shape: tuple[int, ...]) -> FloatArray:
    """Create a finite, C-contiguous, immutable float32 array."""

    array = np.array(values, dtype=np.float32, copy=True, order="C")
    if array.shape != shape:
        raise ValueError(f"array must have shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError("array must contain finite values")
    array.setflags(write=False)
    _validate_array(array, shape, "array")
    return array


def remap_hamer_to_humanego_aria(
    keypoints: NDArray[np.floating],
) -> FloatArray:
    """Remap MediaPipe/HaMeR's 21 joints to HumanEgo's Aria-compatible order."""

    source = np.asarray(keypoints, dtype=np.float32)
    if source.shape not in {(21, 2), (21, 3)}:
        raise ValueError("hand keypoints must have shape (21, 2) or (21, 3)")
    result = np.empty_like(source)
    result[:20] = source[np.asarray(_MP_TO_HUMANEGO_ARIA)]
    result[20] = (source[0] + source[5] + source[9]) / 3.0
    return readonly_float_array(result, result.shape)


def detector_bbox_has_valid_geometry(
    bbox_xyxy_px: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    config: HandTrackingConfig,
) -> bool:
    """Reject tiny or near-full-frame ViTPose egocentric fallback boxes."""

    x1, y1, x2, y2 = bbox_xyxy_px
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    minimum_dimension_ratio = min(width / image_width, height / image_height)
    area_ratio = width * height / (image_width * image_height)
    return (
        minimum_dimension_ratio >= config.detector_min_bbox_dimension_ratio
        and area_ratio <= config.detector_max_bbox_area_ratio
    )


def _validate_array(array: FloatArray, shape: tuple[int, ...], label: str) -> None:
    if not isinstance(array, np.ndarray) or array.dtype != np.float32:
        raise TypeError(f"{label} must be a float32 NumPy array")
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain finite values")
    if array.flags.writeable:
        raise ValueError(f"{label} must be read-only")
