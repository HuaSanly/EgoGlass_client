"""EgoGlass boundary around the HumanEgo HaMeR inference chain."""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Iterable, Iterator, Mapping

import numpy as np

from perception.sensor_preprocessing import PreparedFrameBundle

from .models import (
    DetectedHand,
    HandDetector,
    Handedness,
    HandReconstruction,
    HandReconstructor,
    HandTrackingConfig,
    HandTrackingError,
    HandTrackingResult,
    MetricDepthStatus,
    ReconstructionBackend,
    TrackedHand,
    readonly_float_array,
    remap_hamer_to_humanego_aria,
)

LOGGER = logging.getLogger(__name__)


class HumanEgoHandTrackingPipeline:
    """Run HumanEgo's detector -> HaMeR chain on prepared EgoGlass frames."""

    def __init__(
        self,
        config: HandTrackingConfig,
        detector: HandDetector,
        reconstructor: HandReconstructor | None,
    ) -> None:
        if config.require_hamer and (
            reconstructor is None or not reconstructor.is_available
        ):
            raise HandTrackingError("configuration requires HaMeR but it is unavailable")
        self.config = config
        self.detector = detector
        self.reconstructor = reconstructor

    @classmethod
    def from_config_file(cls, path: str) -> HumanEgoHandTrackingPipeline:
        """Load config, pinned weights, HumanEgo detector, and HaMeR model."""

        from .humanego_hamer import (
            HumanEgoHaMeRModel,
            HumanEgoMediaPipeDetector,
            HumanEgoViTPoseDetector,
        )
        from .weights import ensure_hand_tracking_weights

        config = HandTrackingConfig.load(path)
        weights = ensure_hand_tracking_weights(config)
        detector: HandDetector
        if config.detector == "vitpose":
            try:
                detector = HumanEgoViTPoseDetector(config, weights)
            except Exception as exc:
                if config.fallback_detector != "mediapipe":
                    raise HandTrackingError("ViTPose detector failed to initialize") from exc
                if weights.mediapipe_model is None:
                    raise HandTrackingError("MediaPipe fallback weights are missing") from exc
                LOGGER.warning(
                    "hand_tracking_detector_fallback from=vitpose to=mediapipe reason=%r",
                    exc,
                )
                detector = HumanEgoMediaPipeDetector(weights.mediapipe_model)
        else:
            if weights.mediapipe_model is None:
                raise HandTrackingError("MediaPipe detector weights are missing")
            detector = HumanEgoMediaPipeDetector(weights.mediapipe_model)

        reconstructor: HandReconstructor | None
        try:
            reconstructor = HumanEgoHaMeRModel(config, weights)
        except Exception as exc:
            if config.require_hamer:
                raise HandTrackingError("HaMeR failed to initialize") from exc
            LOGGER.warning("hand_tracking_hamer_unavailable reason=%r", exc)
            reconstructor = None
        return cls(config, detector, reconstructor)

    def process_frame(self, bundle: PreparedFrameBundle) -> HandTrackingResult:
        """Track hands in one rectified frame without inventing a world pose."""

        if not isinstance(bundle, PreparedFrameBundle):
            raise TypeError("bundle must be a PreparedFrameBundle")
        started_ns = time.perf_counter_ns()
        image_rgb = np.ascontiguousarray(bundle.image_bgr[:, :, ::-1])
        camera_matrix = np.asarray(
            bundle.calibration.rectified_camera_matrix,
            dtype=np.float32,
        )
        prepared_ns = time.perf_counter_ns()
        detections = tuple(self.detector.detect(image_rgb))
        detected_ns = time.perf_counter_ns()
        reconstructions: tuple[HandReconstruction | None, ...] = tuple(
            None for _ in detections
        )
        reconstruction_batch_size = 0
        if self.reconstructor is not None and self.reconstructor.is_available and detections:
            reconstruction_batch_size = len(detections)
            try:
                predicted = tuple(
                    self.reconstructor.predict_batch(image_rgb, detections)
                )
                if len(predicted) != len(detections):
                    raise HandTrackingError(
                        "HaMeR batch result count does not match detector input"
                    )
                reconstructions = predicted
            except Exception:
                LOGGER.exception(
                    "hand_tracking_batch_failed backend=hamer batch_size=%d",
                    len(detections),
                )
        reconstructed_ns = time.perf_counter_ns()
        best_by_side: dict[Handedness, TrackedHand] = {}
        for detection, reconstruction in zip(
            detections,
            reconstructions,
            strict=True,
        ):
            tracked = self._reconstruct_detection(
                camera_matrix,
                detection,
                reconstruction,
            )
            if tracked is None or tracked.confidence < self.config.minimum_hand_confidence:
                continue
            previous = best_by_side.get(tracked.handedness)
            if previous is None or tracked.confidence > previous.confidence:
                best_by_side[tracked.handedness] = tracked

        hands = tuple(
            best_by_side[side]
            for side in (Handedness.LEFT, Handedness.RIGHT)
            if side in best_by_side
        )
        postprocessed_ns = time.perf_counter_ns()
        frame_preparation_duration_ns = prepared_ns - started_ns
        detector_duration_ns = detected_ns - prepared_ns
        reconstruction_duration_ns = reconstructed_ns - detected_ns
        postprocessing_duration_ns = postprocessed_ns - reconstructed_ns
        duration_ns = postprocessed_ns - started_ns
        execution_device = (
            self.reconstructor.execution_device
            if self.reconstructor is not None
            else "none"
        )
        hamer_loaded = bool(
            self.reconstructor is not None and self.reconstructor.is_available
        )
        amp_enabled = bool(
            self.reconstructor is not None and self.reconstructor.amp_enabled
        )
        hamer_hand_count = sum(
            hand.reconstruction_backend is ReconstructionBackend.HAMER for hand in hands
        )
        LOGGER.debug(
            "hand_tracking_frame session_id=%s frame_index=%d detector=%s "
            "hamer_loaded=%s amp_enabled=%s batch_size=%d hamer_hands=%d "
            "fallback_hands=%d prepare_ms=%.3f detector_ms=%.3f "
            "reconstruction_ms=%.3f postprocess_ms=%.3f duration_ms=%.3f",
            bundle.session_id,
            bundle.frame_index,
            self.detector.name,
            hamer_loaded,
            amp_enabled,
            reconstruction_batch_size,
            hamer_hand_count,
            len(hands) - hamer_hand_count,
            frame_preparation_duration_ns / 1_000_000,
            detector_duration_ns / 1_000_000,
            reconstruction_duration_ns / 1_000_000,
            postprocessing_duration_ns / 1_000_000,
            duration_ns / 1_000_000,
        )
        return HandTrackingResult(
            schema_version="1.0",
            session_id=bundle.session_id,
            sequence_id=bundle.sequence_id,
            frame_index=bundle.frame_index,
            session_time_ns=bundle.session_time_ns,
            timestamp_uncertainty_ns=bundle.timestamp_uncertainty_ns,
            image_width_px=int(bundle.image_bgr.shape[1]),
            image_height_px=int(bundle.image_bgr.shape[0]),
            source_rotation_degrees=bundle.calibration.rotation_degrees,
            detector_backend=self.detector.name,
            requested_device=self.config.device,
            execution_device=execution_device,
            hamer_loaded=hamer_loaded,
            inference_duration_ns=duration_ns,
            hands=hands,
            frame_preparation_duration_ns=frame_preparation_duration_ns,
            detector_duration_ns=detector_duration_ns,
            reconstruction_duration_ns=reconstruction_duration_ns,
            postprocessing_duration_ns=postprocessing_duration_ns,
            reconstruction_batch_size=reconstruction_batch_size,
            amp_enabled=amp_enabled,
        )

    def process_frames(
        self,
        bundles: Iterable[PreparedFrameBundle],
    ) -> Iterator[HandTrackingResult]:
        """Process an offline replay or live stream with the same frame contract."""

        for bundle in bundles:
            yield self.process_frame(bundle)

    def _reconstruct_detection(
        self,
        camera_matrix: np.ndarray,
        detection: DetectedHand,
        reconstruction: HandReconstruction | None,
    ) -> TrackedHand | None:
        detector_confidence = float(detection.confidence)
        reconstruction_quality: float | None = None
        depth_score: float | None = None
        coverage_score: float | None = None
        compactness_score: float | None = None
        if reconstruction is not None:
            keypoints_3d = reconstruction.keypoints_3d_m
            keypoints_2d = reconstruction.keypoints_2d_px
            reconstruction_quality = float(reconstruction.confidence)
            depth_score = float(reconstruction.depth_score)
            coverage_score = float(reconstruction.coverage_score)
            compactness_score = float(reconstruction.compactness_score)
            confidence = float(detector_confidence * reconstruction_quality)
            depth_status = MetricDepthStatus.MODEL_ESTIMATED
            wrist_depth = float(keypoints_3d[0, 2])
            if not self.config.minimum_depth_m <= wrist_depth <= self.config.maximum_depth_m:
                recovered = _recover_absolute_3d(
                    keypoints_3d,
                    detection.keypoints_2d_px,
                    camera_matrix,
                    self.config,
                )
                if recovered is None:
                    return None
                keypoints_3d = recovered
                confidence = detector_confidence
                depth_status = MetricDepthStatus.PHYSICAL_SIZE_ESTIMATED
            backend = ReconstructionBackend.HAMER
        elif (
            self.config.allow_mediapipe_reconstruction_fallback
            and detection.relative_keypoints_3d_m is not None
        ):
            recovered = _recover_absolute_3d(
                detection.relative_keypoints_3d_m,
                detection.keypoints_2d_px,
                camera_matrix,
                self.config,
            )
            if recovered is None:
                return None
            keypoints_3d = recovered
            keypoints_2d = detection.keypoints_2d_px
            confidence = detector_confidence
            backend = ReconstructionBackend.MEDIAPIPE
            depth_status = MetricDepthStatus.PHYSICAL_SIZE_ESTIMATED
        else:
            return None

        keypoints_3d_aria = remap_hamer_to_humanego_aria(keypoints_3d)
        keypoints_2d_aria = remap_hamer_to_humanego_aria(keypoints_2d)
        joint_angles = _joint_angles_degrees(keypoints_3d_aria)
        thumb_to_index = float(
            np.linalg.norm(keypoints_3d_aria[0] - keypoints_3d_aria[1])
        )
        palm_size = float(
            np.linalg.norm(keypoints_3d_aria[11] - keypoints_3d_aria[5])
        )
        grasp_ratio = (
            thumb_to_index / palm_size
            if palm_size > 0.01
            else thumb_to_index / self.config.physical_wrist_to_middle_mcp_m
        )
        return TrackedHand(
            handedness=detection.handedness,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            detector_confidence=detector_confidence,
            reconstruction_quality=reconstruction_quality,
            depth_score=depth_score,
            coverage_score=coverage_score,
            compactness_score=compactness_score,
            reconstruction_backend=backend,
            metric_depth_status=depth_status,
            bbox_xyxy_px=detection.bbox_xyxy_px,
            keypoints_2d_px=keypoints_2d_aria,
            keypoints_3d_camera_m=keypoints_3d_aria,
            joint_angles_degrees=joint_angles,
            grasp_ratio=grasp_ratio,
            is_grasping=grasp_ratio < self.config.grasp_ratio_threshold,
        )


def release_pipeline_resources(
    pipeline: HumanEgoHandTrackingPipeline | None,
) -> None:
    """Drop one inactive model graph and return its cached CUDA memory."""

    if pipeline is None:
        return
    del pipeline
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _recover_absolute_3d(
    relative_keypoints_3d: np.ndarray,
    detected_keypoints_2d: np.ndarray,
    camera_matrix: np.ndarray,
    config: HandTrackingConfig,
) -> np.ndarray | None:
    """Apply HumanEgo's physical-size depth estimate in rectified camera space."""

    wrist_2d = detected_keypoints_2d[0]
    middle_mcp_2d = detected_keypoints_2d[9]
    physical_distance = float(
        np.linalg.norm(relative_keypoints_3d[9] - relative_keypoints_3d[0])
    )
    if physical_distance < 0.01:
        physical_distance = config.physical_wrist_to_middle_mcp_m
    pixel_distance = float(np.linalg.norm(middle_mcp_2d - wrist_2d))
    if pixel_distance < 5.0:
        return None
    focal_x = float(camera_matrix[0, 0])
    focal_y = float(camera_matrix[1, 1])
    depth = ((focal_x + focal_y) / 2.0) * physical_distance / pixel_distance
    if not config.minimum_depth_m <= depth <= config.maximum_depth_m:
        return None
    center_x = float(camera_matrix[0, 2])
    center_y = float(camera_matrix[1, 2])
    wrist_camera = np.asarray(
        [
            (float(wrist_2d[0]) - center_x) * depth / focal_x,
            (float(wrist_2d[1]) - center_y) * depth / focal_y,
            depth,
        ],
        dtype=np.float32,
    )
    camera_keypoints = (
        wrist_camera[np.newaxis, :]
        + relative_keypoints_3d
        - relative_keypoints_3d[0:1]
    )
    camera_keypoints[:, 2] = np.clip(camera_keypoints[:, 2], 0.01, None)
    return readonly_float_array(camera_keypoints, (21, 3))


def _joint_angles_degrees(keypoints: np.ndarray) -> Mapping[str, float]:
    """Compute HumanEgo's 20 flexion/abduction values from its joint order."""

    def angle(first: np.ndarray, second: np.ndarray) -> float:
        first_norm = first / (np.linalg.norm(first) + 1e-6)
        second_norm = second / (np.linalg.norm(second) + 1e-6)
        return float(
            np.degrees(np.arccos(np.clip(np.dot(first_norm, second_norm), -1.0, 1.0)))
        )

    def abduction(vector: np.ndarray, reference: np.ndarray, normal: np.ndarray) -> float:
        def project(value: np.ndarray) -> np.ndarray:
            return value - np.dot(value, normal) * normal

        return angle(project(vector), project(reference))

    wrist_to_middle = keypoints[11] - keypoints[5]
    wrist_to_index = keypoints[8] - keypoints[5]
    palm_normal = np.cross(wrist_to_index, wrist_to_middle)
    palm_normal /= np.linalg.norm(palm_normal) + 1e-6
    middle_reference = keypoints[12] - keypoints[11]
    result: dict[str, float] = {}
    fingers = {
        "index": (8, 9, 10, 1),
        "middle": (11, 12, 13, 2),
        "ring": (14, 15, 16, 3),
        "pinky": (17, 18, 19, 4),
    }
    for name, (mcp, pip, dip, tip) in fingers.items():
        metacarpal = keypoints[mcp] - keypoints[5]
        proximal = keypoints[pip] - keypoints[mcp]
        intermediate = keypoints[dip] - keypoints[pip]
        distal = keypoints[tip] - keypoints[dip]
        result[f"{name}_mcp_flex"] = angle(metacarpal, proximal)
        result[f"{name}_pip_flex"] = angle(proximal, intermediate)
        result[f"{name}_dip_flex"] = angle(intermediate, distal)
        result[f"{name}_mcp_abduction"] = (
            0.0 if name == "middle" else abduction(proximal, middle_reference, palm_normal)
        )

    thumb_metacarpal = keypoints[6] - keypoints[5]
    thumb_proximal = keypoints[7] - keypoints[6]
    thumb_distal = keypoints[0] - keypoints[7]
    result["thumb_cmc_flex"] = angle(keypoints[11] - keypoints[5], thumb_metacarpal)
    result["thumb_cmc_abduction"] = abduction(
        thumb_metacarpal,
        middle_reference,
        palm_normal,
    )
    result["thumb_mcp_flex"] = angle(thumb_metacarpal, thumb_proximal)
    result["thumb_ip_flex"] = angle(thumb_proximal, thumb_distal)
    return result
