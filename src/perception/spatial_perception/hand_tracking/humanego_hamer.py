"""HumanEgo's ViTPose/MediaPipe + HaMeR inference core, adapted for EgoGlass.

This module is derived from HumanEgo ``preprocess/HaMeRHands.py`` at commit
``18fb1082abb87b79f88e53f2abb5bfb9f61de19b``. HumanEgo is licensed under
PolyForm Noncommercial 1.0.0; the required license is stored beside this file.
The model inference and joint-remapping steps are retained. Aria/MPS storage,
world transforms, plotting, and CLI execution were removed at the boundary.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

# HaMeR defaults to EGL before importing pyrender. Windows uses native OpenGL,
# and inference does not construct a renderer.
os.environ.setdefault("PYOPENGL_PLATFORM", "win32")

import numpy as np
import torch
from numpy.typing import NDArray

from .models import (
    DetectedHand,
    Handedness,
    HandReconstruction,
    HandTrackingConfig,
    HandTrackingError,
    detector_bbox_has_valid_geometry,
    readonly_float_array,
)
from .weights import HandTrackingWeights

LOGGER = logging.getLogger(__name__)


class _ViTPosePostprocessor(Protocol):
    postprocess: Callable[..., object]


@dataclass(frozen=True, slots=True)
class _HaMeRQualityScores:
    reconstruction_quality: float
    depth_score: float
    coverage_score: float
    compactness_score: float


class HumanEgoHaMeRModel:
    """Thin HaMeR wrapper following HumanEgo's crop inference path."""

    def __init__(
        self,
        config: HandTrackingConfig,
        weights: HandTrackingWeights,
    ) -> None:
        if config.require_cuda and not torch.cuda.is_available():
            raise HandTrackingError("CUDA is required but PyTorch cannot access it")
        requested_device = config.device
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise HandTrackingError("hand tracking requested CUDA but it is unavailable")
        self._device = torch.device(requested_device)
        self._amp_enabled = _cuda_amp_enabled(config, self._device)
        self._model, self._cfg = _load_hamer_without_renderer(
            weights.hamer_checkpoint,
            weights.hamer_root,
        )
        self._model = self._model.to(self._device)
        self._model.eval()
        LOGGER.info(
            "hand_tracking_model_loaded backend=hamer device=%s amp_enabled=%s "
            "checkpoint=%s",
            self._device,
            self._amp_enabled,
            weights.hamer_checkpoint,
        )

    @property
    def name(self) -> str:
        return "hamer"

    @property
    def execution_device(self) -> str:
        return str(self._device)

    @property
    def is_available(self) -> bool:
        return self._model is not None

    @property
    def amp_enabled(self) -> bool:
        return self._amp_enabled

    @torch.inference_mode()
    def predict_from_crop(
        self,
        image_rgb: NDArray[np.uint8],
        detection: DetectedHand,
    ) -> HandReconstruction | None:
        """Compatibility wrapper for callers that still submit one detector crop."""

        return self.predict_batch(image_rgb, (detection,))[0]

    @torch.inference_mode()
    def predict_batch(
        self,
        image_rgb: NDArray[np.uint8],
        detections: Sequence[DetectedHand],
    ) -> tuple[HandReconstruction | None, ...]:
        """Run all valid detector crops from one frame in a single HaMeR forward."""

        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.utils import recursive_to
        from hamer.utils.renderer import cam_crop_to_full
        from torch.utils.data import default_collate

        results: list[HandReconstruction | None] = [None] * len(detections)
        valid_indices: list[int] = []
        valid_boxes: list[NDArray[np.float32]] = []
        valid_handedness: list[float] = []
        for index, detection in enumerate(detections):
            bbox = np.asarray(detection.bbox_xyxy_px, dtype=np.float32)
            x1, y1, x2, y2 = bbox.astype(int)
            if max(x2 - x1, y2 - y1) < 10:
                continue
            valid_indices.append(index)
            valid_boxes.append(np.asarray([x1, y1, x2, y2], dtype=np.float32))
            valid_handedness.append(
                1.0 if detection.handedness is Handedness.RIGHT else 0.0
            )
        if not valid_indices:
            return tuple(results)

        image_height, image_width = image_rgb.shape[:2]
        dataset = ViTDetDataset(
            self._cfg,
            img_cv2=image_rgb[:, :, ::-1],
            boxes=np.stack(valid_boxes),
            right=np.asarray(valid_handedness, dtype=np.float32),
        )
        if not dataset:
            return tuple(results)
        batch = default_collate(
            [dataset[index] for index in range(len(dataset))]
        )
        batch = recursive_to(batch, self._device)
        with torch.autocast(
            device_type=self._device.type,
            dtype=torch.float16,
            enabled=self._amp_enabled,
        ):
            output = self._model(batch)

        pred_cam = output["pred_cam"].float().clone()
        relative_3d_batch = (
            output["pred_keypoints_3d"].float().detach().cpu().numpy()
        )
        multiplier = 2 * batch["right"].float() - 1
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]

        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        image_size = batch["img_size"].float()
        scaled_focal_length = (
            self._cfg.EXTRA.FOCAL_LENGTH
            / self._cfg.MODEL.IMAGE_SIZE
            * image_size.max()
        )
        camera_translations = cam_crop_to_full(
            pred_cam,
            box_center,
            box_size,
            image_size,
            scaled_focal_length,
        ).float().detach().cpu().numpy()
        focal = float(scaled_focal_length.float().detach().cpu())

        for batch_index, detection_index in enumerate(valid_indices):
            detection = detections[detection_index]
            relative_3d = relative_3d_batch[batch_index].copy()
            if detection.handedness is Handedness.LEFT:
                relative_3d[:, 0] = -relative_3d[:, 0]
            joints_camera = relative_3d + camera_translations[batch_index][np.newaxis, :]
            joints_2d = np.zeros((21, 2), dtype=np.float32)
            if np.all(joints_camera[:, 2] > 0):
                joints_2d[:, 0] = (
                    joints_camera[:, 0] / joints_camera[:, 2] * focal
                    + image_width / 2.0
                )
                joints_2d[:, 1] = (
                    joints_camera[:, 1] / joints_camera[:, 2] * focal
                    + image_height / 2.0
                )

            quality = _compute_hamer_quality_scores(
                relative_3d,
                joints_camera,
                joints_2d,
                np.asarray(detection.bbox_xyxy_px, dtype=np.float32),
            )
            results[detection_index] = HandReconstruction(
                keypoints_3d_m=readonly_float_array(joints_camera, (21, 3)),
                keypoints_2d_px=readonly_float_array(joints_2d, (21, 2)),
                confidence=quality.reconstruction_quality,
                depth_score=quality.depth_score,
                coverage_score=quality.coverage_score,
                compactness_score=quality.compactness_score,
            )
        return tuple(results)


class HumanEgoMediaPipeDetector:
    """MediaPipe Tasks detector used by HumanEgo as detector and 3D fallback."""

    def __init__(self, model_path: Path) -> None:
        import mediapipe as mp

        self._mp = mp
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    @property
    def name(self) -> str:
        return "mediapipe"

    def detect(self, image_rgb: NDArray[np.uint8]) -> tuple[DetectedHand, ...]:
        image_height, image_width = image_rgb.shape[:2]
        media_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(image_rgb),
        )
        result = self._landmarker.detect(media_image)
        detections: list[DetectedHand] = []
        for landmarks, world_landmarks, handedness in zip(
            result.hand_landmarks,
            result.hand_world_landmarks,
            result.handedness,
            strict=True,
        ):
            category = handedness[0]
            side = (
                Handedness.RIGHT
                if category.category_name.lower() == "right"
                else Handedness.LEFT
            )
            keypoints_2d = np.asarray(
                [[point.x * image_width, point.y * image_height] for point in landmarks],
                dtype=np.float32,
            )
            relative_3d = np.asarray(
                [[point.x, point.y, point.z] for point in world_landmarks],
                dtype=np.float32,
            )
            x_min, y_min = keypoints_2d.min(axis=0)
            x_max, y_max = keypoints_2d.max(axis=0)
            pad_x = (x_max - x_min) * 0.3
            pad_y = (y_max - y_min) * 0.3
            detections.append(
                DetectedHand(
                    bbox_xyxy_px=(
                        float(max(0.0, x_min - pad_x)),
                        float(max(0.0, y_min - pad_y)),
                        float(min(image_width, x_max + pad_x)),
                        float(min(image_height, y_max + pad_y)),
                    ),
                    handedness=side,
                    confidence=float(category.score),
                    keypoints_2d_px=readonly_float_array(keypoints_2d, (21, 2)),
                    relative_keypoints_3d_m=readonly_float_array(relative_3d, (21, 3)),
                )
            )
        return tuple(detections)


class HumanEgoViTPoseDetector:
    """YOLO + easy_ViTPose whole-body detector used by HumanEgo before HaMeR."""

    LEFT_HAND_SLICE = slice(91, 112)
    RIGHT_HAND_SLICE = slice(112, 133)

    def __init__(
        self,
        config: HandTrackingConfig,
        weights: HandTrackingWeights,
    ) -> None:
        from easy_ViTPose import VitInference

        if weights.vitpose_model is None or weights.yolo_model is None:
            raise HandTrackingError("ViTPose weights were not resolved")
        self._config = config
        self._device = torch.device(config.device)
        self._amp_enabled = _cuda_amp_enabled(config, self._device)
        self._model = VitInference(
            model=str(weights.vitpose_model),
            yolo=str(weights.yolo_model),
            model_name=config.vitpose_variant,
            dataset="wholebody",
            device=config.device,
        )
        _install_vitpose_fp32_postprocess(self._model)

    @property
    def name(self) -> str:
        return f"vitpose-{self._config.vitpose_variant}+yolov8s"

    @torch.inference_mode()
    def detect(self, image_rgb: NDArray[np.uint8]) -> tuple[DetectedHand, ...]:
        image_height, image_width = image_rgb.shape[:2]
        with torch.autocast(
            device_type=self._device.type,
            dtype=torch.float16,
            enabled=self._amp_enabled,
        ):
            keypoints = self._model.inference(image_rgb)
        egocentric_fallback = False
        confidence_threshold = self._config.detector_keypoint_confidence
        if not keypoints:
            from easy_ViTPose.vit_utils.inference import pad_image

            padded, (left_pad, top_pad) = pad_image(image_rgb, 3 / 4)
            with torch.autocast(
                device_type=self._device.type,
                dtype=torch.float16,
                enabled=self._amp_enabled,
            ):
                raw_keypoints = self._model._inference(padded)[0]
            raw_keypoints[:, :2] -= [top_pad, left_pad]
            keypoints = {0: raw_keypoints}
            egocentric_fallback = True
            confidence_threshold = min(confidence_threshold, 0.15)

        detections: list[DetectedHand] = []
        for person_keypoints in keypoints.values():
            candidates: list[DetectedHand] = []
            for hand_slice, handedness in (
                (self.LEFT_HAND_SLICE, Handedness.LEFT),
                (self.RIGHT_HAND_SLICE, Handedness.RIGHT),
            ):
                hand_keypoints = person_keypoints[hand_slice]
                confidence = hand_keypoints[:, 2]
                valid = confidence > confidence_threshold
                if valid.sum() < self._config.detector_min_valid_keypoints:
                    continue
                keypoints_2d = np.stack(
                    [hand_keypoints[:, 1], hand_keypoints[:, 0]],
                    axis=1,
                ).astype(np.float32)
                valid_points = keypoints_2d[valid]
                x_min, y_min = valid_points.min(axis=0)
                x_max, y_max = valid_points.max(axis=0)
                pad_x = (
                    (x_max - x_min) * self._config.detector_bbox_padding_ratio
                )
                pad_y = (
                    (y_max - y_min) * self._config.detector_bbox_padding_ratio
                )
                bbox = (
                    float(max(0.0, x_min - pad_x)),
                    float(max(0.0, y_min - pad_y)),
                    float(min(image_width, x_max + pad_x)),
                    float(min(image_height, y_max + pad_y)),
                )
                if bbox[2] - bbox[0] < 20 or bbox[3] - bbox[1] < 20:
                    continue
                if not detector_bbox_has_valid_geometry(
                    bbox,
                    image_width,
                    image_height,
                    self._config,
                ):
                    continue
                candidates.append(
                    DetectedHand(
                        bbox_xyxy_px=bbox,
                        handedness=handedness,
                        confidence=float(confidence[valid].mean()),
                        keypoints_2d_px=readonly_float_array(keypoints_2d, (21, 2)),
                        egocentric_fallback=egocentric_fallback,
                    )
                )
            detections.extend(_deduplicate_handedness_candidates(candidates))
        return tuple(detections)


def _load_hamer_without_renderer(checkpoint: Path, cache_root: Path):
    """Load the official checkpoint while skipping HaMeR's unused EGL renderer."""

    import hamer.configs
    from hamer.configs import get_config
    from hamer.models import HAMER

    hamer.configs.CACHE_DIR_HAMER = str(cache_root)
    config = get_config(
        str(checkpoint.parent.parent / "model_config.yaml"),
        update_cachedir=True,
    )
    if config.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in config.MODEL:
        config.defrost()
        if config.MODEL.IMAGE_SIZE != 256:
            raise HandTrackingError("HaMeR ViT checkpoint image size must be 256")
        config.MODEL.BBOX_SHAPE = [192, 256]
        config.freeze()
    if "PRETRAINED_WEIGHTS" in config.MODEL.BACKBONE:
        config.defrost()
        config.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
        config.freeze()
    model = HAMER.load_from_checkpoint(
        str(checkpoint),
        strict=False,
        cfg=config,
        init_renderer=False,
    )
    return model, config


def _cuda_amp_enabled(config: HandTrackingConfig, device: torch.device) -> bool:
    """Enable FP16 only on CUDA; CPU inference always stays in full precision."""

    return config.enable_cuda_amp and device.type == "cuda"


def _install_vitpose_fp32_postprocess(model: _ViTPosePostprocessor) -> None:
    """Keep OpenCV's heatmap postprocessing in FP32 after an AMP forward."""

    original_postprocess = model.postprocess

    def fp32_postprocess(
        heatmaps: object,
        original_width: int,
        original_height: int,
    ) -> object:
        if isinstance(heatmaps, torch.Tensor):
            heatmaps = heatmaps.float()
        elif isinstance(heatmaps, np.ndarray) and heatmaps.dtype == np.float16:
            heatmaps = heatmaps.astype(np.float32)
        return original_postprocess(heatmaps, original_width, original_height)

    model.postprocess = fp32_postprocess


def _compute_hamer_quality_scores(
    relative_3d: NDArray[np.floating],
    joints_camera: NDArray[np.floating],
    joints_2d: NDArray[np.floating],
    bbox: NDArray[np.floating],
) -> _HaMeRQualityScores:
    wrist_depth = float(joints_camera[0, 2])
    bbox_width = max(float(bbox[2] - bbox[0]), 1.0)
    bbox_height = max(float(bbox[3] - bbox[1]), 1.0)
    valid_2d = joints_2d[(joints_2d[:, 0] > 0) & (joints_2d[:, 1] > 0)]
    if len(valid_2d) > 5:
        coverage = (
            (float(np.ptp(valid_2d[:, 0])) / bbox_width)
            + (float(np.ptp(valid_2d[:, 1])) / bbox_height)
        ) / 2.0
        coverage_score = float(np.clip(coverage, 0.1, 1.0))
    else:
        coverage_score = 0.3
    hand_span = float(np.linalg.norm(relative_3d.max(axis=0) - relative_3d.min(axis=0)))
    compactness_score = 0.3 if hand_span < 0.05 or hand_span > 0.5 else 1.0
    if wrist_depth < 0.05 or wrist_depth > 3.0:
        depth_score = 0.15
        reconstruction_quality = 0.15
    else:
        depth_score = (
            0.5 if wrist_depth < 0.1 else 0.6 if wrist_depth > 2.0 else 1.0
        )
        reconstruction_quality = float(
            np.clip(
                0.95 * depth_score * coverage_score * compactness_score,
                0.1,
                0.99,
            )
        )
    return _HaMeRQualityScores(
        reconstruction_quality=reconstruction_quality,
        depth_score=depth_score,
        coverage_score=coverage_score,
        compactness_score=compactness_score,
    )


def _deduplicate_handedness_candidates(
    candidates: list[DetectedHand],
) -> tuple[DetectedHand, ...]:
    if len(candidates) != 2:
        return tuple(candidates)
    first, second = candidates
    first_box = first.bbox_xyxy_px
    second_box = second.bbox_xyxy_px
    intersection_width = max(
        0.0,
        min(first_box[2], second_box[2]) - max(first_box[0], second_box[0]),
    )
    intersection_height = max(
        0.0,
        min(first_box[3], second_box[3]) - max(first_box[1], second_box[1]),
    )
    intersection = intersection_width * intersection_height
    first_area = (first_box[2] - first_box[0]) * (first_box[3] - first_box[1])
    second_area = (second_box[2] - second_box[0]) * (second_box[3] - second_box[1])
    union = max(first_area + second_area - intersection, 1e-6)
    if intersection / union > 0.3:
        return (max(candidates, key=lambda candidate: candidate.confidence),)
    return tuple(candidates)
