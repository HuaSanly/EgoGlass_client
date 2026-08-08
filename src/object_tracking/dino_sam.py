"""Grounding DINO plus SAM2 adapter with a model-free protocol for gate tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from schemas.object_tracking import BoundingBox

from .config import ObjectTrackingConfig, ObjectTrackingError


@dataclass(frozen=True, slots=True)
class SegmentationPrediction:
    mask: NDArray[np.uint8]
    boxes: tuple[BoundingBox, ...]

    def __post_init__(self) -> None:
        if self.mask.ndim != 2:
            raise ValueError("segmentation mask must be single-channel")


class ObjectSegmenter(Protocol):
    def segment(self, image_bgr: NDArray[np.uint8], prompt: str) -> SegmentationPrediction: ...


class DinoSamSegmenter:
    """Load Grounding DINO and SAM2 only in an offline object-processing stage."""

    def __init__(self, config: ObjectTrackingConfig) -> None:
        self.config = config
        self._processor: object | None = None
        self._dino: object | None = None
        self._predictor: object | None = None
        self._torch: object | None = None
        self._device = config.device

    def segment(self, image_bgr: NDArray[np.uint8], prompt: str) -> SegmentationPrediction:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("DINO-SAM expects BGR image data")
        self._load()
        assert (
            self._processor is not None and self._dino is not None and self._predictor is not None
        )
        import cv2
        from PIL import Image

        torch = self._torch
        assert torch is not None
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        width, height = image.size
        inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._dino(**inputs)
        logits = outputs.logits.sigmoid()[0]
        boxes = outputs.pred_boxes[0]
        keep = logits.max(-1)[0] > self.config.box_threshold
        boxes = boxes[keep]
        confidences = logits[keep].max(-1)[0]
        if len(boxes) == 0:
            return SegmentationPrediction(
                mask=np.zeros(image_bgr.shape[:2], dtype=np.uint8),
                boxes=(),
            )
        scale = torch.tensor([width, height, width, height], device=self._device)
        center_xywh = boxes * scale
        center_x, center_y, box_width, box_height = center_xywh.unbind(-1)
        xyxy = (
            torch.stack(
                (
                    center_x - box_width / 2.0,
                    center_y - box_height / 2.0,
                    center_x + box_width / 2.0,
                    center_y + box_height / 2.0,
                ),
                dim=-1,
            )
            .detach()
            .cpu()
            .numpy()
        )
        self._predictor.set_image(image_rgb)
        masks, _, _ = self._predictor.predict(box=xyxy, multimask_output=False)
        binary = _merge_sam_masks(np.asarray(masks))
        payload = tuple(
            BoundingBox(
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3]),
                confidence=float(confidence),
            )
            for box, confidence in zip(xyxy, confidences.detach().cpu().numpy(), strict=True)
        )
        return SegmentationPrediction(mask=binary, boxes=payload)

    def close(self) -> None:
        if self._dino is not None:
            self._dino.to("cpu")
        predictor_model = getattr(self._predictor, "model", None)
        if predictor_model is not None:
            predictor_model.to("cpu")
        self._processor = self._dino = self._predictor = self._torch = None

    def _load(self) -> None:
        if self._predictor is not None:
            return
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise ObjectTrackingError(
                "DINO-SAM dependencies are missing. Run scripts/setup_client.ps1 after "
                "the object-tracking dependency update."
            ) from exc
        if self.config.require_cuda and not torch.cuda.is_available():
            raise ObjectTrackingError(
                "object tracking requires CUDA but no CUDA device is available"
            )
        device = self.config.device if torch.cuda.is_available() else "cpu"
        self._device = device
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(
            self.config.dino_model_id,
            revision=self.config.dino_model_revision,
        )
        self._dino = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.config.dino_model_id,
            revision=self.config.dino_model_revision,
        ).to(device)
        checkpoint = hf_hub_download(
            repo_id=self.config.sam2_repo_id,
            filename=self.config.sam2_checkpoint_name,
            revision=self.config.sam2_model_revision,
        )
        self._predictor = SAM2ImagePredictor(
            build_sam2(self.config.sam2_config, checkpoint, device=device)
        )


def _merge_sam_masks(masks: NDArray[np.bool_ | np.uint8]) -> NDArray[np.uint8]:
    """Normalize SAM2's version-dependent mask batch shape to one binary mask."""

    values = np.asarray(masks)
    if values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 3:
        raise ObjectTrackingError(
            f"SAM2 returned unsupported mask shape {tuple(values.shape)}"
        )
    return np.any(values > 0, axis=0).astype(np.uint8) * 255
