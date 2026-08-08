"""CoTracker3 offline adapter with deterministic protocol boundaries."""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import ObjectTrackingConfig, ObjectTrackingError


class PointTracker(Protocol):
    def track(
        self,
        images_bgr: list[NDArray[np.uint8]],
        initial_points_xy_px: NDArray[np.float32],
        reference_index: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...


class CoTracker3Tracker:
    """HumanEgo-style bidirectional point tracker for one object-centric window."""

    def __init__(self, config: ObjectTrackingConfig) -> None:
        self.config = config
        self._model: object | None = None
        self._torch: object | None = None
        self._device = config.device

    def track(
        self,
        images_bgr: list[NDArray[np.uint8]],
        initial_points_xy_px: NDArray[np.float32],
        reference_index: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if not images_bgr:
            raise ValueError("CoTracker requires at least one image")
        if initial_points_xy_px.ndim != 2 or initial_points_xy_px.shape[1] != 2:
            raise ValueError("initial points must have shape (N, 2)")
        if not 0 <= reference_index < len(images_bgr):
            raise ValueError("CoTracker reference index is outside the video")
        self._load()
        square_frames, metadata = zip(
            *(self._letterbox(image) for image in images_bgr), strict=True
        )
        reference_points = self._letterbox_points(initial_points_xy_px, metadata[reference_index])
        tracks = np.zeros((len(images_bgr), len(initial_points_xy_px), 2), dtype=np.float32)
        visibility = np.zeros((len(images_bgr), len(initial_points_xy_px)), dtype=np.float32)
        self._forward(square_frames, reference_points, reference_index, tracks, visibility)
        if reference_index > 0:
            self._backward(square_frames, reference_points, reference_index, tracks, visibility)
        return self._unletterbox_per_frame(tracks, metadata), visibility

    def close(self) -> None:
        if self._model is not None:
            self._model.to("cpu")
        self._model = self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from cotracker.predictor import CoTrackerPredictor
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ObjectTrackingError(
                "CoTracker3 is missing. Run scripts/setup_client.ps1 after the "
                "object-tracking dependency update."
            ) from exc
        if self.config.require_cuda and not torch.cuda.is_available():
            raise ObjectTrackingError("CoTracker requires CUDA but no CUDA device is available")
        checkpoint = hf_hub_download(
            repo_id=self.config.cotracker_repository,
            filename=self.config.cotracker_checkpoint_name,
            revision=self.config.cotracker_model_revision,
        )
        device = self.config.device if torch.cuda.is_available() else "cpu"
        self._device = device
        self._torch = torch
        self._model = CoTrackerPredictor(checkpoint=checkpoint).to(device)

    def _forward(
        self,
        frames: tuple[NDArray[np.uint8], ...],
        reference_points: NDArray[np.float32],
        start: int,
        tracks: NDArray[np.float32],
        visibility: NDArray[np.float32],
    ) -> None:
        queries = reference_points
        stride = max(1, self.config.cotracker_chunk_size - 1)
        for begin in range(start, max(start + 1, len(frames) - 1), stride):
            end = min(begin + self.config.cotracker_chunk_size, len(frames))
            chunk_tracks, chunk_visibility = self._infer_chunk(frames[begin:end], queries)
            tracks[begin:end] = chunk_tracks
            visibility[begin:end] = chunk_visibility
            queries = chunk_tracks[-1]
            if end == len(frames):
                break

    def _backward(
        self,
        frames: tuple[NDArray[np.uint8], ...],
        reference_points: NDArray[np.float32],
        start: int,
        tracks: NDArray[np.float32],
        visibility: NDArray[np.float32],
    ) -> None:
        reverse = list(range(start, -1, -1))
        queries = reference_points
        stride = max(1, self.config.cotracker_chunk_size - 1)
        for begin in range(0, max(1, len(reverse) - 1), stride):
            indices = reverse[begin : begin + self.config.cotracker_chunk_size]
            chunk_tracks, chunk_visibility = self._infer_chunk(
                tuple(frames[index] for index in indices), queries
            )
            for local_index, global_index in enumerate(indices):
                tracks[global_index] = chunk_tracks[local_index]
                visibility[global_index] = chunk_visibility[local_index]
            queries = chunk_tracks[-1]
            if indices[-1] == 0:
                break

    def _infer_chunk(
        self,
        frames: tuple[NDArray[np.uint8], ...],
        points_xy: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        assert self._model is not None and self._torch is not None
        torch = self._torch
        video = (
            torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].float().to(self._device)
        )
        queries = torch.from_numpy(
            np.column_stack((np.zeros(len(points_xy), dtype=np.float32), points_xy))
        )[None].to(self._device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self._device == "cuda"):
            predicted_tracks, predicted_visibility = self._model(video, queries=queries)
        return (
            predicted_tracks[0].detach().float().cpu().numpy(),
            predicted_visibility[0].detach().float().cpu().numpy(),
        )

    def _letterbox(
        self, image_bgr: NDArray[np.uint8]
    ) -> tuple[NDArray[np.uint8], tuple[float, int, int]]:
        height, width = image_bgr.shape[:2]
        resolution = self.config.cotracker_resolution
        scale = resolution / max(height, width)
        resized = cv2.resize(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            (round(width * scale), round(height * scale)),
        )
        pad_x = resolution - resized.shape[1]
        pad_y = resolution - resized.shape[0]
        left, top = pad_x // 2, pad_y // 2
        return (
            cv2.copyMakeBorder(
                resized,
                top,
                pad_y - top,
                left,
                pad_x - left,
                cv2.BORDER_CONSTANT,
            ),
            (scale, left, top),
        )

    @staticmethod
    def _letterbox_points(
        points: NDArray[np.float32], metadata: tuple[float, int, int]
    ) -> NDArray[np.float32]:
        scale, left, top = metadata
        output = points.astype(np.float32).copy()
        output[:, 0] = output[:, 0] * scale + left
        output[:, 1] = output[:, 1] * scale + top
        return output

    @staticmethod
    def _unletterbox_per_frame(
        tracks: NDArray[np.float32], metadata: tuple[tuple[float, int, int], ...]
    ) -> NDArray[np.float32]:
        output = tracks.copy()
        for index, (scale, left, top) in enumerate(metadata):
            output[index, :, 0] = (output[index, :, 0] - left) / scale
            output[index, :, 1] = (output[index, :, 1] - top) / scale
        return output
