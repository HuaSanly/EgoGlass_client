"""Deterministic contour keypoint selection adapted from HumanEgo KptsSelector."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import ObjectTrackingConfig


class ContourKeypointSelector:
    """Select evenly spaced, inside-contour points suitable for CoTracker queries."""

    def __init__(self, config: ObjectTrackingConfig) -> None:
        self.config = config

    def select(self, mask: NDArray[np.uint8]) -> NDArray[np.float32]:
        if mask.ndim != 2:
            raise ValueError("object mask must be a single-channel image")
        cleaned = self._preprocess(mask)
        safe = self._inner_edge(cleaned)
        contours, _ = cv2.findContours(safe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise ValueError("object mask has no usable contour")
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
        if len(contour) < 3:
            raise ValueError("object contour is too short")
        top_index = int(np.lexsort((contour[:, 0], contour[:, 1]))[0])
        contour = np.roll(contour, -top_index, axis=0)
        closed = np.vstack((contour, contour[:1]))
        distances = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        cumulative = np.concatenate((np.array([0.0], dtype=np.float32), np.cumsum(distances)))
        perimeter = float(cumulative[-1])
        if perimeter <= 1e-6:
            raise ValueError("object contour has zero perimeter")
        targets = np.linspace(0.0, perimeter, self.config.keypoint_count, endpoint=False)
        selected: list[NDArray[np.float32]] = []
        for target in targets:
            index = max(0, min(len(contour) - 1, int(np.searchsorted(cumulative, target) - 1)))
            next_index = (index + 1) % len(contour)
            start_distance = float(cumulative[index])
            segment = float(distances[index])
            alpha = 0.0 if segment <= 1e-6 else (float(target) - start_distance) / segment
            selected.append(contour[index] * (1.0 - alpha) + contour[next_index] * alpha)
        return np.asarray(selected, dtype=np.float32)

    def _preprocess(self, mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
        binary = (mask > 127).astype(np.uint8) * 255
        close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morphology_close_kernel, self.config.morphology_close_kernel),
        )
        erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morphology_erode_kernel, self.config.morphology_erode_kernel),
        )
        return cv2.erode(cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close), erode, iterations=1)

    def _inner_edge(self, mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.inner_edge_kernel, self.config.inner_edge_kernel),
        )
        inner = cv2.erode(mask, kernel, iterations=1)
        return inner if int(np.count_nonzero(inner)) >= self.config.keypoint_count else mask
