"""Render hand-tracking outputs in their rectified camera-image coordinates."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .models import Handedness, HandTrackingResult

_BONES = (
    (5, 6),
    (6, 7),
    (7, 0),
    (5, 8),
    (8, 9),
    (9, 10),
    (10, 1),
    (5, 11),
    (11, 12),
    (12, 13),
    (13, 2),
    (5, 14),
    (14, 15),
    (15, 16),
    (16, 3),
    (5, 17),
    (17, 18),
    (18, 19),
    (19, 4),
)
_LEFT_COLOR = (250, 190, 52)
_RIGHT_COLOR = (77, 218, 130)


def render_hand_tracking_overlay(
    image_bgr: NDArray[np.uint8],
    result: HandTrackingResult,
) -> NDArray[np.uint8]:
    """Draw the latest 2D skeleton, labels, and confidence onto a BGR frame."""

    if image_bgr.shape != (result.image_height_px, result.image_width_px, 3):
        raise ValueError("overlay image dimensions do not match hand-tracking result")
    overlay = np.ascontiguousarray(image_bgr.copy())
    for hand in result.hands:
        color = _LEFT_COLOR if hand.handedness is Handedness.LEFT else _RIGHT_COLOR
        points = np.rint(hand.keypoints_2d_px).astype(np.int32)
        for first, second in _BONES:
            cv2.line(
                overlay,
                tuple(points[first]),
                tuple(points[second]),
                color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        for point in points[:20]:
            cv2.circle(
                overlay,
                tuple(point),
                radius=3,
                color=color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
        x1, y1, x2, y2 = (int(round(value)) for value in hand.bbox_xyxy_px)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)
        label = f"{hand.handedness.value.upper()} {hand.confidence:.2f}"
        if hand.is_grasping:
            label += " GRASP"
        cv2.putText(
            overlay,
            label,
            (max(8, x1), max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    return overlay
