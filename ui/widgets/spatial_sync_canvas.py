from __future__ import annotations

import math
import time
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QLineF, QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from ingest_gateway.imu_preview import ImuPoseSnapshot

HAND_CONNECTIONS = (
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
    (6, 8),
    (8, 11),
    (11, 14),
    (14, 17),
)


@dataclass(frozen=True, slots=True)
class SpatialSyncCanvasStatus:
    has_imu_pose: bool
    has_left_hand: bool
    has_right_hand: bool
    latest_frame_index: int | None


@dataclass(frozen=True, slots=True)
class _HandPose:
    side: str
    points: tuple[tuple[float, float, float], ...]


class SpatialSyncCanvas(QWidget):
    """Paint a lightweight spatial sync view from IMU pose and hand 3D points."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("spatialSyncCanvas")
        self.setMinimumSize(320, 230)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._pose: ImuPoseSnapshot | None = None
        self._hand_result: dict[str, object] | None = None
        self._hands: tuple[_HandPose, ...] = ()
        self._latest_frame_index: int | None = None
        self._latest_paint_ms: float | None = None

    def sizeHint(self) -> QSize:
        return QSize(340, 250)

    def set_pose(self, pose: ImuPoseSnapshot | None) -> None:
        self._pose = pose
        self.update()

    def set_hand_result(self, result: dict[str, object] | None) -> None:
        self._hand_result = result
        self._hands = _hands_from_result(result)
        frame_index = result.get("frame_index") if isinstance(result, dict) else None
        self._latest_frame_index = frame_index if _is_real_int(frame_index) else None
        self.update()

    def status(self) -> SpatialSyncCanvasStatus:
        sides = {hand.side for hand in self._hands}
        return SpatialSyncCanvasStatus(
            has_imu_pose=self._pose is not None and self._pose.samples_received > 0,
            has_left_hand="left" in sides,
            has_right_hand="right" in sides,
            latest_frame_index=self._latest_frame_index,
        )

    def paintEvent(self, _event: object) -> None:
        started_ns = time.perf_counter_ns()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        outer = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(outer, 7, 7)
        painter.setClipPath(path)

        background = QLinearGradient(outer.topLeft(), outer.bottomRight())
        background.setColorAt(0.0, QColor("#070914"))
        background.setColorAt(0.58, QColor("#0b1120"))
        background.setColorAt(1.0, QColor("#101725"))
        painter.fillPath(path, background)
        self._paint_grid(painter, outer)
        self._paint_axis(painter, outer)
        self._paint_glasses(painter, outer)
        self._paint_hands(painter, outer)
        self._paint_empty_hint(painter, outer)

        painter.setClipping(False)
        painter.setPen(QPen(QColor("#273247"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(outer, 7, 7)
        painter.end()
        self._latest_paint_ms = (time.perf_counter_ns() - started_ns) / 1_000_000

    def _paint_grid(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(QColor(55, 69, 96, 120), 1))
        for value in np.linspace(-3.0, 3.0, 13):
            first = _project((value, 0.0, -2.0), rect)
            second = _project((value, 0.0, 4.0), rect)
            painter.drawLine(QLineF(first, second))
            first = _project((-3.0, 0.0, value), rect)
            second = _project((3.0, 0.0, value), rect)
            painter.drawLine(QLineF(first, second))

    def _paint_axis(self, painter: QPainter, rect: QRectF) -> None:
        origin = _project((0.0, 0.0, 0.0), rect)
        axes = (
            ((1.0, 0.0, 0.0), QColor("#ef4444")),
            ((0.0, 0.8, 0.0), QColor("#22c55e")),
            ((0.0, 0.0, 1.0), QColor("#38bdf8")),
        )
        for point, color in axes:
            painter.setPen(QPen(color, 2.2))
            painter.drawLine(QLineF(origin, _project(point, rect)))

    def _paint_glasses(self, painter: QPainter, rect: QRectF) -> None:
        pose = self._pose
        if pose is None or pose.samples_received == 0:
            return
        rotation = _quaternion_matrix(pose.quaternion_wxyz)
        model = np.asarray(
            [
                [-0.55, 0.38, 0.0],
                [0.55, 0.38, 0.0],
                [0.55, -0.28, 0.0],
                [-0.55, -0.28, 0.0],
                [-0.82, 0.28, -0.9],
                [0.82, 0.28, -0.9],
            ],
            dtype=np.float64,
        )
        transformed = (model @ rotation.T) + np.asarray([0.0, 0.78, 1.05])
        projected = [_project(point, rect) for point in transformed]

        face = QPainterPath()
        face.moveTo(projected[0])
        for point in projected[1:4]:
            face.lineTo(point)
        face.closeSubpath()
        painter.setPen(QPen(QColor("#f59e0b"), 2.0))
        painter.setBrush(QColor(245, 158, 11, 82))
        painter.drawPath(face)
        painter.setPen(QPen(QColor("#94a3b8"), 1.5))
        painter.drawLine(QLineF(projected[0], projected[4]))
        painter.drawLine(QLineF(projected[1], projected[5]))

    def _paint_hands(self, painter: QPainter, rect: QRectF) -> None:
        for hand in self._hands:
            color = QColor("#a855f7") if hand.side == "left" else QColor("#facc15")
            points = [_project(_hand_to_scene(point, hand.side), rect) for point in hand.points]
            painter.setPen(QPen(color, 2.0))
            for first, second in HAND_CONNECTIONS:
                painter.drawLine(QLineF(points[first], points[second]))
            painter.setBrush(color)
            painter.setPen(QPen(QColor("#f8fafc"), 0.8))
            for point in points:
                painter.drawEllipse(point, 2.6, 2.6)

    def _paint_empty_hint(self, painter: QPainter, rect: QRectF) -> None:
        status = self.status()
        if status.has_imu_pose or status.has_left_hand or status.has_right_hand:
            return
        painter.setPen(QColor("#64748b"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "等待 IMU 与手部位姿")


def _project(point: Iterable[float], rect: QRectF) -> QPointF:
    x, y, z = (float(value) for value in point)
    depth = max(1.0, z + 5.0)
    scale = min(rect.width(), rect.height()) * 0.82 / depth
    screen_x = rect.center().x() + x * scale + z * rect.width() * 0.03
    screen_y = rect.bottom() - rect.height() * 0.18 - y * scale - z * rect.height() * 0.12
    return QPointF(screen_x, screen_y)


def _hand_to_scene(point: tuple[float, float, float], side: str) -> tuple[float, float, float]:
    x, y, z = point
    side_offset = -0.55 if side == "left" else 0.55
    return x * 1.7 + side_offset, -y * 1.7 + 0.45, z * 1.2 + 1.35


def _hands_from_result(result: dict[str, object] | None) -> tuple[_HandPose, ...]:
    if not isinstance(result, dict):
        return ()
    raw_hands = result.get("hands")
    if not isinstance(raw_hands, list):
        return ()
    hands: list[_HandPose] = []
    for raw in raw_hands:
        if not isinstance(raw, dict):
            continue
        side = raw.get("handedness")
        points = _points_3d(raw.get("keypoints_3d_camera_m"))
        if side in {"left", "right"} and len(points) == 21:
            hands.append(_HandPose(str(side), tuple(points)))
    return tuple(hands)


def _points_3d(value: object) -> list[tuple[float, float, float]]:
    if not isinstance(value, list):
        return []
    points: list[tuple[float, float, float]] = []
    for point in value:
        if not isinstance(point, list) or len(point) != 3:
            return []
        if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in point):
            return []
        if not all(math.isfinite(float(item)) for item in point):
            return []
        points.append((float(point[0]), float(point[1]), float(point[2])))
    return points


def _quaternion_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _is_real_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
