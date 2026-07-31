from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QLineF, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from ingest_gateway.imu_preview import ImuPoseSnapshot


class ImuPoseCanvas(QWidget):
    """Paint the lightweight glasses pose preview from the latest IMU quaternion."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pose: ImuPoseSnapshot | None = None
        self.setMinimumHeight(128)

    def sizeHint(self) -> QSize:
        return QSize(320, 148)

    def set_pose(self, pose: ImuPoseSnapshot | None) -> None:
        self._pose = pose
        self.update()

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#d7dde7"), 1))
        painter.setBrush(QColor("#f7f9fc"))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 5, 5)
        if self._pose is None:
            painter.setPen(QColor("#8a94a3"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "等待 IMU")
            return

        rotation = _quaternion_matrix(self._pose.quaternion_wxyz)
        points = _wireframe_points() @ rotation.T
        projected = _project(points, self.width(), self.height())
        painter.setPen(QPen(QColor("#2f6fed"), 2.2))
        for first, second in _wireframe_edges():
            painter.drawLine(QLineF(*projected[first], *projected[second]))

        origin = np.array([[0.0, 0.0, 0.0]])
        axes = np.array([[0.9, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 0.9]])
        axis_points = _project(np.vstack((origin, axes)) @ rotation.T, self.width(), self.height())
        for index, color in enumerate(("#dc4c3e", "#16a085", "#2f6fed"), start=1):
            painter.setPen(QPen(QColor(color), 2.6))
            painter.drawLine(QLineF(*axis_points[0], *axis_points[index]))


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


def _wireframe_points() -> np.ndarray:
    return np.array(
        [
            [-1.45, -0.45, 0.0],
            [-0.15, -0.45, 0.0],
            [-0.15, 0.45, 0.0],
            [-1.45, 0.45, 0.0],
            [0.15, -0.45, 0.0],
            [1.45, -0.45, 0.0],
            [1.45, 0.45, 0.0],
            [0.15, 0.45, 0.0],
            [-1.45, 0.30, 0.0],
            [-1.75, 0.20, -1.9],
            [1.45, 0.30, 0.0],
            [1.75, 0.20, -1.9],
        ],
        dtype=np.float64,
    )


def _wireframe_edges() -> tuple[tuple[int, int], ...]:
    return (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (1, 4),
        (2, 7),
        (8, 9),
        (10, 11),
    )


def _project(points: np.ndarray, width: int, height: int) -> list[tuple[float, float]]:
    depth = np.maximum(2.0, points[:, 2] + 5.5)
    scale = min(width, height) * 1.9
    x_values = width / 2 + points[:, 0] * scale / depth
    y_values = height / 2 - points[:, 1] * scale / depth
    return list(zip(x_values.tolist(), y_values.tolist(), strict=True))
