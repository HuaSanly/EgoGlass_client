from __future__ import annotations

import dearpygui.dearpygui as dpg
import numpy as np

from ingest_gateway.imu_preview import ImuPoseSnapshot


class ImuPoseWidget:
    """Project a lightweight glasses wireframe from the latest preview quaternion."""

    def __init__(self, *, parent: int | str, width: int = 320, height: int = 150) -> None:
        self.width = width
        self.height = height
        self._drawlist = "imu-pose-drawlist"
        self._layer = "imu-pose-layer"
        with dpg.drawlist(width=width, height=height, parent=parent, tag=self._drawlist):
            dpg.draw_rectangle(
                (0, 0),
                (width, height),
                color=(48, 56, 59, 255),
                fill=(12, 15, 16, 255),
            )
            dpg.add_draw_layer(tag=self._layer)

    def update(self, pose: ImuPoseSnapshot | None) -> None:
        dpg.delete_item(self._layer, children_only=True)
        if pose is None:
            return
        rotation = _quaternion_matrix(pose.quaternion_wxyz)
        points = _wireframe_points() @ rotation.T
        projected = _project(points, self.width, self.height)
        for first, second in _wireframe_edges():
            dpg.draw_line(
                tuple(projected[first]),
                tuple(projected[second]),
                color=(93, 199, 164, 255),
                thickness=2.0,
                parent=self._layer,
            )
        origin = np.array([[0.0, 0.0, 0.0]])
        axes = np.array([[0.9, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 0.9]])
        axis_points = _project(np.vstack((origin, axes)) @ rotation.T, self.width, self.height)
        for index, color in enumerate(
            ((229, 89, 79, 255), (93, 199, 164, 255), (89, 151, 227, 255)),
            start=1,
        ):
            dpg.draw_line(
                tuple(axis_points[0]),
                tuple(axis_points[index]),
                color=color,
                thickness=2.5,
                parent=self._layer,
            )


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


def _project(points: np.ndarray, width: int, height: int) -> np.ndarray:
    depth = np.maximum(2.0, points[:, 2] + 5.5)
    scale = min(width, height) * 1.9
    result = np.empty((len(points), 2), dtype=np.float64)
    result[:, 0] = width / 2 + points[:, 0] * scale / depth
    result[:, 1] = height / 2 - points[:, 1] * scale / depth
    return result
