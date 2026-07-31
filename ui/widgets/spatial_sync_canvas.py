from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget
from pyqtgraph.opengl import GLLinePlotItem, GLScatterPlotItem, GLViewWidget

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

_EMPTY_POINTS = np.empty((0, 3), dtype=np.float32)
_LEFT_COLOR = (0.66, 0.33, 0.96, 1.0)
_RIGHT_COLOR = (0.98, 0.78, 0.16, 1.0)
_GRID_COLOR = (0.23, 0.30, 0.43, 0.42)
_GLASSES_COLOR = (0.96, 0.45, 0.18, 1.0)
_GLASSES_BODY_COLOR = (0.96, 0.45, 0.18, 0.45)


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
    """OpenGL spatial view for synchronized glasses pose and hand 3D keypoints."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("spatialSyncCanvas")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumSize(340, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(
            """
            #spatialSyncCanvas {
                background: #050713;
                border: 1px solid #182235;
                border-radius: 10px;
            }
            QLabel#spatialSyncTitle {
                color: #f8fafc;
                font-size: 13px;
                font-weight: 600;
                letter-spacing: 0px;
            }
            QLabel#spatialSyncState {
                color: #8ea0b7;
                font-size: 11px;
            }
            """
        )

        self._pose: ImuPoseSnapshot | None = None
        self._hands: tuple[_HandPose, ...] = ()
        self._latest_frame_index: int | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 9, 12, 12)
        root.setSpacing(7)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        title = QLabel("空间同步", self)
        title.setObjectName("spatialSyncTitle")
        self._state_label = QLabel("等待 IMU 与手部位姿", self)
        self._state_label.setObjectName("spatialSyncState")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self._state_label)
        root.addLayout(header)

        self.view = GLViewWidget(self)
        self.view.setObjectName("spatialSyncViewport")
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.view.setBackgroundColor("#050713")
        self.view.opts["fov"] = 43
        self.view.setCameraPosition(distance=4.4, elevation=24, azimuth=-45)
        root.addWidget(self.view, 1)

        self._grid_item = GLLinePlotItem(
            pos=_floor_grid(),
            color=_GRID_COLOR,
            width=1,
            mode="lines",
            antialias=True,
        )
        self._axis_item = GLLinePlotItem(
            pos=_axis_lines(),
            color=np.asarray(
                [
                    (0.95, 0.18, 0.21, 0.95),
                    (0.95, 0.18, 0.21, 0.95),
                    (0.22, 0.82, 0.48, 0.95),
                    (0.22, 0.82, 0.48, 0.95),
                    (0.25, 0.65, 1.00, 0.95),
                    (0.25, 0.65, 1.00, 0.95),
                ],
                dtype=np.float32,
            ),
            width=2,
            mode="lines",
            antialias=True,
        )
        self._glasses_item = GLLinePlotItem(
            pos=_EMPTY_POINTS,
            color=_GLASSES_COLOR,
            width=3,
            mode="lines",
            antialias=True,
        )
        self._glasses_points = GLScatterPlotItem(
            pos=_EMPTY_POINTS,
            color=_GLASSES_BODY_COLOR,
            size=8,
            pxMode=True,
        )
        self._left_lines = GLLinePlotItem(
            pos=_EMPTY_POINTS,
            color=_LEFT_COLOR,
            width=3,
            mode="lines",
            antialias=True,
        )
        self._left_points = GLScatterPlotItem(
            pos=_EMPTY_POINTS,
            color=_LEFT_COLOR,
            size=5,
            pxMode=True,
        )
        self._right_lines = GLLinePlotItem(
            pos=_EMPTY_POINTS,
            color=_RIGHT_COLOR,
            width=3,
            mode="lines",
            antialias=True,
        )
        self._right_points = GLScatterPlotItem(
            pos=_EMPTY_POINTS,
            color=_RIGHT_COLOR,
            size=5,
            pxMode=True,
        )
        for item in (
            self._grid_item,
            self._axis_item,
            self._glasses_item,
            self._glasses_points,
            self._left_lines,
            self._left_points,
            self._right_lines,
            self._right_points,
        ):
            self.view.addItem(item)

    def set_pose(self, pose: ImuPoseSnapshot | None) -> None:
        self._pose = pose
        self._update_glasses_item()
        self._update_state_label()

    def set_hand_result(self, result: dict[str, object] | None) -> None:
        self._hands = _hands_from_result(result)
        frame_index = result.get("frame_index") if isinstance(result, dict) else None
        self._latest_frame_index = frame_index if _is_real_int(frame_index) else None
        self._update_hand_items()
        self._update_state_label()

    def status(self) -> SpatialSyncCanvasStatus:
        sides = {hand.side for hand in self._hands}
        return SpatialSyncCanvasStatus(
            has_imu_pose=self._pose is not None and self._pose.samples_received > 0,
            has_left_hand="left" in sides,
            has_right_hand="right" in sides,
            latest_frame_index=self._latest_frame_index,
        )

    def _update_glasses_item(self) -> None:
        pose = self._pose
        if pose is None or pose.samples_received == 0:
            self._glasses_item.setData(pos=_EMPTY_POINTS)
            self._glasses_points.setData(pos=_EMPTY_POINTS)
            return
        points = _glasses_model_points(pose.quaternion_wxyz)
        self._glasses_item.setData(pos=_line_segments(points, _GLASSES_CONNECTIONS))
        self._glasses_points.setData(pos=points)

    def _update_hand_items(self) -> None:
        left = next((hand for hand in self._hands if hand.side == "left"), None)
        right = next((hand for hand in self._hands if hand.side == "right"), None)
        self._set_hand_items(left, self._left_lines, self._left_points)
        self._set_hand_items(right, self._right_lines, self._right_points)

    def _set_hand_items(
        self,
        hand: _HandPose | None,
        lines: GLLinePlotItem,
        points: GLScatterPlotItem,
    ) -> None:
        if hand is None:
            lines.setData(pos=_EMPTY_POINTS)
            points.setData(pos=_EMPTY_POINTS)
            return
        scene_points = np.asarray(
            [_hand_to_scene(point, hand.side) for point in hand.points],
            dtype=np.float32,
        )
        lines.setData(pos=_line_segments(scene_points, HAND_CONNECTIONS))
        points.setData(pos=scene_points)

    def _update_state_label(self) -> None:
        status = self.status()
        parts: list[str] = []
        if status.has_imu_pose:
            parts.append("IMU")
        if status.has_left_hand:
            parts.append("左手")
        if status.has_right_hand:
            parts.append("右手")
        value = " · ".join(parts) if parts else "等待 IMU 与手部位姿"
        if status.latest_frame_index is not None:
            value = f"{value} · F{status.latest_frame_index}"
        self._state_label.setText(value)


_GLASSES_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (0, 4),
    (1, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)


def _floor_grid() -> np.ndarray:
    lines: list[tuple[float, float, float]] = []
    for value in np.linspace(-1.8, 1.8, 13):
        lines.extend(((value, -0.7, 0.0), (value, 2.8, 0.0)))
        lines.extend(((-1.8, value + 1.1, 0.0), (1.8, value + 1.1, 0.0)))
    return np.asarray(lines, dtype=np.float32)


def _axis_lines() -> np.ndarray:
    return np.asarray(
        [
            (0.0, 0.0, 0.02),
            (0.65, 0.0, 0.02),
            (0.0, 0.0, 0.02),
            (0.0, 0.65, 0.02),
            (0.0, 0.0, 0.02),
            (0.0, 0.0, 0.65),
        ],
        dtype=np.float32,
    )


def _line_segments(
    points: np.ndarray,
    connections: Iterable[tuple[int, int]],
) -> np.ndarray:
    segments: list[np.ndarray] = []
    for first, second in connections:
        segments.extend((points[first], points[second]))
    return np.asarray(segments, dtype=np.float32)


def _glasses_model_points(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    rotation = _quaternion_matrix(quaternion)
    model = np.asarray(
        [
            (-0.42, -0.03, 0.25),
            (0.42, -0.03, 0.25),
            (0.42, -0.03, -0.18),
            (-0.42, -0.03, -0.18),
            (-0.64, 0.34, 0.16),
            (0.64, 0.34, 0.16),
            (-0.72, 0.86, 0.05),
            (0.72, 0.86, 0.05),
        ],
        dtype=np.float64,
    )
    transformed = (model @ rotation.T) + np.asarray([0.0, 0.65, 0.92])
    return transformed.astype(np.float32)


def _hand_to_scene(point: tuple[float, float, float], side: str) -> tuple[float, float, float]:
    x, y, z = point
    side_offset = -0.50 if side == "left" else 0.50
    return x * 1.8 + side_offset, z * 1.25 + 0.9, -y * 1.8 + 0.55


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
