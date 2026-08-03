from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget
from pyqtgraph.opengl import (
    GLLinePlotItem,
    GLMeshItem,
    GLScatterPlotItem,
    GLTextItem,
    GLViewWidget,
)
from qfluentwidgets import CaptionLabel, FluentIcon, StrongBodyLabel, TransparentToolButton

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
_IDENTITY_QUATERNION = (1.0, 0.0, 0.0, 0.0)
_SCENE_ORIGIN = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)

# Camera coordinates are x-right, y-down, z-forward. The OpenGL scene uses
# x-right, y-forward, z-up, so this matrix changes only the axis convention.
_CAMERA_TO_SCENE = np.asarray(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, -1.0, 0.0),
    ),
    dtype=np.float64,
)

_LEFT_COLOR = (0.66, 0.33, 0.96, 1.0)
_RIGHT_COLOR = (0.98, 0.78, 0.16, 1.0)
_GRID_COLOR = (0.23, 0.30, 0.43, 0.42)
_FRAME_COLOR = (0.95, 0.48, 0.18, 1.0)
_LENS_COLOR = (0.18, 0.62, 0.92, 0.18)
_CAMERA_BODY_COLOR = (0.20, 0.24, 0.33, 1.0)
_CAMERA_LENS_COLOR = (0.30, 0.85, 1.0, 0.95)
_AXIS_COLORS = np.asarray(
    (
        (0.95, 0.18, 0.21, 0.98),
        (0.95, 0.18, 0.21, 0.98),
        (0.22, 0.82, 0.48, 0.98),
        (0.22, 0.82, 0.48, 0.98),
        (0.25, 0.65, 1.00, 0.98),
        (0.25, 0.65, 1.00, 0.98),
    ),
    dtype=np.float32,
)


@dataclass(frozen=True, slots=True)
class SpatialSyncCanvasStatus:
    has_imu_pose: bool
    has_left_hand: bool
    has_right_hand: bool
    latest_frame_index: int | None


class SpatialPoseSnapshot(Protocol):
    quaternion_wxyz: tuple[float, float, float, float]
    samples_received: int


@dataclass(frozen=True, slots=True)
class _HandPose:
    side: str
    points: tuple[tuple[float, float, float], ...]


class SpatialSyncCanvas(QWidget):
    """OpenGL camera-frame view of the glasses and tracked hands."""

    reset_pose_requested = pyqtSignal()

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
            StrongBodyLabel#spatialSyncTitle {
                color: #f8fafc;
                font-size: 13px;
                font-weight: 600;
                letter-spacing: 0px;
            }
            CaptionLabel#spatialSyncState {
                color: #8ea0b7;
                font-size: 11px;
            }
            TransparentToolButton#imuPoseResetButton {
                background: rgba(255, 255, 255, 0.14);
                border: 1px solid rgba(255, 255, 255, 0.28);
                border-radius: 14px;
            }
            TransparentToolButton#imuPoseResetButton:hover {
                background: rgba(255, 255, 255, 0.22);
                border: 1px solid rgba(255, 255, 255, 0.42);
            }
            TransparentToolButton#imuPoseResetButton:pressed {
                background: rgba(255, 255, 255, 0.30);
            }
            """
        )

        self._pose: SpatialPoseSnapshot | None = None
        self._hands: tuple[_HandPose, ...] = ()
        self._latest_frame_index: int | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 9, 12, 12)
        root.setSpacing(7)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        title = StrongBodyLabel("空间同步", self)
        title.setObjectName("spatialSyncTitle")
        self._state_label = CaptionLabel("相机XYZ坐标系 | 等待 IMU 与手部位姿", self)
        self._state_label.setObjectName("spatialSyncState")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self._state_label)
        self.reset_pose_button = TransparentToolButton()
        self.reset_pose_button.setObjectName("imuPoseResetButton")
        self.reset_pose_button.setIcon(FluentIcon.SYNC.icon(color=QColor("#f8fafc")))
        self.reset_pose_button.setIconSize(QSize(15, 15))
        self.reset_pose_button.setToolTip("刷新眼镜姿态")
        self.reset_pose_button.setFixedSize(28, 28)
        self.reset_pose_button.clicked.connect(self.reset_pose_requested.emit)
        header.addWidget(self.reset_pose_button)
        root.addLayout(header)

        self.view = GLViewWidget(self)
        self.view.setObjectName("spatialSyncViewport")
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.view.setBackgroundColor("#050713")
        self.view.opts["fov"] = 43
        self.view.setCameraPosition(distance=1.65, elevation=20, azimuth=-48)
        root.addWidget(self.view, 1)

        self._grid_item = GLLinePlotItem(
            pos=_floor_grid(),
            color=_GRID_COLOR,
            width=1,
            mode="lines",
            antialias=True,
        )
        self._axis_item = GLLinePlotItem(
            pos=_EMPTY_POINTS,
            color=_AXIS_COLORS,
            width=2,
            mode="lines",
            antialias=True,
        )
        axis_font = QFont("Segoe UI", 8)
        self._axis_labels = tuple(
            GLTextItem(text=label, color=color, font=axis_font, pos=(0.0, 0.0, 0.0))
            for label, color in (
                ("X", (255, 77, 77, 255)),
                ("Y", (77, 255, 128, 255)),
                ("Z", (77, 179, 255, 255)),
            )
        )
        self._camera_frustum_item = GLLinePlotItem(
            pos=_EMPTY_POINTS,
            color=(0.35, 0.72, 1.0, 0.50),
            width=1,
            mode="lines",
            antialias=True,
        )
        self._camera_origin_item = GLScatterPlotItem(
            pos=_EMPTY_POINTS,
            color=_CAMERA_LENS_COLOR,
            size=7,
            pxMode=True,
        )

        frame_vertices, frame_faces = _glasses_frame_mesh()
        lens_vertices, lens_faces = _glasses_lens_mesh()
        camera_vertices, camera_faces = _camera_housing_mesh()
        self._frame_vertices = frame_vertices
        self._frame_faces = frame_faces
        self._lens_vertices = lens_vertices
        self._lens_faces = lens_faces
        self._camera_vertices = camera_vertices
        self._camera_faces = camera_faces
        self._glasses_frame_item = GLMeshItem(
            vertexes=frame_vertices,
            faces=frame_faces,
            color=_FRAME_COLOR,
            smooth=False,
            drawEdges=False,
            shader="shaded",
        )
        self._glasses_lens_item = GLMeshItem(
            vertexes=lens_vertices,
            faces=lens_faces,
            color=_LENS_COLOR,
            smooth=False,
            drawEdges=False,
            glOptions="translucent",
        )
        self._camera_body_item = GLMeshItem(
            vertexes=camera_vertices,
            faces=camera_faces,
            color=_CAMERA_BODY_COLOR,
            smooth=False,
            drawEdges=False,
            shader="shaded",
        )
        self._camera_lens_item = GLScatterPlotItem(
            pos=_EMPTY_POINTS,
            color=_CAMERA_LENS_COLOR,
            size=9,
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
            *self._axis_labels,
            self._camera_frustum_item,
            self._camera_origin_item,
            self._glasses_frame_item,
            self._glasses_lens_item,
            self._camera_body_item,
            self._camera_lens_item,
            self._left_lines,
            self._left_points,
            self._right_lines,
            self._right_points,
        ):
            self.view.addItem(item)
        self._update_reference_items()

    def set_pose(self, pose: SpatialPoseSnapshot | None) -> None:
        self._pose = pose
        self._update_reference_items()
        self._update_hand_items()
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

    def _update_reference_items(self) -> None:
        quaternion = _pose_quaternion(self._pose)
        axis_points = _camera_points_to_scene(_camera_axis_points(), quaternion)
        self._axis_item.setData(pos=_line_segments(axis_points, ((0, 1), (0, 2), (0, 3))))
        for label, position in zip(self._axis_labels, axis_points[1:], strict=True):
            label.setData(pos=tuple(float(value) for value in position))
        frustum = _camera_points_to_scene(_camera_frustum_points(), quaternion)
        self._camera_frustum_item.setData(pos=_line_segments(frustum, _FRUSTUM_CONNECTIONS))
        self._camera_origin_item.setData(pos=axis_points[:1])

        self._glasses_frame_item.setMeshData(
            vertexes=_camera_points_to_scene(self._frame_vertices, quaternion),
            faces=self._frame_faces,
        )
        self._glasses_lens_item.setMeshData(
            vertexes=_camera_points_to_scene(self._lens_vertices, quaternion),
            faces=self._lens_faces,
        )
        self._camera_body_item.setMeshData(
            vertexes=_camera_points_to_scene(self._camera_vertices, quaternion),
            faces=self._camera_faces,
        )
        camera_lens = _camera_points_to_scene(
            np.asarray(((0.0, 0.0, 0.018),), dtype=np.float64), quaternion
        )
        self._camera_lens_item.setData(pos=camera_lens)

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
        camera_points = np.asarray(hand.points, dtype=np.float64)
        scene_points = _camera_points_to_scene(camera_points, _pose_quaternion(self._pose))
        lines.setData(pos=_line_segments(scene_points, HAND_CONNECTIONS))
        points.setData(pos=scene_points)

    def _update_state_label(self) -> None:
        status = self.status()
        parts = ["相机XYZ"]
        if status.has_imu_pose:
            parts.append("IMU")
        if status.has_left_hand:
            parts.append("左手")
        if status.has_right_hand:
            parts.append("右手")
        value = " | ".join(parts)
        if status.latest_frame_index is not None:
            value = f"{value} | F{status.latest_frame_index}"
        self._state_label.setText(value)


_FRUSTUM_CONNECTIONS = (
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 1),
)


def _floor_grid() -> np.ndarray:
    lines: list[tuple[float, float, float]] = []
    grid_count = 9
    for x in np.linspace(-0.8, 0.8, grid_count):
        lines.extend(((float(x), -0.25, -0.42), (float(x), 1.25, -0.42)))
    for y in np.linspace(-0.25, 1.25, grid_count):
        lines.extend(((-0.8, float(y), -0.42), (0.8, float(y), -0.42)))
    return np.asarray(lines, dtype=np.float32)


def _camera_axis_points() -> np.ndarray:
    return np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.22, 0.0, 0.0),
            (0.0, 0.22, 0.0),
            (0.0, 0.0, 0.22),
        ),
        dtype=np.float64,
    )


def _camera_frustum_points() -> np.ndarray:
    return np.asarray(
        (
            (0.0, 0.0, 0.0),
            (-0.13, -0.09, 0.22),
            (0.13, -0.09, 0.22),
            (0.13, 0.09, 0.22),
            (-0.13, 0.09, 0.22),
        ),
        dtype=np.float64,
    )


def _camera_points_to_scene(
    points: np.ndarray,
    quaternion: tuple[float, float, float, float],
) -> np.ndarray:
    """Map camera-frame meters into the OpenGL scene using the IMU rotation."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("camera points must have shape (N, 3)")
    camera_to_scene_rotation = _CAMERA_TO_SCENE @ _quaternion_matrix(quaternion)
    return (values @ camera_to_scene_rotation.T + _SCENE_ORIGIN).astype(np.float32)


def _glasses_frame_mesh() -> tuple[np.ndarray, np.ndarray]:
    polylines: list[np.ndarray] = []
    for center_x in (-0.038, 0.038):
        polylines.append(_ellipse_polyline(center_x, 0.004, -0.024, 0.031, 0.020))
    polylines.extend(
        (
            np.asarray(((-0.009, -0.002, -0.024), (0.0, -0.009, -0.021), (0.009, -0.002, -0.024))),
            np.asarray(((-0.069, 0.004, -0.024), (-0.074, 0.005, -0.070), (-0.070, 0.006, -0.145))),
            np.asarray(((0.069, 0.004, -0.024), (0.074, 0.005, -0.070), (0.070, 0.006, -0.145))),
            np.asarray(((-0.067, -0.012, -0.023), (-0.009, -0.012, -0.023))),
            np.asarray(((0.067, -0.012, -0.023), (0.009, -0.012, -0.023))),
        )
    )
    return _tube_mesh(polylines, radius=0.0035, sides=6)


def _glasses_lens_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for center_x in (-0.038, 0.038):
        start = len(vertices)
        vertices.append((center_x, 0.004, -0.021))
        segments = 24
        angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
        vertices.extend(
            (center_x + 0.026 * math.cos(angle), 0.004 + 0.016 * math.sin(angle), -0.021)
            for angle in angles
        )
        for index in range(segments):
            faces.append((start, start + 1 + index, start + 1 + ((index + 1) % segments)))
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _ellipse_polyline(
    center_x: float,
    center_y: float,
    z: float,
    radius_x: float,
    radius_y: float,
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, 33)
    return np.column_stack(
        (
            center_x + radius_x * np.cos(angles),
            center_y + radius_y * np.sin(angles),
            np.full_like(angles, z),
        )
    )


def _camera_housing_mesh() -> tuple[np.ndarray, np.ndarray]:
    # A compact bridge-mounted camera body with the optical origin at (0, 0, 0).
    center = np.asarray((0.0, -0.014, 0.005), dtype=np.float64)
    half = np.asarray((0.012, 0.009, 0.014), dtype=np.float64)
    corners = np.asarray(
        [
            center + (-half[0], -half[1], -half[2]),
            center + (half[0], -half[1], -half[2]),
            center + (half[0], half[1], -half[2]),
            center + (-half[0], half[1], -half[2]),
            center + (-half[0], -half[1], half[2]),
            center + (half[0], -half[1], half[2]),
            center + (half[0], half[1], half[2]),
            center + (-half[0], half[1], half[2]),
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        (
            (0, 1, 2),
            (0, 2, 3),
            (4, 6, 5),
            (4, 7, 6),
            (0, 4, 5),
            (0, 5, 1),
            (1, 5, 6),
            (1, 6, 2),
            (2, 6, 7),
            (2, 7, 3),
            (3, 7, 4),
            (3, 4, 0),
        ),
        dtype=np.int32,
    )
    return corners, faces


def _tube_mesh(
    polylines: Iterable[np.ndarray],
    *,
    radius: float,
    sides: int,
) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    for polyline in polylines:
        points = np.asarray(polyline, dtype=np.float64)
        for first, second in zip(points[:-1], points[1:], strict=True):
            direction = second - first
            length = float(np.linalg.norm(direction))
            if length < 1e-8:
                continue
            direction /= length
            reference = np.asarray((0.0, 1.0, 0.0))
            if abs(float(np.dot(direction, reference))) > 0.9:
                reference = np.asarray((1.0, 0.0, 0.0))
            basis_x = np.cross(direction, reference)
            basis_x /= np.linalg.norm(basis_x)
            basis_y = np.cross(direction, basis_x)
            start = len(vertices)
            for point in (first, second):
                for angle in np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False):
                    vertices.append(
                        point
                        + radius * (math.cos(angle) * basis_x + math.sin(angle) * basis_y)
                    )
            for index in range(sides):
                next_index = (index + 1) % sides
                a = start + index
                b = start + next_index
                c = start + sides + next_index
                d = start + sides + index
                faces.extend(((a, b, c), (a, c, d)))
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _line_segments(
    points: np.ndarray,
    connections: Iterable[tuple[int, int]],
) -> np.ndarray:
    segments: list[np.ndarray] = []
    for first, second in connections:
        segments.extend((points[first], points[second]))
    return np.asarray(segments, dtype=np.float32)


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


def _pose_quaternion(
    pose: SpatialPoseSnapshot | None,
) -> tuple[float, float, float, float]:
    if pose is None or pose.samples_received == 0:
        return _IDENTITY_QUATERNION
    values = tuple(float(value) for value in pose.quaternion_wxyz)
    norm = math.sqrt(sum(value * value for value in values))
    if len(values) != 4 or not math.isfinite(norm) or norm < 1e-8:
        return _IDENTITY_QUATERNION
    return tuple(value / norm for value in values)


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
