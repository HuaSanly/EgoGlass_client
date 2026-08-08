"""OpenGL spatial synchronization view with explicit camera/world references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QVector3D
from PyQt6.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget
from pyqtgraph.opengl import (
    GLLinePlotItem,
    GLScatterPlotItem,
    GLTextItem,
    GLViewWidget,
)
from qfluentwidgets import (
    CaptionLabel,
    FluentIcon,
    SegmentedWidget,
    StrongBodyLabel,
    TransparentToolButton,
)

from schemas import VioPose
from ui.presentation.spatial_scene import (
    SpatialReferenceFrame,
    SpatialSceneState,
    build_spatial_scene_state,
)

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
_CAMERA_TO_SCENE = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    dtype=np.float64,
)
_LEFT_COLOR = (0.12, 0.82, 0.91, 1.0)
_RIGHT_COLOR = (1.0, 0.40, 0.34, 1.0)
_OBJECT_COLOR = (1.0, 0.76, 0.20, 0.92)
_GRID_COLOR = (0.24, 0.32, 0.43, 0.42)
_AXIS_COLORS = np.asarray(
    (
        (0.95, 0.18, 0.21, 0.98),
        (0.95, 0.18, 0.21, 0.98),
        (0.22, 0.82, 0.48, 0.98),
        (0.22, 0.82, 0.48, 0.98),
        (0.25, 0.65, 1.0, 0.98),
        (0.25, 0.65, 1.0, 0.98),
    ),
    dtype=np.float32,
)


@dataclass(frozen=True, slots=True)
class SpatialViewPreset:
    center_xyz: tuple[float, float, float]
    distance: float
    elevation_degrees: float
    azimuth_degrees: float
    field_of_view_degrees: float


_VIEW_PRESETS = {
    SpatialReferenceFrame.WORLD: SpatialViewPreset(
        center_xyz=(0.0, 0.30, -0.10),
        distance=1.65,
        elevation_degrees=12.0,
        azimuth_degrees=-90.0,
        field_of_view_degrees=43.0,
    ),
    SpatialReferenceFrame.CAMERA: SpatialViewPreset(
        center_xyz=(0.0, 0.40, -0.10),
        distance=1.10,
        elevation_degrees=5.0,
        azimuth_degrees=-90.0,
        field_of_view_degrees=43.0,
    ),
}


@dataclass(frozen=True, slots=True)
class SpatialSyncCanvasStatus:
    has_imu_pose: bool
    has_left_hand: bool
    has_right_hand: bool
    latest_frame_index: int | None
    reference_frame: SpatialReferenceFrame = SpatialReferenceFrame.CAMERA
    head_pose_source: str = "等待数据"
    has_head_pose: bool = False
    has_vio_pose: bool = False
    # Kept as a read-only compatibility diagnostic; no trajectory is rendered.
    trajectory_pose_count: int = 0


class SpatialPoseSnapshot(Protocol):
    quaternion_wxyz: tuple[float, float, float, float]
    samples_received: int


class SpatialSyncCanvas(QWidget):
    """Draw a person's head coordinate frame and tracked hands."""

    reset_pose_requested = pyqtSignal()
    reference_frame_changed = pyqtSignal(str)

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
            StrongBodyLabel#spatialSyncTitle { color: #f8fafc; }
            CaptionLabel#spatialSyncState { color: #8ea0b7; }
            SegmentedWidget { background: rgba(255, 255, 255, 0.08); }
            TransparentToolButton#imuPoseResetButton {
                background: rgba(255, 255, 255, 0.14);
                border: 1px solid rgba(255, 255, 255, 0.28);
                border-radius: 14px;
            }
            TransparentToolButton#imuPoseResetButton:hover {
                background: rgba(255, 255, 255, 0.22);
            }
            """
        )
        self._reference_frame = SpatialReferenceFrame.CAMERA
        self._scene_state: SpatialSceneState | None = None
        self._legacy_pose: SpatialPoseSnapshot | None = None
        self._legacy_hand_result: dict[str, object] | None = None
        self._legacy_vio_pose: VioPose | None = None
        self._legacy_vio_first_pose: VioPose | None = None
        self._trajectory_pose_count = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 9, 12, 12)
        root.setSpacing(7)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        title = StrongBodyLabel("空间同步", self)
        title.setObjectName("spatialSyncTitle")
        header.addWidget(title)
        self.reference_selector = SegmentedWidget(self)
        self.reference_selector.setObjectName("spatialReferenceSelector")
        self.reference_selector.addItem(
            SpatialReferenceFrame.WORLD.value,
            "世界系",
            onClick=lambda: self.set_reference_frame(SpatialReferenceFrame.WORLD),
        )
        self.reference_selector.addItem(
            SpatialReferenceFrame.CAMERA.value,
            "相机系",
            onClick=lambda: self.set_reference_frame(SpatialReferenceFrame.CAMERA),
        )
        self.reference_selector.setCurrentItem(SpatialReferenceFrame.CAMERA.value)
        header.addWidget(self.reference_selector)
        header.addStretch(1)
        self._state_label = CaptionLabel("相机坐标系 | 等待数据", self)
        self._state_label.setObjectName("spatialSyncState")
        header.addWidget(self._state_label)
        self.reset_pose_button = TransparentToolButton(self)
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
        self._apply_view_preset(self._reference_frame)
        root.addWidget(self.view, 1)

        self._grid_item = GLLinePlotItem(
            pos=_floor_grid(), color=_GRID_COLOR, width=1, mode="lines", antialias=True
        )
        self._head_axis_item = GLLinePlotItem(
            pos=_EMPTY_POINTS, color=_AXIS_COLORS, width=2, mode="lines", antialias=True
        )
        self._camera_axis_item = GLLinePlotItem(
            pos=_EMPTY_POINTS, color=_AXIS_COLORS, width=2, mode="lines", antialias=True
        )
        self._head_origin_item = GLScatterPlotItem(
            pos=_EMPTY_POINTS, color=(0.96, 0.96, 0.98, 1.0), size=8, pxMode=True
        )
        self._left_lines = GLLinePlotItem(
            pos=_EMPTY_POINTS, color=_LEFT_COLOR, width=3, mode="lines", antialias=True
        )
        self._left_points = GLScatterPlotItem(
            pos=_EMPTY_POINTS, color=_LEFT_COLOR, size=5, pxMode=True
        )
        self._right_lines = GLLinePlotItem(
            pos=_EMPTY_POINTS, color=_RIGHT_COLOR, width=3, mode="lines", antialias=True
        )
        self._right_points = GLScatterPlotItem(
            pos=_EMPTY_POINTS, color=_RIGHT_COLOR, size=5, pxMode=True
        )
        self._object_points = GLScatterPlotItem(
            pos=_EMPTY_POINTS, color=_OBJECT_COLOR, size=6, pxMode=True
        )
        self._object_axes = GLLinePlotItem(
            pos=_EMPTY_POINTS, color=_AXIS_COLORS, width=2, mode="lines", antialias=True
        )
        axis_font = QFont("Segoe UI", 8)
        self._head_axis_labels = tuple(
            GLTextItem(text=label, color=color, font=axis_font, pos=(0.0, 0.0, 0.0))
            for label, color in (
                ("X", (255, 77, 77, 255)),
                ("Y", (77, 255, 128, 255)),
                ("Z", (77, 179, 255, 255)),
            )
        )
        self._camera_axis_labels = tuple(
            GLTextItem(text=label, color=color, font=axis_font, pos=(0.0, 0.0, 0.0))
            for label, color in (
                ("X", (255, 77, 77, 255)),
                ("Y", (77, 255, 128, 255)),
                ("Z", (77, 179, 255, 255)),
            )
        )
        for item in (
            self._grid_item,
            self._head_axis_item,
            self._camera_axis_item,
            self._head_origin_item,
            self._left_lines,
            self._left_points,
            self._right_lines,
            self._right_points,
            self._object_points,
            self._object_axes,
            *self._head_axis_labels,
            *self._camera_axis_labels,
        ):
            self.view.addItem(item)
        self._clear_geometry()

    def set_reference_frame(self, reference_frame: SpatialReferenceFrame | str) -> None:
        """Switch display coordinates without recreating the OpenGL view."""

        frame = SpatialReferenceFrame(reference_frame)
        self._reference_frame = frame
        self._apply_view_preset(frame)
        self.reference_selector.blockSignals(True)
        self.reference_selector.setCurrentItem(frame.value)
        self.reference_selector.blockSignals(False)
        self.reference_frame_changed.emit(frame.value)
        self._update_state_label()

    @property
    def reference_frame(self) -> SpatialReferenceFrame:
        return self._reference_frame

    def set_scene_state(self, state: SpatialSceneState | None) -> None:
        """Render an already transformed scene state."""

        self._scene_state = state
        if state is None:
            self._clear_geometry()
        else:
            self._set_axis_items(state)
            self._set_hand_items(state.left_hand_points_m, self._left_lines, self._left_points)
            self._set_hand_items(state.right_hand_points_m, self._right_lines, self._right_points)
            self._set_object_items(state)
            self._grid_item.setVisible(state.show_ground)
        self._update_state_label()

    # Compatibility adapters for callers that still provide individual streams.
    def set_pose(self, pose: SpatialPoseSnapshot | None) -> None:
        self._legacy_pose = pose
        self._rebuild_legacy_state()

    def set_hand_result(self, result: dict[str, object] | None) -> None:
        self._legacy_hand_result = result
        self._rebuild_legacy_state()

    def set_vio_trajectory(self, trajectory: object | None, pose: VioPose | None) -> None:
        """Compatibility shim; trajectory geometry is intentionally never drawn."""

        self._legacy_vio_pose = pose
        poses = getattr(trajectory, "poses", ())
        self._legacy_vio_first_pose = poses[0] if poses else None
        self._trajectory_pose_count = len(poses)
        self._rebuild_legacy_state()

    def status(self) -> SpatialSyncCanvasStatus:
        state = self._scene_state
        return SpatialSyncCanvasStatus(
            has_imu_pose=state.has_imu_pose if state is not None else False,
            has_left_hand=bool(state and state.left_hand_points_m),
            has_right_hand=bool(state and state.right_hand_points_m),
            latest_frame_index=state.latest_frame_index if state is not None else None,
            reference_frame=state.reference_frame if state is not None else self._reference_frame,
            head_pose_source=state.head_pose_source if state is not None else "等待数据",
            has_head_pose=bool(state and state.head_axes_m),
            has_vio_pose=state.has_vio_pose if state is not None else False,
            trajectory_pose_count=self._trajectory_pose_count,
        )

    def _rebuild_legacy_state(self) -> None:
        self.set_scene_state(
            build_spatial_scene_state(
                self._reference_frame,
                hand_result=self._legacy_hand_result,
                imu_pose=self._legacy_pose,
                vio_pose=self._legacy_vio_pose,
                vio_first_pose=self._legacy_vio_first_pose,
            )
        )

    def _set_axis_items(self, state: SpatialSceneState) -> None:
        head = _to_scene(np.asarray(state.head_axes_m, dtype=np.float64))
        camera = _to_scene(np.asarray(state.camera_axes_m, dtype=np.float64))
        self._head_axis_item.setData(pos=_line_segments(head, ((0, 1), (0, 2), (0, 3))))
        self._camera_axis_item.setData(pos=_line_segments(camera, ((0, 1), (0, 2), (0, 3))))
        self._head_origin_item.setData(pos=head[:1] if len(head) else _EMPTY_POINTS)
        head_label_positions = (
            zip(self._head_axis_labels, head[1:], strict=True) if len(head) else ()
        )
        for label, position in head_label_positions:
            label.setData(pos=tuple(float(value) for value in position))
        camera_label_positions = (
            zip(self._camera_axis_labels, camera[1:], strict=True)
            if len(camera)
            else ()
        )
        for label, position in camera_label_positions:
            label.setData(pos=tuple(float(value) for value in position))
        self._head_axis_item.setVisible(bool(state.head_axes_m))
        self._head_origin_item.setVisible(bool(state.head_axes_m))
        for label in self._head_axis_labels:
            label.setVisible(bool(state.head_axes_m))
        self._camera_axis_item.setVisible(bool(state.camera_axes_m))
        for label in self._camera_axis_labels:
            label.setVisible(bool(state.camera_axes_m))

    def _set_hand_items(
        self,
        points: tuple[tuple[float, float, float], ...],
        lines: GLLinePlotItem,
        scatter: GLScatterPlotItem,
    ) -> None:
        values = _to_scene(np.asarray(points, dtype=np.float64)) if points else _EMPTY_POINTS
        lines.setData(
            pos=(
                _line_segments(values, HAND_CONNECTIONS)
                if len(values) == 21
                else _EMPTY_POINTS
            )
        )
        scatter.setData(pos=values if len(values) == 21 else _EMPTY_POINTS)

    def _set_object_items(self, state: SpatialSceneState) -> None:
        points = (
            _to_scene(np.asarray(state.object_points_m, dtype=np.float64))
            if state.object_points_m
            else _EMPTY_POINTS
        )
        axis_segments: list[np.ndarray] = []
        for axis in state.object_axes_m:
            values = _to_scene(np.asarray(axis, dtype=np.float64))
            axis_segments.append(_line_segments(values, ((0, 1), (0, 2), (0, 3))))
        segments = np.concatenate(axis_segments) if axis_segments else _EMPTY_POINTS
        self._object_points.setData(pos=points)
        self._object_axes.setData(
            pos=segments,
            color=(
                np.tile(_AXIS_COLORS, (len(axis_segments), 1))
                if axis_segments
                else _AXIS_COLORS
            ),
        )
        visible = state.reference_frame is SpatialReferenceFrame.WORLD
        self._object_points.setVisible(visible and len(points) > 0)
        self._object_axes.setVisible(visible and len(segments) > 0)

    def _clear_geometry(self) -> None:
        for item in (
            self._head_axis_item,
            self._camera_axis_item,
            self._head_origin_item,
            self._left_lines,
            self._left_points,
            self._right_lines,
            self._right_points,
            self._object_points,
            self._object_axes,
        ):
            item.setData(pos=_EMPTY_POINTS)
        self._grid_item.setVisible(False)
        for label in (*self._head_axis_labels, *self._camera_axis_labels):
            label.setVisible(False)

    def _update_state_label(self) -> None:
        status = self.status()
        parts = ["世界系" if status.reference_frame is SpatialReferenceFrame.WORLD else "相机系"]
        if status.has_imu_pose:
            parts.append("IMU")
        if status.has_vio_pose:
            parts.append("VIO")
        if status.has_left_hand:
            parts.append("左手")
        if status.has_right_hand:
            parts.append("右手")
        if status.latest_frame_index is not None:
            parts.append(f"F{status.latest_frame_index}")
        self._state_label.setText(" | ".join(parts))

    def _apply_view_preset(self, reference_frame: SpatialReferenceFrame) -> None:
        """Reset the observer camera only when the reference frame changes."""

        preset = _VIEW_PRESETS[reference_frame]
        self.view.opts["fov"] = preset.field_of_view_degrees
        self.view.setCameraPosition(
            pos=QVector3D(
                preset.center_xyz[0],
                preset.center_xyz[1],
                preset.center_xyz[2],
            ),
            distance=preset.distance,
            elevation=preset.elevation_degrees,
            azimuth=preset.azimuth_degrees,
        )


def _floor_grid() -> np.ndarray:
    """Return a horizontal scene-space grid below the head origin.

    Hand points arrive in camera coordinates where Y points down and Z points
    forward. ``_to_scene`` maps those axes to X-right, Y-forward, Z-up. The
    grid is authored in the final scene coordinates so its Z value stays
    constant and it cannot accidentally become a vertical wall.
    """

    lines: list[tuple[float, float, float]] = []
    for x in np.linspace(-0.8, 0.8, 9):
        lines.extend(((float(x), -0.25, -0.42), (float(x), 1.25, -0.42)))
    for y in np.linspace(-0.25, 1.25, 9):
        lines.extend(((-0.8, float(y), -0.42), (0.8, float(y), -0.42)))
    return np.asarray(lines, dtype=np.float32)


def _to_scene(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.size == 0:
        return _EMPTY_POINTS
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("scene points must have shape (N, 3)")
    return (values @ _CAMERA_TO_SCENE.T).astype(np.float32)


def _line_segments(points: np.ndarray, connections: tuple[tuple[int, int], ...]) -> np.ndarray:
    if len(points) == 0:
        return _EMPTY_POINTS
    segments: list[np.ndarray] = []
    for first, second in connections:
        if first < len(points) and second < len(points):
            segments.extend((points[first], points[second]))
    return np.asarray(segments, dtype=np.float32)


def _camera_points_to_scene(
    points: np.ndarray,
    quaternion: tuple[float, float, float, float],
    translation: np.ndarray | None = None,
) -> np.ndarray:
    """Legacy numerical adapter retained for external diagnostics."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("camera points must have shape (N, 3)")
    w, x, y, z = quaternion
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    offset = np.zeros(3, dtype=np.float64) if translation is None else np.asarray(translation)
    return (values @ (_CAMERA_TO_SCENE @ rotation).T + offset).astype(np.float32)


def _glasses_frame_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Deprecated compatibility helper; the widget no longer creates a glasses model."""

    return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)


__all__ = [
    "SpatialPoseSnapshot",
    "SpatialReferenceFrame",
    "SpatialSceneState",
    "SpatialSyncCanvas",
    "SpatialSyncCanvasStatus",
]
