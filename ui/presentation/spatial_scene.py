"""Coordinate-aware scene state for the spatial synchronization view."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

from schemas import VioPose

_IDENTITY_TRANSFORM = np.eye(4, dtype=np.float64)


class SpatialReferenceFrame(StrEnum):
    WORLD = "world"
    CAMERA = "camera"


class _OrientationPose(Protocol):
    quaternion_wxyz: tuple[float, float, float, float]
    samples_received: int


@dataclass(frozen=True, slots=True)
class SpatialSceneState:
    """Already-transformed geometry consumed by the OpenGL widget."""

    reference_frame: SpatialReferenceFrame
    left_hand_points_m: tuple[tuple[float, float, float], ...]
    right_hand_points_m: tuple[tuple[float, float, float], ...]
    head_axes_m: tuple[tuple[float, float, float], ...]
    camera_axes_m: tuple[tuple[float, float, float], ...]
    head_pose_source: str
    latest_frame_index: int | None
    has_imu_pose: bool
    has_vio_pose: bool
    show_ground: bool
    object_points_m: tuple[tuple[float, float, float], ...] = ()
    object_axes_m: tuple[tuple[tuple[float, float, float], ...], ...] = ()


def build_spatial_scene_state(
    reference_frame: SpatialReferenceFrame,
    *,
    hand_result: dict[str, object] | None = None,
    imu_pose: _OrientationPose | None = None,
    vio_pose: VioPose | None = None,
    vio_first_pose: VioPose | None = None,
    transform_camera_to_imu: tuple[tuple[float, float, float, float], ...] = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    allow_imu_world_fallback: bool = True,
    object_overlays: object | None = None,
) -> SpatialSceneState:
    """Build one scene using a single frame's IMU, VIO and hand observations.

    In world mode Basalt's first pose becomes the display origin. When no VIO
    pose is available, the IMU relative orientation is used with zero
    translation. Camera mode intentionally leaves hand points in camera axes.
    """

    hands = _hands_from_result(hand_result)
    frame_index = _frame_index(hand_result)
    camera_axes = _axis_points(0.22)
    head_axes: tuple[tuple[float, float, float], ...] = ()
    source = "等待 VIO"
    has_imu = imu_pose is not None and imu_pose.samples_received > 0
    has_vio = vio_pose is not None

    if reference_frame is SpatialReferenceFrame.CAMERA:
        left, right = hands
        return SpatialSceneState(
            reference_frame,
            left,
            right,
            (),
            camera_axes,
            "相机坐标系",
            frame_index,
            has_imu,
            has_vio,
            False,
        )

    if vio_pose is not None and vio_first_pose is not None:
        display_imu = _relative_vio_transform(vio_first_pose, vio_pose)
        source = "Basalt VIO"
    elif has_imu and allow_imu_world_fallback:
        display_imu = _orientation_transform(imu_pose.quaternion_wxyz)
        source = "IMU 朝向，无平移"
    else:
        display_imu = _IDENTITY_TRANSFORM.copy()
        source = "等待 IMU / VIO"

    if not has_vio and not allow_imu_world_fallback:
        return SpatialSceneState(
            reference_frame,
            (),
            (),
            (),
            (),
            "世界坐标不可用",
            frame_index,
            has_imu,
            False,
            False,
        )

    transform_i_c = _matrix4(transform_camera_to_imu)
    transform_display_camera = display_imu @ transform_i_c
    left, right = hands
    left_world = _transform_points(left, transform_display_camera)
    right_world = _transform_points(right, transform_display_camera)
    head_axes = tuple(
        tuple(float(value) for value in point)
        for point in _transform_points(_axis_points(0.28), display_imu)
    )
    camera_axes_world = tuple(
        tuple(float(value) for value in point)
        for point in _transform_points(_axis_points(0.22), transform_display_camera)
    )
    overlay_payload = (
        object_overlays
        if object_overlays is not None
        else hand_result.get("object_overlays")
        if isinstance(hand_result, dict)
        else None
    )
    object_points, object_axes = _object_geometry(
        overlay_payload,
        vio_first_pose=vio_first_pose,
    )
    return SpatialSceneState(
        reference_frame,
        left_world,
        right_world,
        head_axes,
        camera_axes_world,
        source,
        frame_index,
        has_imu,
        has_vio,
        True,
        object_points,
        object_axes,
    )


def _relative_vio_transform(first: VioPose, current: VioPose) -> np.ndarray:
    return np.linalg.inv(_vio_transform(first)) @ _vio_transform(current)


def _vio_transform(pose: VioPose) -> np.ndarray:
    transform = _orientation_transform(pose.quaternion_wxyz)
    transform[:3, 3] = np.asarray(pose.position_m, dtype=np.float64)
    return transform


def _orientation_transform(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if values.shape != (4,) or not math.isfinite(norm) or norm < 1e-9:
        return _IDENTITY_TRANSFORM.copy()
    w, x, y, z = values / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _matrix4(value: tuple[tuple[float, float, float, float], ...]) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("camera-to-IMU transform must be a finite 4x4 matrix")
    return matrix


def _transform_points(
    points: tuple[tuple[float, float, float], ...],
    transform: np.ndarray,
) -> tuple[tuple[float, float, float], ...]:
    if not points:
        return ()
    homogeneous = np.column_stack(
        (np.asarray(points, dtype=np.float64), np.ones(len(points), dtype=np.float64))
    )
    result = (transform @ homogeneous.T).T[:, :3]
    return tuple(tuple(float(value) for value in point) for point in result)


def _axis_points(length: float) -> tuple[tuple[float, float, float], ...]:
    return (
        (0.0, 0.0, 0.0),
        (length, 0.0, 0.0),
        (0.0, length, 0.0),
        (0.0, 0.0, length),
    )


def _hands_from_result(
    result: dict[str, object] | None,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, float, float], ...],
]:
    found: dict[str, tuple[tuple[float, float, float], ...]] = {}
    if isinstance(result, dict) and isinstance(result.get("hands"), list):
        for raw in result["hands"]:
            if not isinstance(raw, dict) or raw.get("handedness") not in {"left", "right"}:
                continue
            points = _points(raw.get("keypoints_3d_camera_m"))
            if len(points) == 21:
                found[str(raw["handedness"])] = points
    return found.get("left", ()), found.get("right", ())


def _points(value: object) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(value, list):
        return ()
    points: list[tuple[float, float, float]] = []
    for point in value:
        if (
            not isinstance(point, list)
            or len(point) != 3
            or not all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in point
            )
            or not all(math.isfinite(float(item)) for item in point)
        ):
            return ()
        points.append(tuple(float(item) for item in point))
    return tuple(points)


def _frame_index(result: dict[str, object] | None) -> int | None:
    if not isinstance(result, dict):
        return None
    value = result.get("frame_index")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _object_geometry(
    overlays: object,
    *,
    vio_first_pose: VioPose | None,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[tuple[float, float, float], ...], ...],
]:
    """Map current object clouds and coordinate frames into display-world axes."""

    if not isinstance(overlays, list) or vio_first_pose is None:
        return (), ()
    transform_display_world = np.linalg.inv(_vio_transform(vio_first_pose))
    display_points: list[tuple[float, float, float]] = []
    display_axes: list[tuple[tuple[float, float, float], ...]] = []
    for overlay in overlays:
        if not isinstance(overlay, dict):
            continue
        triangulation = overlay.get("triangulation")
        if not isinstance(triangulation, dict):
            continue
        points_world = _finite_points(triangulation.get("points_world_m"))
        initial_pose = _flat_transform(triangulation.get("transform_object_to_world"))
        if not points_world or initial_pose is None:
            continue
        pose = overlay.get("pose")
        current_pose = initial_pose
        if pose is not None:
            if not isinstance(pose, dict):
                continue
            current_pose = _flat_transform(pose.get("transform_object_to_world"))
            if current_pose is None:
                continue
        transform_display_object = transform_display_world @ current_pose
        transform_display_points = transform_display_object @ np.linalg.inv(initial_pose)
        display_points.extend(_transform_points(points_world, transform_display_points))
        display_axes.append(_transform_points(_axis_points(0.12), transform_display_object))
    return tuple(display_points), tuple(display_axes)


def _finite_points(value: object) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    points: list[tuple[float, float, float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 3:
            return ()
        if not all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in point
        ):
            return ()
        points.append(tuple(float(item) for item in point))
    return tuple(points)


def _flat_transform(value: object) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or len(value) != 16:
        return None
    if not all(
        isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math.isfinite(float(item))
        for item in value
    ):
        return None
    transform = np.asarray(value, dtype=np.float64).reshape(4, 4)
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        return None
    try:
        np.linalg.inv(transform)
    except np.linalg.LinAlgError:
        return None
    return transform


__all__ = ["SpatialReferenceFrame", "SpatialSceneState", "build_spatial_scene_state"]
