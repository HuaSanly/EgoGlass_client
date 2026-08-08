"""HumanEgo-style DLT plus robust point-only bundle adjustment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from schemas.object_tracking import ObjectTriangulation
from schemas.trajectory import VioTrajectory

from .config import ObjectTrackingConfig, ObjectTrackingError


@dataclass(frozen=True, slots=True)
class CameraObservation:
    frame_index: int
    session_time_ns: int
    transform_camera_to_world: NDArray[np.float64]
    intrinsics: NDArray[np.float64]


class MultiViewTriangulator:
    """Recover a static object point cloud from known moving-camera observations."""

    def __init__(self, config: ObjectTrackingConfig) -> None:
        self.config = config

    def triangulate(
        self,
        object_id: str,
        observations: tuple[CameraObservation, ...],
        tracks_xy_px: NDArray[np.float32],
        visibility: NDArray[np.float32],
        reference_index: int,
        orientation_method: str,
    ) -> ObjectTriangulation:
        if len(observations) < 2:
            raise ObjectTrackingError("triangulation needs at least two camera observations")
        if tracks_xy_px.shape[:2] != visibility.shape or tracks_xy_px.shape[0] != len(observations):
            raise ValueError("track and visibility shapes do not match camera observations")
        if not 0 <= reference_index < len(observations):
            raise ValueError("triangulation reference index is invalid")
        selected = tuple(range(0, len(observations), self.config.triangulation_frame_stride))
        if selected[-1] != len(observations) - 1:
            selected = (*selected, len(observations) - 1)
        cameras = tuple(observations[index] for index in selected)
        points = tracks_xy_px[np.asarray(selected)]
        vis = visibility[np.asarray(selected)]
        reconstructed: list[NDArray[np.float64]] = []
        errors: list[float] = []
        for point_index in range(points.shape[1]):
            initial = _triangulate_dlt(cameras, points[:, point_index], vis[:, point_index])
            if initial is None:
                continue
            refined = _bundle_adjust(
                initial,
                cameras,
                points[:, point_index],
                vis[:, point_index],
                self.config.triangulation_huber_scale_px,
            )
            error = _mean_reprojection_error(
                refined, cameras, points[:, point_index], vis[:, point_index]
            )
            if np.isfinite(error):
                reconstructed.append(refined)
                errors.append(error)
        if len(reconstructed) < 3:
            raise ObjectTrackingError("object triangulation produced fewer than three valid points")
        cloud = np.asarray(reconstructed, dtype=np.float64)
        transform_object_to_world = _pca_pose(cloud, orientation_method)
        reference_camera = observations[reference_index].transform_camera_to_world
        transform_object_to_camera = np.linalg.inv(reference_camera) @ transform_object_to_world
        return ObjectTriangulation(
            object_id=object_id,
            points_world_m=tuple(tuple(float(value) for value in point) for point in cloud),
            transform_object_to_world=tuple(
                float(value) for value in transform_object_to_world.reshape(-1)
            ),
            transform_object_to_camera=tuple(
                float(value) for value in transform_object_to_camera.reshape(-1)
            ),
            mean_reprojection_error_px=float(np.mean(errors)),
            contributing_frame_count=len(cameras),
            valid_point_count=len(cloud),
            orientation_method=orientation_method,
        )

    def camera_observations(
        self,
        frame_indices: tuple[int, ...],
        session_times_ns: tuple[int, ...],
        intrinsics: tuple[NDArray[np.float64], ...],
        trajectory: VioTrajectory,
        transform_camera_to_imu: NDArray[np.float64],
    ) -> tuple[CameraObservation, ...]:
        if not (len(frame_indices) == len(session_times_ns) == len(intrinsics)):
            raise ValueError("camera observation vectors must have matching lengths")
        maximum_gap_ns = self.config.maximum_vio_pose_gap_ms * 1_000_000
        observations: list[CameraObservation] = []
        for frame_index, timestamp_ns, matrix in zip(
            frame_indices, session_times_ns, intrinsics, strict=True
        ):
            pose = trajectory.pose_at(timestamp_ns, max_gap_ns=maximum_gap_ns)
            if pose is None:
                raise ObjectTrackingError("VIO coverage is incomplete for triangulation")
            transform_imu_to_world = _transform_from_pose(pose.position_m, pose.quaternion_wxyz)
            observations.append(
                CameraObservation(
                    frame_index=frame_index,
                    session_time_ns=timestamp_ns,
                    transform_camera_to_world=transform_imu_to_world @ transform_camera_to_imu,
                    intrinsics=np.asarray(matrix, dtype=np.float64),
                )
            )
        return tuple(observations)


def _triangulate_dlt(
    cameras: tuple[CameraObservation, ...],
    observations_xy: NDArray[np.float32],
    visibility: NDArray[np.float32],
) -> NDArray[np.float64] | None:
    rows: list[NDArray[np.float64]] = []
    for camera, xy, visible in zip(cameras, observations_xy, visibility, strict=True):
        if visible <= 0.0:
            continue
        fx, fy = camera.intrinsics[0, 0], camera.intrinsics[1, 1]
        cx, cy = camera.intrinsics[0, 2], camera.intrinsics[1, 2]
        if fx <= 0.0 or fy <= 0.0:
            continue
        u = (float(xy[0]) - cx) / fx
        v = (float(xy[1]) - cy) / fy
        projection = np.linalg.inv(camera.transform_camera_to_world)[:3, :]
        rows.extend((u * projection[2] - projection[0], v * projection[2] - projection[1]))
    if len(rows) < 4:
        return None
    _, _, right = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    homogeneous = right[-1]
    if abs(homogeneous[3]) < 1e-9:
        return None
    return homogeneous[:3] / homogeneous[3]


def _bundle_adjust(
    initial: NDArray[np.float64],
    cameras: tuple[CameraObservation, ...],
    observations_xy: NDArray[np.float32],
    visibility: NDArray[np.float32],
    huber_scale: float,
) -> NDArray[np.float64]:
    def residual(point: NDArray[np.float64]) -> NDArray[np.float64]:
        output: list[float] = []
        for camera, xy, visible in zip(cameras, observations_xy, visibility, strict=True):
            if visible <= 0.0:
                continue
            camera_point = (
                np.linalg.inv(camera.transform_camera_to_world)[:3, :3] @ point
                + np.linalg.inv(camera.transform_camera_to_world)[:3, 3]
            )
            if camera_point[2] <= 1e-6:
                continue
            projected = camera.intrinsics @ camera_point
            output.extend(projected[:2] / projected[2] - xy)
        return np.asarray(output, dtype=np.float64)

    solution = least_squares(residual, initial, loss="huber", f_scale=huber_scale, max_nfev=60)
    return np.asarray(solution.x, dtype=np.float64)


def _mean_reprojection_error(
    point: NDArray[np.float64],
    cameras: tuple[CameraObservation, ...],
    observations_xy: NDArray[np.float32],
    visibility: NDArray[np.float32],
) -> float:
    errors: list[float] = []
    for camera, xy, visible in zip(cameras, observations_xy, visibility, strict=True):
        if visible <= 0.0:
            continue
        transform_world_to_camera = np.linalg.inv(camera.transform_camera_to_world)
        camera_point = transform_world_to_camera[:3, :3] @ point + transform_world_to_camera[:3, 3]
        if camera_point[2] <= 1e-6:
            continue
        projected = camera.intrinsics @ camera_point
        errors.append(float(np.linalg.norm(projected[:2] / projected[2] - xy)))
    return float(np.mean(errors)) if errors else float("inf")


def _pca_pose(points_world: NDArray[np.float64], method: str) -> NDArray[np.float64]:
    center = points_world.mean(axis=0)
    _, _, right = np.linalg.svd(points_world - center, full_matrices=False)
    x_axis = right[0]
    z_axis = right[2]
    if method == "pca2":
        z_axis = center / max(float(np.linalg.norm(center)), 1e-9)
    x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-9)
    z_axis = z_axis - x_axis * float(np.dot(z_axis, x_axis))
    z_axis = z_axis / max(float(np.linalg.norm(z_axis)), 1e-9)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-9)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 2] *= -1.0
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = center
    return transform


def _transform_from_pose(
    position: tuple[float, float, float], quaternion_wxyz: tuple[float, float, float, float]
) -> NDArray[np.float64]:
    w, x, y, z = quaternion_wxyz
    norm = max(float(np.sqrt(w * w + x * x + y * y + z * z)), 1e-9)
    w, x, y, z = (value / norm for value in (w, x, y, z))
    rotation = np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform
