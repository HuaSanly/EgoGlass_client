"""Typed object-perception artifacts for the offline HumanEgo-style pipeline."""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_box(self) -> BoundingBox:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bounding box must have positive area")
        return self


class ObjectMaskObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    object_id: str = Field(min_length=1)
    clip_id: str = Field(min_length=1)
    frame_index: int = Field(ge=0)
    session_time_ns: int = Field(ge=0)
    mask_relative_path: str = Field(min_length=1)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    boxes: tuple[BoundingBox, ...] = ()
    mask_area_ratio: float = Field(ge=0.0, le=1.0)


class ObjectKeypointTrack(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    object_id: str = Field(min_length=1)
    clip_id: str = Field(min_length=1)
    frame_indices: tuple[int, ...] = Field(min_length=1)
    session_times_ns: tuple[int, ...] = Field(min_length=1)
    points_xy_px: tuple[tuple[tuple[float, float], ...], ...] = Field(min_length=1)
    visibility: tuple[tuple[float, ...], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_track_shape(self) -> ObjectKeypointTrack:
        count = len(self.frame_indices)
        if len(self.session_times_ns) != count:
            raise ValueError("track timestamps must match frame count")
        if len(self.points_xy_px) != count or len(self.visibility) != count:
            raise ValueError("track observations must match frame count")
        if any(frame < 0 for frame in self.frame_indices):
            raise ValueError("track frame index must be non-negative")
        if any(
            current <= previous
            for previous, current in zip(self.frame_indices, self.frame_indices[1:], strict=False)
        ):
            raise ValueError("track frame indices must strictly increase")
        if any(
            current <= previous
            for previous, current in zip(
                self.session_times_ns, self.session_times_ns[1:], strict=False
            )
        ):
            raise ValueError("track timestamps must strictly increase")
        point_count = len(self.points_xy_px[0])
        if point_count < 1 or any(len(points) != point_count for points in self.points_xy_px):
            raise ValueError("track point count must be stable")
        if any(len(values) != point_count for values in self.visibility):
            raise ValueError("track visibility must match point count")
        if any(
            not math.isfinite(value)
            for frame in self.points_xy_px
            for point in frame
            for value in point
        ):
            raise ValueError("track points must be finite")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for frame in self.visibility
            for value in frame
        ):
            raise ValueError("track visibility must be finite and within [0, 1]")
        return self


class ObjectPose(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    object_id: str = Field(min_length=1)
    clip_id: str = Field(min_length=1)
    frame_index: int = Field(ge=0)
    session_time_ns: int = Field(ge=0)
    transform_object_to_world: tuple[float, ...] = Field(min_length=16, max_length=16)
    source: str = Field(min_length=1)
    grasped_by: str | None = None
    dynamic: bool = False

    @model_validator(mode="after")
    def validate_transform(self) -> ObjectPose:
        _validate_transform(self.transform_object_to_world, "object pose")
        return self


class ObjectTriangulation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    object_id: str = Field(min_length=1)
    points_world_m: tuple[tuple[float, float, float], ...] = Field(min_length=3)
    transform_object_to_world: tuple[float, ...] = Field(min_length=16, max_length=16)
    transform_object_to_camera: tuple[float, ...] = Field(min_length=16, max_length=16)
    mean_reprojection_error_px: float = Field(ge=0.0)
    contributing_frame_count: int = Field(ge=2)
    valid_point_count: int = Field(ge=3)
    orientation_method: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_geometry(self) -> ObjectTriangulation:
        if self.valid_point_count != len(self.points_world_m):
            raise ValueError("valid point count must match the stored point cloud")
        if any(not math.isfinite(value) for point in self.points_world_m for value in point):
            raise ValueError("object point cloud must be finite")
        _validate_transform(self.transform_object_to_world, "object-to-world transform")
        _validate_transform(self.transform_object_to_camera, "object-to-camera transform")
        return self


class ObjectTrackingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    schema_version: str = "1.0"
    processing_run_id: str = Field(min_length=1)
    task_profile_id: str = Field(min_length=1)
    masks: tuple[ObjectMaskObservation, ...]
    tracks: tuple[ObjectKeypointTrack, ...]
    triangulations: tuple[ObjectTriangulation, ...]
    poses: tuple[ObjectPose, ...]


def _validate_transform(values: tuple[float, ...], name: str) -> None:
    matrix = np.asarray(values, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError(f"{name} must be a homogeneous transform")
    if abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-9:
        raise ValueError(f"{name} rotation must be invertible")
