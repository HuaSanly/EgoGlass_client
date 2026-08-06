"""HumanEgo-derived temporal cleanup and world-space hand kinematics.

Required Notice: Copyright (c) 2026 The HumanEgo Authors -
https://humanego-ai.github.io
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from .models import (
    Handedness,
    HandKinematics,
    HandTemporalMetadata,
    HandTrackingResult,
    OfflineHandTemporalConfig,
    TemporalSource,
    TrackedHand,
    readonly_float_array,
)

_THUMB_TIP = 0
_INDEX_TIP = 1
_WRIST = 5
_THUMB_BASE = 6
_INDEX_BASE = 8
_MIDDLE_BASE = 11
_PALM_CENTER = 20
_SIDES = (Handedness.LEFT, Handedness.RIGHT)


class VioPoseLike(Protocol):
    timestamp_ns: int
    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]


class VioTrajectoryLike(Protocol):
    def pose_at(
        self, timestamp_ns: int, *, max_gap_ns: int | None = None
    ) -> VioPoseLike | None: ...


@dataclass(frozen=True, slots=True)
class TemporalProcessingStats:
    """Measurable evidence for every temporal processing stage."""

    raw_hand_frames: int
    confidence_rejected: int
    interpolated_frames: int
    suppressed_frames: int
    final_hand_frames: int
    grasp_transitions_before: int
    grasp_transitions_after: int
    grasp_states_changed: int
    vio_matched_frames: int
    world_optimized_frames: int
    temporal_processing_duration_ns: int
    confidence_filter_duration_ns: int = 0
    interpolation_duration_ns: int = 0
    segment_suppression_duration_ns: int = 0
    grasp_smoothing_duration_ns: int = 0
    world_mapping_duration_ns: int = 0
    kinematic_optimization_duration_ns: int = 0

    @property
    def vio_coverage_ratio(self) -> float:
        if self.final_hand_frames == 0:
            return 0.0
        return self.vio_matched_frames / self.final_hand_frames


@dataclass(frozen=True, slots=True)
class TemporalProcessingOutput:
    """Raw immutable inference and finalized frame-addressable results."""

    raw_results: tuple[HandTrackingResult, ...]
    final_results: tuple[HandTrackingResult, ...]
    stats: TemporalProcessingStats
    partial_world_coverage: bool


class OfflineHandTemporalProcessor:
    """Apply HumanEgo cleanup without importing UI or Project Aria code."""

    def __init__(
        self,
        config: OfflineHandTemporalConfig,
        *,
        grasp_ratio_threshold: float,
    ) -> None:
        if not np.isfinite(grasp_ratio_threshold) or grasp_ratio_threshold <= 0.0:
            raise ValueError("grasp_ratio_threshold must be finite and positive")
        self.config = config
        self.grasp_ratio_threshold = float(grasp_ratio_threshold)

    def process_clip(
        self,
        results: Sequence[HandTrackingResult],
        *,
        trajectory: VioTrajectoryLike | None,
        transform_camera_to_imu: object | None,
    ) -> TemporalProcessingOutput:
        """Finalize one clip; data never flows across sequence boundaries."""

        started_ns = time.perf_counter_ns()
        raw_results = tuple(results)
        self._validate_clip(raw_results)
        raw_hand_frames = sum(len(result.hands) for result in raw_results)
        if not self.config.enabled:
            stats = TemporalProcessingStats(
                raw_hand_frames=raw_hand_frames,
                confidence_rejected=0,
                interpolated_frames=0,
                suppressed_frames=0,
                final_hand_frames=raw_hand_frames,
                grasp_transitions_before=0,
                grasp_transitions_after=0,
                grasp_states_changed=0,
                vio_matched_frames=0,
                world_optimized_frames=0,
                temporal_processing_duration_ns=time.perf_counter_ns() - started_ns,
            )
            return TemporalProcessingOutput(raw_results, raw_results, stats, trajectory is None)

        stage_started_ns = time.perf_counter_ns()
        hands_by_frame, rejected = self._confidence_filter(raw_results)
        confidence_filter_duration_ns = time.perf_counter_ns() - stage_started_ns

        stage_started_ns = time.perf_counter_ns()
        interpolated = self._interpolate_short_gaps(raw_results, hands_by_frame)
        interpolation_duration_ns = time.perf_counter_ns() - stage_started_ns

        stage_started_ns = time.perf_counter_ns()
        suppressed = self._suppress_short_segments(hands_by_frame)
        segment_suppression_duration_ns = time.perf_counter_ns() - stage_started_ns

        transitions_before = _count_grasp_transitions(hands_by_frame)
        stage_started_ns = time.perf_counter_ns()
        grasp_changed = self._smooth_grasp(hands_by_frame)
        grasp_smoothing_duration_ns = time.perf_counter_ns() - stage_started_ns

        transform = _validated_transform(transform_camera_to_imu)
        stage_started_ns = time.perf_counter_ns()
        matched = self._attach_world_kinematics(raw_results, hands_by_frame, trajectory, transform)
        world_mapping_duration_ns = time.perf_counter_ns() - stage_started_ns

        stage_started_ns = time.perf_counter_ns()
        optimized = self._optimize_world_segments(raw_results, hands_by_frame)
        kinematic_optimization_duration_ns = time.perf_counter_ns() - stage_started_ns

        stage_started_ns = time.perf_counter_ns()
        grasp_changed += self._smooth_grasp(hands_by_frame)
        grasp_smoothing_duration_ns += time.perf_counter_ns() - stage_started_ns
        transitions_after = _count_grasp_transitions(hands_by_frame)

        final_results = tuple(
            replace(
                result,
                hands=tuple(
                    hand
                    for side in _SIDES
                    if (hand := hands_by_frame[index].get(side)) is not None
                ),
            )
            for index, result in enumerate(raw_results)
        )
        final_hand_frames = sum(len(result.hands) for result in final_results)
        stats = TemporalProcessingStats(
            raw_hand_frames=raw_hand_frames,
            confidence_rejected=rejected,
            interpolated_frames=interpolated,
            suppressed_frames=suppressed,
            final_hand_frames=final_hand_frames,
            grasp_transitions_before=transitions_before,
            grasp_transitions_after=transitions_after,
            grasp_states_changed=grasp_changed,
            vio_matched_frames=matched,
            world_optimized_frames=optimized,
            temporal_processing_duration_ns=time.perf_counter_ns() - started_ns,
            confidence_filter_duration_ns=confidence_filter_duration_ns,
            interpolation_duration_ns=interpolation_duration_ns,
            segment_suppression_duration_ns=segment_suppression_duration_ns,
            grasp_smoothing_duration_ns=grasp_smoothing_duration_ns,
            world_mapping_duration_ns=world_mapping_duration_ns,
            kinematic_optimization_duration_ns=kinematic_optimization_duration_ns,
        )
        partial = trajectory is None or matched < final_hand_frames
        return TemporalProcessingOutput(raw_results, final_results, stats, partial)

    @staticmethod
    def _validate_clip(results: tuple[HandTrackingResult, ...]) -> None:
        if not results:
            return
        sequence_id = results[0].sequence_id
        previous_time = -1
        for result in results:
            if result.sequence_id != sequence_id:
                raise ValueError("process_clip cannot cross clip boundaries")
            if result.session_time_ns <= previous_time:
                raise ValueError("clip results must have strictly increasing session timestamps")
            previous_time = result.session_time_ns

    def _confidence_filter(
        self,
        results: tuple[HandTrackingResult, ...],
    ) -> tuple[list[dict[Handedness, TrackedHand]], int]:
        frames: list[dict[Handedness, TrackedHand]] = []
        rejected = 0
        for result in results:
            selected: dict[Handedness, TrackedHand] = {}
            for hand in result.hands:
                if hand.confidence < self.config.confidence_threshold:
                    rejected += 1
                    continue
                current = selected.get(hand.handedness)
                if current is None or hand.confidence > current.confidence:
                    if current is not None:
                        rejected += 1
                    selected[hand.handedness] = replace(
                        hand,
                        temporal=HandTemporalMetadata(TemporalSource.OBSERVED),
                        kinematics=None,
                    )
                else:
                    rejected += 1
            frames.append(selected)
        return frames, rejected

    def _interpolate_short_gaps(
        self,
        results: tuple[HandTrackingResult, ...],
        frames: list[dict[Handedness, TrackedHand]],
    ) -> int:
        count = 0
        for side in _SIDES:
            present = [index for index, frame in enumerate(frames) if side in frame]
            for start, end in zip(present, present[1:], strict=False):
                gap = end - start - 1
                if gap < 1 or gap > self.config.interpolation_max_gap_frames:
                    continue
                first = frames[start][side]
                last = frames[end][side]
                for offset in range(1, gap + 1):
                    ratio = offset / (gap + 1)
                    frames[start + offset][side] = _interpolate_hand(
                        first,
                        last,
                        ratio,
                        source_frames=(results[start].frame_index, results[end].frame_index),
                        grasp_ratio_threshold=self.grasp_ratio_threshold,
                    )
                    count += 1
        return count

    def _suppress_short_segments(self, frames: list[dict[Handedness, TrackedHand]]) -> int:
        count = 0
        for side in _SIDES:
            for start, end in _true_runs([side in frame for frame in frames]):
                if end - start >= self.config.minimum_segment_frames:
                    continue
                for index in range(start, end):
                    frames[index].pop(side, None)
                    count += 1
        return count

    def _smooth_grasp(self, frames: list[dict[Handedness, TrackedHand]]) -> int:
        changed = 0
        for side in _SIDES:
            presence = np.asarray([side in frame for frame in frames], dtype=bool)
            for start, end in _true_runs(presence.tolist()):
                ratios = np.asarray(
                    [frames[index][side].grasp_ratio for index in range(start, end)],
                    dtype=np.float64,
                )
                window = min(self.config.grasp_smoothing_window_frames, len(ratios))
                if window > 1:
                    kernel = np.ones(window, dtype=np.float64)
                    sample_count = np.convolve(
                        np.ones(len(ratios), dtype=np.float64), kernel, mode="same"
                    )
                    ratios = np.convolve(ratios, kernel, mode="same") / sample_count
                binary = _suppress_bracketed_flicker(
                    ratios < self.grasp_ratio_threshold,
                    maximum_run=self.config.grasp_flicker_max_frames,
                )
                for offset, index in enumerate(range(start, end)):
                    hand = frames[index][side]
                    state = bool(binary[offset])
                    if state != hand.is_grasping:
                        changed += 1
                        frames[index][side] = replace(hand, is_grasping=state)
        return changed

    def _attach_world_kinematics(
        self,
        results: tuple[HandTrackingResult, ...],
        frames: list[dict[Handedness, TrackedHand]],
        trajectory: VioTrajectoryLike | None,
        transform_camera_to_imu: NDArray[np.float64] | None,
    ) -> int:
        if trajectory is None or transform_camera_to_imu is None:
            return 0
        max_gap_ns = self.config.maximum_vio_pose_gap_ms * 1_000_000
        matched = 0
        for result, frame in zip(results, frames, strict=True):
            pose = trajectory.pose_at(result.session_time_ns, max_gap_ns=max_gap_ns)
            if pose is None:
                continue
            world_from_imu = _pose_matrix(pose)
            world_from_camera = world_from_imu @ transform_camera_to_imu
            difference_ns = abs(int(pose.timestamp_ns) - result.session_time_ns)
            for side, hand in tuple(frame.items()):
                world_points = _transform_points(world_from_camera, hand.keypoints_3d_camera_m)
                wrist_pose = _wrist_pose(world_points)
                midpoint_pose = _midpoint_pose(world_points, wrist_pose[:3, :3])
                kinematics = _initial_kinematics(world_points, wrist_pose, midpoint_pose)
                temporal = replace(
                    hand.temporal or HandTemporalMetadata(TemporalSource.OBSERVED),
                    vio_pose_timestamp_ns=int(pose.timestamp_ns),
                    vio_time_difference_ns=difference_ns,
                    world_kinematics_available=True,
                    kinematics_optimized=False,
                )
                frame[side] = replace(hand, temporal=temporal, kinematics=kinematics)
                matched += 1
        return matched

    def _optimize_world_segments(
        self,
        results: tuple[HandTrackingResult, ...],
        frames: list[dict[Handedness, TrackedHand]],
    ) -> int:
        optimized_count = 0
        for side in _SIDES:
            present = [side in frame for frame in frames]
            available = [
                (hand := frame.get(side)) is not None and hand.kinematics is not None
                for frame in frames
            ]
            for start, end in _joined_world_runs(
                present,
                available,
                maximum_gap=self.config.smoothing_fill_max_gap_frames,
            ):
                valid_indexes = [index for index in range(start, end) if available[index]]
                if len(valid_indexes) < self.config.minimum_smoothing_frames:
                    self._assign_velocities(results, frames, side, valid_indexes)
                    continue
                raw = [frames[index][side].kinematics for index in valid_indexes]
                assert all(item is not None for item in raw)
                kinematics = [item for item in raw if item is not None]
                offsets = [index - start for index in valid_indexes]
                sample_count = end - start
                wrist = self._smooth_positions(
                    _fill_position_samples(
                        sample_count,
                        offsets,
                        [item.wrist_pose_raw_world[:3, 3] for item in kinematics],
                    )
                )
                thumb = self._smooth_positions(
                    _fill_position_samples(
                        sample_count,
                        offsets,
                        [item.thumb_tip_raw_world_m for item in kinematics],
                    )
                )
                index_tip = self._smooth_positions(
                    _fill_position_samples(
                        sample_count,
                        offsets,
                        [item.index_tip_raw_world_m for item in kinematics],
                    )
                )
                thumb_base = self._smooth_positions(
                    _fill_position_samples(
                        sample_count,
                        offsets,
                        [item.thumb_base_raw_world_m for item in kinematics],
                    )
                )
                index_base = self._smooth_positions(
                    _fill_position_samples(
                        sample_count,
                        offsets,
                        [item.index_base_raw_world_m for item in kinematics],
                    )
                )
                wrist_rotations = _ema_rotations(
                    _fill_rotation_samples(
                        sample_count,
                        offsets,
                        [item.wrist_pose_raw_world[:3, :3] for item in kinematics],
                    ),
                    self.config.orientation_ema_alpha,
                )
                midpoint_rotations = _smoothed_midpoint_rotations(
                    thumb,
                    index_tip,
                    thumb_base,
                    index_base,
                    wrist,
                    self.config.orientation_ema_alpha,
                    fallback=wrist_rotations,
                )
                midpoint = (thumb + index_tip) * 0.5
                for frame_index in valid_indexes:
                    offset = frame_index - start
                    hand = frames[frame_index][side]
                    current = hand.kinematics
                    assert current is not None
                    wrist_pose = _make_pose(wrist_rotations[offset], wrist[offset])
                    midpoint_pose = _make_pose(midpoint_rotations[offset], midpoint[offset])
                    frames[frame_index][side] = replace(
                        hand,
                        temporal=replace(hand.temporal, kinematics_optimized=True),  # type: ignore[arg-type]
                        kinematics=replace(
                            current,
                            wrist_pose_optimized_world=_readonly_matrix(wrist_pose),
                            midpoint_pose_optimized_world=_readonly_matrix(midpoint_pose),
                            thumb_tip_optimized_world_m=_readonly_vector(thumb[offset]),
                            index_tip_optimized_world_m=_readonly_vector(index_tip[offset]),
                            thumb_base_optimized_world_m=_readonly_vector(thumb_base[offset]),
                            index_base_optimized_world_m=_readonly_vector(index_base[offset]),
                        ),
                    )
                    optimized_count += 1
                self._assign_velocities(results, frames, side, valid_indexes)
        return optimized_count

    def _smooth_positions(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        count = len(values)
        window = min(
            self.config.sg_window_frames,
            count if count % 2 else count - 1,
        )
        if window <= self.config.sg_polyorder or window < 3:
            return values.copy()
        return np.asarray(
            savgol_filter(
                values,
                window_length=window,
                polyorder=min(self.config.sg_polyorder, window - 1),
                axis=0,
                mode="interp",
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _assign_velocities(
        results: tuple[HandTrackingResult, ...],
        frames: list[dict[Handedness, TrackedHand]],
        side: Handedness,
        indexes: list[int],
    ) -> None:
        previous: tuple[int, HandKinematics] | None = None
        for index in indexes:
            hand = frames[index][side]
            current = hand.kinematics
            assert current is not None
            zero = np.zeros(3, dtype=np.float64)
            velocities = (zero, zero, zero, zero, zero, zero, zero, zero)
            if previous is not None:
                previous_time, old = previous
                dt = (results[index].session_time_ns - previous_time) / 1_000_000_000.0
                if dt > 0.0:
                    velocities = (
                        (
                            current.wrist_pose_raw_world[:3, 3]
                            - old.wrist_pose_raw_world[:3, 3]
                        )
                        / dt,
                        (
                            current.wrist_pose_optimized_world[:3, 3]
                            - old.wrist_pose_optimized_world[:3, 3]
                        )
                        / dt,
                        _angular_velocity(
                            old.wrist_pose_raw_world[:3, :3],
                            current.wrist_pose_raw_world[:3, :3],
                            dt,
                        ),
                        _angular_velocity(
                            old.wrist_pose_optimized_world[:3, :3],
                            current.wrist_pose_optimized_world[:3, :3],
                            dt,
                        ),
                        (
                            current.midpoint_pose_raw_world[:3, 3]
                            - old.midpoint_pose_raw_world[:3, 3]
                        )
                        / dt,
                        (
                            current.midpoint_pose_optimized_world[:3, 3]
                            - old.midpoint_pose_optimized_world[:3, 3]
                        )
                        / dt,
                        _angular_velocity(
                            old.midpoint_pose_raw_world[:3, :3],
                            current.midpoint_pose_raw_world[:3, :3],
                            dt,
                        ),
                        _angular_velocity(
                            old.midpoint_pose_optimized_world[:3, :3],
                            current.midpoint_pose_optimized_world[:3, :3],
                            dt,
                        ),
                    )
            updated = replace(
                current,
                wrist_linear_velocity_raw_m_s=_readonly_vector(velocities[0]),
                wrist_linear_velocity_optimized_m_s=_readonly_vector(velocities[1]),
                wrist_angular_velocity_raw_rad_s=_readonly_vector(velocities[2]),
                wrist_angular_velocity_optimized_rad_s=_readonly_vector(velocities[3]),
                midpoint_linear_velocity_raw_m_s=_readonly_vector(velocities[4]),
                midpoint_linear_velocity_optimized_m_s=_readonly_vector(velocities[5]),
                midpoint_angular_velocity_raw_rad_s=_readonly_vector(velocities[6]),
                midpoint_angular_velocity_optimized_rad_s=_readonly_vector(velocities[7]),
            )
            frames[index][side] = replace(hand, kinematics=updated)
            previous = (results[index].session_time_ns, updated)


def _interpolate_hand(
    first: TrackedHand,
    last: TrackedHand,
    ratio: float,
    *,
    source_frames: tuple[int, int],
    grasp_ratio_threshold: float,
) -> TrackedHand:
    points_first = np.asarray(first.keypoints_3d_camera_m, dtype=np.float64)
    points_last = np.asarray(last.keypoints_3d_camera_m, dtype=np.float64)
    pose_first = _wrist_pose(points_first)
    pose_last = _wrist_pose(points_last)
    rotation = Slerp(
        [0.0, 1.0],
        Rotation.from_matrix([pose_first[:3, :3], pose_last[:3, :3]]),
    )([ratio]).as_matrix()[0]
    translation = _lerp(pose_first[:3, 3], pose_last[:3, 3], ratio)
    local_first = (pose_first[:3, :3].T @ (points_first - pose_first[:3, 3]).T).T
    local_last = (pose_last[:3, :3].T @ (points_last - pose_last[:3, 3]).T).T
    points = (rotation @ _lerp(local_first, local_last, ratio).T).T + translation
    grasp_ratio = float(_lerp(first.grasp_ratio, last.grasp_ratio, ratio))
    return TrackedHand(
        handedness=first.handedness,
        confidence=float(_lerp(first.confidence, last.confidence, ratio)),
        detector_confidence=float(
            _lerp(first.detector_confidence, last.detector_confidence, ratio)
        ),
        reconstruction_quality=_optional_lerp(
            first.reconstruction_quality, last.reconstruction_quality, ratio
        ),
        depth_score=_optional_lerp(first.depth_score, last.depth_score, ratio),
        coverage_score=_optional_lerp(first.coverage_score, last.coverage_score, ratio),
        compactness_score=_optional_lerp(
            first.compactness_score, last.compactness_score, ratio
        ),
        reconstruction_backend=first.reconstruction_backend,
        metric_depth_status=first.metric_depth_status,
        bbox_xyxy_px=tuple(
            float(value)
            for value in _lerp(
                np.asarray(first.bbox_xyxy_px), np.asarray(last.bbox_xyxy_px), ratio
            )
        ),
        keypoints_2d_px=readonly_float_array(
            _lerp(first.keypoints_2d_px, last.keypoints_2d_px, ratio), (21, 2)
        ),
        keypoints_3d_camera_m=readonly_float_array(points, (21, 3)),
        joint_angles_degrees={
            key: float(
                _lerp(first.joint_angles_degrees[key], last.joint_angles_degrees[key], ratio)
            )
            for key in first.joint_angles_degrees.keys() & last.joint_angles_degrees.keys()
        },
        grasp_ratio=grasp_ratio,
        is_grasping=grasp_ratio < grasp_ratio_threshold,
        temporal=HandTemporalMetadata(
            TemporalSource.INTERPOLATED,
            interpolation_source_frames=source_frames,
        ),
    )


def _validated_transform(value: object | None) -> NDArray[np.float64] | None:
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("transform_camera_to_imu must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError("transform_camera_to_imu must be homogeneous")
    return matrix


def _pose_matrix(pose: VioPoseLike) -> NDArray[np.float64]:
    qw, qx, qy, qz = pose.quaternion_wxyz
    quaternion_xyzw = np.asarray((qx, qy, qz, qw), dtype=np.float64)
    rotation = Rotation.from_quat(quaternion_xyzw).as_matrix()
    return _make_pose(rotation, np.asarray(pose.position_m, dtype=np.float64))


def _transform_points(
    transform: NDArray[np.float64], points: NDArray[np.float32]
) -> NDArray[np.float64]:
    values = np.asarray(points, dtype=np.float64)
    return (transform[:3, :3] @ values.T).T + transform[:3, 3]


def _wrist_pose(points: NDArray[np.floating]) -> NDArray[np.float64]:
    wrist = np.asarray(points[_WRIST], dtype=np.float64)
    x_raw = np.asarray(points[_INDEX_BASE] - points[_THUMB_BASE], dtype=np.float64)
    y_raw = np.asarray(points[_MIDDLE_BASE] - points[_WRIST], dtype=np.float64)
    rotation = _orthonormal_basis(x_raw, y_raw)
    if rotation is None:
        palm = np.asarray(points[_PALM_CENTER] - points[_WRIST], dtype=np.float64)
        rotation = _orthonormal_basis(x_raw, palm)
    if rotation is None:
        rotation = np.eye(3, dtype=np.float64)
    return _make_pose(rotation, wrist)


def _midpoint_pose(
    points: NDArray[np.floating], fallback: NDArray[np.float64]
) -> NDArray[np.float64]:
    midpoint = (points[_THUMB_TIP] + points[_INDEX_TIP]) * 0.5
    x_raw = points[_INDEX_BASE] - points[_THUMB_BASE]
    y_raw = ((points[_THUMB_BASE] + points[_INDEX_BASE]) * 0.5) - points[_WRIST]
    rotation = _orthonormal_basis(x_raw, y_raw)
    return _make_pose(fallback if rotation is None else rotation, midpoint)


def _orthonormal_basis(
    x_value: NDArray[np.floating], y_value: NDArray[np.floating]
) -> NDArray[np.float64] | None:
    x = _unit(x_value)
    if x is None:
        return None
    y = np.asarray(y_value, dtype=np.float64) - np.dot(y_value, x) * x
    y = _unit(y)
    if y is None:
        return None
    z = _unit(np.cross(x, y))
    if z is None:
        return None
    y = _unit(np.cross(z, x))
    if y is None:
        return None
    return np.column_stack((x, y, z))


def _unit(value: NDArray[np.floating]) -> NDArray[np.float64] | None:
    array = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm < 1e-8:
        return None
    return array / norm


def _make_pose(
    rotation: NDArray[np.floating], translation: NDArray[np.floating]
) -> NDArray[np.float64]:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def _initial_kinematics(
    points: NDArray[np.float64],
    wrist_pose: NDArray[np.float64],
    midpoint_pose: NDArray[np.float64],
) -> HandKinematics:
    zero = _readonly_vector(np.zeros(3))
    return HandKinematics(
        keypoints_3d_world_m=readonly_float_array(points, (21, 3)),
        wrist_pose_raw_world=_readonly_matrix(wrist_pose),
        wrist_pose_optimized_world=_readonly_matrix(wrist_pose),
        midpoint_pose_raw_world=_readonly_matrix(midpoint_pose),
        midpoint_pose_optimized_world=_readonly_matrix(midpoint_pose),
        thumb_tip_raw_world_m=_readonly_vector(points[_THUMB_TIP]),
        thumb_tip_optimized_world_m=_readonly_vector(points[_THUMB_TIP]),
        index_tip_raw_world_m=_readonly_vector(points[_INDEX_TIP]),
        index_tip_optimized_world_m=_readonly_vector(points[_INDEX_TIP]),
        thumb_base_raw_world_m=_readonly_vector(points[_THUMB_BASE]),
        thumb_base_optimized_world_m=_readonly_vector(points[_THUMB_BASE]),
        index_base_raw_world_m=_readonly_vector(points[_INDEX_BASE]),
        index_base_optimized_world_m=_readonly_vector(points[_INDEX_BASE]),
        wrist_linear_velocity_raw_m_s=zero,
        wrist_linear_velocity_optimized_m_s=zero,
        wrist_angular_velocity_raw_rad_s=zero,
        wrist_angular_velocity_optimized_rad_s=zero,
        midpoint_linear_velocity_raw_m_s=zero,
        midpoint_linear_velocity_optimized_m_s=zero,
        midpoint_angular_velocity_raw_rad_s=zero,
        midpoint_angular_velocity_optimized_rad_s=zero,
    )


def _readonly_matrix(value: NDArray[np.floating]) -> NDArray[np.float32]:
    return readonly_float_array(value, (4, 4))


def _readonly_vector(value: NDArray[np.floating]) -> NDArray[np.float32]:
    return readonly_float_array(value, (3,))


def _ema_rotations(
    rotations: Sequence[NDArray[np.floating]], alpha: float
) -> list[NDArray[np.float64]]:
    result: list[NDArray[np.float64]] = []
    x_ema: NDArray[np.float64] | None = None
    y_ema: NDArray[np.float64] | None = None
    for rotation in rotations:
        x_ema = _ema_unit_vector(np.asarray(rotation)[:, 0], x_ema, alpha)
        y_ema = _ema_unit_vector(np.asarray(rotation)[:, 1], y_ema, alpha)
        basis = _orthonormal_basis(x_ema, y_ema)
        result.append(np.eye(3) if basis is None else basis)
    return result


def _ema_unit_vector(
    value: NDArray[np.floating], previous: NDArray[np.float64] | None, alpha: float
) -> NDArray[np.float64]:
    current = _unit(value)
    if current is None:
        return np.asarray(previous if previous is not None else (1.0, 0.0, 0.0))
    if previous is None:
        return current
    if float(np.dot(current, previous)) < 0.0:
        current = -current
    blended = _unit((1.0 - alpha) * previous + alpha * current)
    return previous if blended is None else blended


def _smoothed_midpoint_rotations(
    thumb: NDArray[np.float64],
    index_tip: NDArray[np.float64],
    thumb_base: NDArray[np.float64],
    index_base: NDArray[np.float64],
    wrist: NDArray[np.float64],
    alpha: float,
    *,
    fallback: Sequence[NDArray[np.float64]],
) -> list[NDArray[np.float64]]:
    raw: list[NDArray[np.float64]] = []
    previous: NDArray[np.float64] | None = None
    for frame in range(len(wrist)):
        basis = _orthonormal_basis(
            index_base[frame] - thumb_base[frame],
            (thumb_base[frame] + index_base[frame]) * 0.5 - wrist[frame],
        )
        if basis is None:
            basis = previous if previous is not None else fallback[frame]
        if previous is not None and np.dot(previous[:, 0], basis[:, 0]) < 0.0:
            basis = basis.copy()
            basis[:, 0] *= -1.0
            basis[:, 1] *= -1.0
        raw.append(basis)
        previous = basis
    return _ema_rotations(raw, alpha)


def _angular_velocity(
    previous: NDArray[np.float32], current: NDArray[np.float32], dt: float
) -> NDArray[np.float64]:
    return Rotation.from_matrix(previous.T @ current).as_rotvec() / dt


def _lerp(first: object, last: object, ratio: float) -> object:
    return (1.0 - ratio) * np.asarray(first) + ratio * np.asarray(last)


def _optional_lerp(first: float | None, last: float | None, ratio: float) -> float | None:
    if first is None or last is None:
        return None
    return float((1.0 - ratio) * first + ratio * last)


def _true_runs(values: Sequence[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(values):
        if not values[index]:
            index += 1
            continue
        end = index + 1
        while end < len(values) and values[end]:
            end += 1
        runs.append((index, end))
        index = end
    return runs


def _joined_world_runs(
    present: Sequence[bool],
    available: Sequence[bool],
    *,
    maximum_gap: int,
) -> list[tuple[int, int]]:
    """Join world-coordinate runs only across bounded VIO gaps with a visible hand."""

    if len(present) != len(available):
        raise ValueError("presence and world availability lengths must match")
    runs: list[tuple[int, int]] = []
    start: int | None = None
    last_valid: int | None = None
    for index, (has_hand, has_world) in enumerate(zip(present, available, strict=True)):
        if not has_hand:
            if start is not None and last_valid is not None:
                runs.append((start, last_valid + 1))
            start = None
            last_valid = None
            continue
        if not has_world:
            continue
        if start is None:
            start = index
        elif last_valid is not None and index - last_valid - 1 > maximum_gap:
            runs.append((start, last_valid + 1))
            start = index
        last_valid = index
    if start is not None and last_valid is not None:
        runs.append((start, last_valid + 1))
    return runs


def _fill_position_samples(
    sample_count: int,
    offsets: Sequence[int],
    values: Sequence[NDArray[np.floating]],
) -> NDArray[np.float64]:
    """Linearly fill bounded internal VIO gaps before Savitzky-Golay filtering."""

    samples = np.asarray(values, dtype=np.float64)
    if samples.shape != (len(offsets), 3) or not offsets:
        raise ValueError("position samples must contain one 3D value per offset")
    target = np.arange(sample_count, dtype=np.float64)
    source = np.asarray(offsets, dtype=np.float64)
    return np.column_stack(
        [np.interp(target, source, samples[:, axis]) for axis in range(3)]
    )


def _fill_rotation_samples(
    sample_count: int,
    offsets: Sequence[int],
    values: Sequence[NDArray[np.floating]],
) -> list[NDArray[np.float64]]:
    """SLERP bounded internal rotation gaps used only by the smoothing calculation."""

    rotations = np.asarray(values, dtype=np.float64)
    if rotations.shape != (len(offsets), 3, 3) or not offsets:
        raise ValueError("rotation samples must contain one 3x3 value per offset")
    if len(offsets) == 1:
        return [rotations[0].copy() for _ in range(sample_count)]
    interpolated = Slerp(
        np.asarray(offsets, dtype=np.float64),
        Rotation.from_matrix(rotations),
    )(np.arange(sample_count, dtype=np.float64)).as_matrix()
    return [np.asarray(value, dtype=np.float64) for value in interpolated]


def _suppress_bracketed_flicker(
    values: NDArray[np.bool_], *, maximum_run: int
) -> NDArray[np.bool_]:
    result = values.copy()
    if maximum_run < 1:
        return result
    start = 0
    while start < len(result):
        end = start + 1
        while end < len(result) and result[end] == result[start]:
            end += 1
        is_bracketed = start > 0 and end < len(result) and result[start - 1] == result[end]
        if is_bracketed and end - start <= maximum_run:
            result[start:end] = result[start - 1]
        start = end
    return result


def _count_grasp_transitions(frames: Sequence[dict[Handedness, TrackedHand]]) -> int:
    count = 0
    for side in _SIDES:
        previous: bool | None = None
        for frame in frames:
            hand = frame.get(side)
            if hand is None:
                previous = None
                continue
            if previous is not None and hand.is_grasping != previous:
                count += 1
            previous = hand.is_grasping
    return count
