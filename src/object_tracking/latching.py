"""Static-object to hand-attached-object state machine for dataset generation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hand_tracking.models import Handedness, TrackedHand
from schemas.object_tracking import ObjectPose


class ObjectPoseLatcher:
    """Propagate a static object from its first valid grasp by one hand."""

    def __init__(
        self,
        object_id: str,
        transform_object_to_world: NDArray[np.float64],
        *,
        maximum_latch_distance_m: float = float("inf"),
    ) -> None:
        self.object_id = object_id
        self._pose = np.asarray(transform_object_to_world, dtype=np.float64)
        if self._pose.shape != (4, 4):
            raise ValueError("object pose must have shape (4, 4)")
        if maximum_latch_distance_m <= 0.0:
            raise ValueError("maximum latch distance must be positive")
        self._maximum_latch_distance_m = maximum_latch_distance_m
        self._hand_to_object: NDArray[np.float64] | None = None
        self._latched_hand: Handedness | None = None

    def update(
        self,
        clip_id: str,
        frame_index: int,
        session_time_ns: int,
        hands: tuple[TrackedHand, ...],
    ) -> ObjectPose:
        active = self._active_hand(hands)
        dynamic = False
        grasped_by: str | None = None
        if active is not None:
            hand, transform_hand_to_world = active
            if self._latched_hand is None:
                self._latched_hand = hand.handedness
                self._hand_to_object = np.linalg.inv(transform_hand_to_world) @ self._pose
            if self._latched_hand is hand.handedness and self._hand_to_object is not None:
                self._pose = transform_hand_to_world @ self._hand_to_object
                dynamic = True
                grasped_by = hand.handedness.value
        elif self._latched_hand is not None:
            grasped_by = self._latched_hand.value
        return ObjectPose(
            object_id=self.object_id,
            clip_id=clip_id,
            frame_index=frame_index,
            session_time_ns=session_time_ns,
            transform_object_to_world=tuple(float(value) for value in self._pose.reshape(-1)),
            source=(
                "hand_latched"
                if dynamic
                else "hand_latched_hold"
                if self._latched_hand is not None
                else "static_triangulation"
            ),
            grasped_by=grasped_by,
            dynamic=dynamic,
        )

    def _active_hand(
        self,
        hands: tuple[TrackedHand, ...],
    ) -> tuple[TrackedHand, NDArray[np.float64]] | None:
        candidates: list[tuple[TrackedHand, NDArray[np.float64]]] = []
        for hand in hands:
            if not hand.is_grasping or hand.kinematics is None:
                continue
            if self._latched_hand is not None and hand.handedness is not self._latched_hand:
                continue
            transform = np.asarray(
                hand.kinematics.midpoint_pose_optimized_world,
                dtype=np.float64,
            )
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                continue
            if self._latched_hand is None and (
                np.linalg.norm(transform[:3, 3] - self._pose[:3, 3])
                > self._maximum_latch_distance_m
            ):
                continue
            candidates.append((hand, transform))
        return max(candidates, key=lambda item: item[0].confidence) if candidates else None
