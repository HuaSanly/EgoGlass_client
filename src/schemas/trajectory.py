"""Public trajectory types returned by an offline VIO run."""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VioPose:
    """One Basalt pose in the IMU/body frame at a session timestamp."""

    timestamp_ns: int
    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("pose timestamp must be non-negative")
        if len(self.position_m) != 3 or len(self.quaternion_wxyz) != 4:
            raise ValueError("pose vectors have invalid dimensions")
        if not all(math.isfinite(value) for value in (*self.position_m, *self.quaternion_wxyz)):
            raise ValueError("pose values must be finite")
        if math.sqrt(sum(value * value for value in self.quaternion_wxyz)) < 1e-9:
            raise ValueError("pose quaternion cannot be zero")


@dataclass(frozen=True, slots=True)
class VioTrajectory:
    """Ordered poses produced by one VIO invocation."""

    poses: tuple[VioPose, ...]
    source: str = "basalt"

    def __post_init__(self) -> None:
        if any(
            current.timestamp_ns <= previous.timestamp_ns
            for previous, current in zip(self.poses, self.poses[1:], strict=False)
        ):
            raise ValueError("trajectory timestamps must strictly increase")

    def pose_at(self, timestamp_ns: int, *, max_gap_ns: int | None = None) -> VioPose | None:
        """Return the temporally nearest pose for one playback timestamp."""

        if not self.poses:
            return None
        timestamps = [pose.timestamp_ns for pose in self.poses]
        index = bisect_left(timestamps, timestamp_ns)
        if index == 0:
            candidate = self.poses[0]
        elif index == len(self.poses):
            candidate = self.poses[-1]
        else:
            before = self.poses[index - 1]
            after = self.poses[index]
            candidate = (
                before
                if timestamp_ns - before.timestamp_ns <= after.timestamp_ns - timestamp_ns
                else after
            )
        if max_gap_ns is not None and abs(candidate.timestamp_ns - timestamp_ns) > max_gap_ns:
            return None
        return candidate
