from __future__ import annotations

from dataclasses import dataclass

from schemas import QualityGate, QualityIssue


@dataclass(frozen=True, slots=True)
class EpisodeInterval:
    clip_id: str
    start_frame_index: int
    end_frame_index_exclusive: int

    def __post_init__(self) -> None:
        if not self.clip_id or self.start_frame_index < 0:
            raise ValueError("episode interval is invalid")
        if self.end_frame_index_exclusive <= self.start_frame_index:
            raise ValueError("episode interval must be non-empty")

    @property
    def frame_count(self) -> int:
        return self.end_frame_index_exclusive - self.start_frame_index


def split_valid_intervals(
    clip_id: str,
    frame_indices: list[int] | tuple[int, ...],
    issues: tuple[QualityIssue, ...] = (),
    *,
    minimum_frames: int = 10,
) -> tuple[EpisodeInterval, ...]:
    """Split a clip at invalid frames without touching the source media."""

    if minimum_frames < 1:
        raise ValueError("minimum_frames must be positive")
    ordered = tuple(sorted(set(frame_indices)))
    if not ordered:
        return ()
    invalid: set[int] = set()
    for issue in issues:
        if issue.restored or (issue.gate is QualityGate.SOFT and issue.restored):
            continue
        invalid.update(
            index
            for index in ordered
            if issue.clip_id == clip_id
            and issue.start_frame_index <= index < issue.end_frame_index_exclusive
        )
    ranges: list[EpisodeInterval] = []
    start: int | None = None
    previous: int | None = None
    for index in ordered:
        if index in invalid or (previous is not None and index != previous + 1):
            if (
                start is not None
                and previous is not None
                and previous - start + 1 >= minimum_frames
            ):
                ranges.append(EpisodeInterval(clip_id, start, previous + 1))
            start = None
        if index not in invalid:
            if start is None:
                start = index
            previous = index
    if start is not None and previous is not None and previous - start + 1 >= minimum_frames:
        ranges.append(EpisodeInterval(clip_id, start, previous + 1))
    return tuple(ranges)
