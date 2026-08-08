"""Deterministic phase segmentation adapted from HumanEgo AriaPhases."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

from schemas.phase import (
    MotionPhase,
    ObjectCentricWindow,
    PhaseAnalysisResult,
    PhaseFrame,
    PhaseSegment,
)

from .models import PhaseAnalysisConfig, PhaseInputFrame


class PhaseAnalysisService:
    """Classify head and hand motion without assuming a fixed frame rate."""

    def __init__(self, config: PhaseAnalysisConfig | None = None) -> None:
        self.config = config or PhaseAnalysisConfig()

    def analyze(
        self,
        processing_run_id: str,
        frames: Iterable[PhaseInputFrame],
    ) -> PhaseAnalysisResult:
        grouped: dict[str, list[PhaseInputFrame]] = defaultdict(list)
        for frame in frames:
            grouped[frame.clip_id].append(frame)
        phase_frames: list[PhaseFrame] = []
        segments: list[PhaseSegment] = []
        windows: list[ObjectCentricWindow] = []
        for clip_id in sorted(grouped):
            ordered = sorted(grouped[clip_id], key=lambda item: item.frame_index)
            self._validate_ordered(ordered)
            classified = self._classify_clip(ordered)
            phase_frames.extend(classified)
            clip_segments = _segments(classified)
            segments.extend(clip_segments)
            windows.extend(self._object_windows(classified, clip_segments))
        return PhaseAnalysisResult(
            processing_run_id=processing_run_id,
            frames=tuple(phase_frames),
            segments=tuple(segments),
            object_centric_windows=tuple(windows),
        )

    @staticmethod
    def _validate_ordered(frames: list[PhaseInputFrame]) -> None:
        if any(
            current.frame_index <= previous.frame_index
            or current.session_time_ns <= previous.session_time_ns
            for previous, current in zip(frames, frames[1:], strict=False)
        ):
            raise ValueError("phase input frames must be strictly ordered per clip")

    def _classify_clip(self, frames: list[PhaseInputFrame]) -> list[PhaseFrame]:
        if not frames:
            return []
        head_linear = _linear_speeds(frames)
        head_angular = _angular_speeds(frames)
        labels = [
            self._classify(speed, angular, frame.hand_linear_speed_m_s, frame.grasping)
            for frame, speed, angular in zip(frames, head_linear, head_angular, strict=True)
        ]
        labels = _suppress_short_runs(labels, self.config.minimum_segment_frames)
        labels = self._mark_finished(labels)
        result: list[PhaseFrame] = []
        for frame, phase, linear, angular in zip(
            frames, labels, head_linear, head_angular, strict=True
        ):
            confidence = _confidence(
                phase, linear, angular, frame.hand_linear_speed_m_s, self.config
            )
            result.append(
                PhaseFrame(
                    clip_id=frame.clip_id,
                    frame_index=frame.frame_index,
                    session_time_ns=frame.session_time_ns,
                    phase=phase,
                    confidence=confidence,
                    head_linear_speed_m_s=linear,
                    head_angular_speed_rad_s=angular,
                    hand_linear_speed_m_s=frame.hand_linear_speed_m_s,
                    grasping=frame.grasping,
                )
            )
        return result

    def _classify(
        self,
        head_linear_speed: float,
        head_angular_speed: float,
        hand_speed: float,
        grasping: bool,
    ) -> MotionPhase:
        if grasping:
            return MotionPhase.MANIPULATION
        if head_angular_speed >= self.config.rotate_angular_speed_rad_s:
            return MotionPhase.ROTATE
        if head_linear_speed >= self.config.forward_linear_speed_m_s:
            return MotionPhase.FORWARD
        if hand_speed >= self.config.manipulation_hand_speed_m_s:
            return MotionPhase.MANIPULATION
        if head_linear_speed <= self.config.stop_linear_speed_m_s:
            return MotionPhase.STOP
        return MotionPhase.TRANSITION

    def _mark_finished(self, labels: list[MotionPhase]) -> list[MotionPhase]:
        if self.config.finished_trailing_frames == 0:
            return labels
        latest = max(
            (index for index, label in enumerate(labels) if label is MotionPhase.MANIPULATION),
            default=-1,
        )
        if latest < 0:
            return labels
        result = list(labels)
        start = latest + 1
        end = min(len(result), start + self.config.finished_trailing_frames)
        for index in range(start, end):
            if result[index] is MotionPhase.STOP:
                result[index] = MotionPhase.FINISHED
        return result

    def _object_windows(
        self,
        frames: list[PhaseFrame],
        segments: list[PhaseSegment],
    ) -> list[ObjectCentricWindow]:
        if not frames:
            return []
        index_by_frame = {frame.frame_index: index for index, frame in enumerate(frames)}
        windows: list[ObjectCentricWindow] = []
        for segment in segments:
            if segment.phase is not MotionPhase.MANIPULATION:
                continue
            manipulation_start = index_by_frame[segment.start_frame_index]
            start = max(0, manipulation_start - self.config.precontact_window_frames)
            end = index_by_frame[segment.end_frame_index_exclusive - 1] + 1
            if end - start < self.config.minimum_object_window_frames:
                continue
            reference = max(start, manipulation_start - 1)
            windows.append(
                ObjectCentricWindow(
                    clip_id=segment.clip_id,
                    start_frame_index=frames[start].frame_index,
                    reference_frame_index=frames[reference].frame_index,
                    end_frame_index_exclusive=frames[end - 1].frame_index + 1,
                    start_session_time_ns=frames[start].session_time_ns,
                    end_session_time_ns=frames[end - 1].session_time_ns,
                    evidence=(
                        "HumanEgo-style pre-contact window preceding a manipulation segment",
                        f"manipulation_start_frame={segment.start_frame_index}",
                    ),
                )
            )
        return windows


def _linear_speeds(frames: list[PhaseInputFrame]) -> list[float]:
    speeds = [0.0] * len(frames)
    for index in range(1, len(frames)):
        previous, current = frames[index - 1], frames[index]
        elapsed = (current.session_time_ns - previous.session_time_ns) / 1_000_000_000
        if elapsed <= 0.0:
            continue
        distance = math.dist(previous.head_position_m, current.head_position_m)
        speeds[index] = distance / elapsed
    if len(speeds) > 1:
        speeds[0] = speeds[1]
    return speeds


def _angular_speeds(frames: list[PhaseInputFrame]) -> list[float]:
    speeds = [0.0] * len(frames)
    for index in range(1, len(frames)):
        previous, current = frames[index - 1], frames[index]
        elapsed = (current.session_time_ns - previous.session_time_ns) / 1_000_000_000
        if elapsed <= 0.0:
            continue
        dot = abs(
            sum(
                a * b
                for a, b in zip(
                    previous.head_quaternion_wxyz, current.head_quaternion_wxyz, strict=True
                )
            )
        )
        dot = min(1.0, max(-1.0, dot))
        speeds[index] = 2.0 * math.acos(dot) / elapsed
    if len(speeds) > 1:
        speeds[0] = speeds[1]
    return speeds


def _suppress_short_runs(labels: list[MotionPhase], minimum: int) -> list[MotionPhase]:
    if minimum <= 1 or len(labels) < 2:
        return labels
    result = list(labels)
    changed = True
    while changed:
        changed = False
        runs = _runs(result)
        for start, end, phase in runs:
            if end - start >= minimum:
                continue
            before = result[start - 1] if start > 0 else None
            after = result[end] if end < len(result) else None
            replacement = before if before is not None and before is after else (before or after)
            if replacement is None or replacement is phase:
                continue
            result[start:end] = [replacement] * (end - start)
            changed = True
            break
    return result


def _runs(labels: list[MotionPhase]) -> list[tuple[int, int, MotionPhase]]:
    if not labels:
        return []
    runs: list[tuple[int, int, MotionPhase]] = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] is not labels[start]:
            runs.append((start, index, labels[start]))
            start = index
    return runs


def _segments(frames: list[PhaseFrame]) -> list[PhaseSegment]:
    if not frames:
        return []
    result: list[PhaseSegment] = []
    for start, end, phase in _runs([frame.phase for frame in frames]):
        values = frames[start:end]
        result.append(
            PhaseSegment(
                clip_id=values[0].clip_id,
                start_frame_index=values[0].frame_index,
                end_frame_index_exclusive=values[-1].frame_index + 1,
                start_session_time_ns=values[0].session_time_ns,
                end_session_time_ns=values[-1].session_time_ns,
                phase=phase,
                confidence=sum(frame.confidence for frame in values) / len(values),
            )
        )
    return result


def _confidence(
    phase: MotionPhase,
    linear: float,
    angular: float,
    hand: float,
    config: PhaseAnalysisConfig,
) -> float:
    if phase is MotionPhase.MANIPULATION:
        return min(1.0, max(0.5, hand / max(config.manipulation_hand_speed_m_s, 1e-6)))
    if phase is MotionPhase.FORWARD:
        return min(1.0, linear / max(config.forward_linear_speed_m_s, 1e-6))
    if phase is MotionPhase.ROTATE:
        return min(1.0, angular / max(config.rotate_angular_speed_rad_s, 1e-6))
    if phase in {MotionPhase.STOP, MotionPhase.FINISHED}:
        return min(1.0, max(0.0, 1.0 - linear / max(config.forward_linear_speed_m_s, 1e-6)))
    return 0.5
