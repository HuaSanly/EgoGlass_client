from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from av import VideoFrame

from perception.sensor_preprocessing import (
    AlignmentStatus,
    CaptureSessionReader,
    derive_recorded_clock_mapping,
    frame_presentation_observation,
)

from .results import ProcessingResultStore

HAND_CONNECTIONS = (
    (5, 6), (6, 7), (7, 0), (5, 8), (8, 9), (9, 10), (10, 1),
    (5, 11), (11, 12), (12, 13), (13, 2), (5, 14), (14, 15),
    (15, 16), (16, 3), (5, 17), (17, 18), (18, 19), (19, 4),
    (6, 8), (8, 11), (11, 14), (14, 17),
)


@dataclass(frozen=True, slots=True)
class ExportSummary:
    path: Path
    frame_count: int

    @property
    def detail(self) -> str:
        return f"已导出 {self.frame_count} 帧到 {self.path.name}"


class _H264Writer:
    _TIME_BASE = Fraction(1, 90_000)

    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        self.container = av.open(
            str(path), mode="w", format="mp4", options={"movflags": "+faststart"}
        )
        self.stream = self.container.add_stream(
            "libx264",
            rate=Fraction(str(fps)).limit_denominator(1001),
            options={"crf": "18", "preset": "veryfast"},
        )
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.stream.time_base = self._TIME_BASE
        self.stream.codec_context.time_base = self._TIME_BASE
        self.origin_ns: int | None = None

    def write(self, image_bgr: np.ndarray, session_time_ns: int) -> None:
        if self.origin_ns is None:
            self.origin_ns = session_time_ns
        relative_ns = session_time_ns - self.origin_ns
        frame = VideoFrame.from_ndarray(image_bgr, format="bgr24")
        frame.pts = round(relative_ns * self._TIME_BASE.denominator / 1_000_000_000)
        frame.time_base = self._TIME_BASE
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        try:
            for packet in self.stream.encode(None):
                self.container.mux(packet)
        finally:
            self.container.close()


def export_annotated_clip(
    session_directory: Path,
    run_directory: Path,
    clip_id: str,
    *,
    hold_previous_frames: int,
) -> ExportSummary:
    reader = CaptureSessionReader.open(session_directory)
    clip = next((item for item in reader.session.clips if item.clip_id == clip_id), None)
    if clip is None:
        raise KeyError(f"unknown complete clip {clip_id!r}")
    references = tuple(reader.iter_frames(clip_id))
    mapper = None
    if any(item.stored_alignment.status is AlignmentStatus.PENDING for item in references):
        all_frames = tuple(
            frame
            for current_clip in reader.session.clips
            for frame in reader.iter_frames(current_clip.clip_id)
        )
        mapper = derive_recorded_clock_mapping(
            reader.session.session_id,
            all_frames,
            tuple(reader.iter_imu_samples()),
        ).mapper
    store = ProcessingResultStore(run_directory / "results.sqlite", read_only=True)
    exports = run_directory / "exports"
    exports.mkdir(exist_ok=True)
    output = exports / f"annotated-{clip_id}.mp4"
    temporary = output.with_suffix(".mp4.tmp")
    if temporary.exists():
        temporary.unlink()
    writer = _H264Writer(temporary, clip.nominal_fps, clip.width, clip.height)
    frame_count = 0
    try:
        with av.open(str(clip.media_path), mode="r") as container:
            decoded = container.decode(container.streams.video[0])
            for frame, reference in zip(decoded, references, strict=True):
                session_time_ns = reference.stored_alignment.session_time_ns
                if session_time_ns is None:
                    assert mapper is not None
                    session_time_ns = mapper.map(
                        frame_presentation_observation(reference)
                    ).session_time_ns
                if session_time_ns is None:
                    raise ValueError("frame has no session-time mapping")
                result = store.result_for_frame(
                    clip_id,
                    reference.frame_index,
                    session_time_ns,
                    hold_previous_frames=hold_previous_frames,
                )
                image_bgr = frame.to_ndarray(format="bgr24")
                _draw_result(image_bgr, result)
                writer.write(image_bgr, session_time_ns)
                frame_count += 1
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    writer.close()
    temporary.replace(output)
    return ExportSummary(output, frame_count)


def _draw_result(image_bgr: np.ndarray, result: dict[str, object] | None) -> None:
    if result is None:
        return
    hands = result.get("hands")
    if not isinstance(hands, list):
        return
    for hand in hands:
        if not isinstance(hand, dict):
            continue
        color = (133, 160, 22) if hand.get("handedness") == "left" else (11, 158, 245)
        points = _points(hand.get("source_keypoints_2d_px"))
        if len(points) == 21:
            for first, second in HAND_CONNECTIONS:
                cv2.line(image_bgr, points[first], points[second], color, 2, cv2.LINE_AA)
            for point in points:
                cv2.circle(image_bgr, point, 3, color, -1, cv2.LINE_AA)
        bbox = _bbox(hand.get("source_bbox_xyxy_px"))
        if bbox is not None:
            cv2.rectangle(image_bgr, bbox[:2], bbox[2:], color, 2, cv2.LINE_AA)


def _points(value: object) -> list[tuple[int, int]]:
    if not isinstance(value, list):
        return []
    points: list[tuple[int, int]] = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            return []
        if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in point):
            return []
        points.append((round(float(point[0])), round(float(point[1]))))
    return points


def _bbox(value: object) -> tuple[int, int, int, int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        return None
    x1, y1, x2, y2 = (round(float(item)) for item in value)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None
