from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import dearpygui.dearpygui as dpg
import numpy as np

from ingest_gateway.live_frames import LiveFrame

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


@dataclass(frozen=True, slots=True)
class VideoSurfaceStatus:
    uploaded_frames: int
    source_frames_skipped: int
    recent_upload_fps: float
    latest_upload_ms: float | None
    latest_frame_index: int | None


class VideoSurface:
    """A raw RGB texture and frame-aligned perception overlay on one drawlist."""

    def __init__(
        self,
        *,
        parent: int | str,
        width: int = 960,
        height: int = 540,
        source_width: int = 1280,
        source_height: int = 720,
    ) -> None:
        self.width = width
        self.height = height
        self._source_width = source_width
        self._source_height = source_height
        self._texture_registry_tag = "main-video-texture-registry"
        self._texture_generation = 0
        self._texture_tags = (self._next_texture_tag(), self._next_texture_tag())
        self._front_texture_index = 0
        self._texture_tag = self._texture_tags[self._front_texture_index]
        self._image_tag = "main-video-image"
        self._overlay_tag = "main-video-overlay"
        self._drawlist_tag = "main-video-drawlist"
        self._texture_buffers = (
            np.zeros((source_height, source_width, 3), dtype=np.float32),
            np.zeros((source_height, source_width, 3), dtype=np.float32),
        )
        self._latest_frame_key: tuple[str, str, int] | None = None
        self._uploaded_frames = 0
        self._uploaded_at_ns: deque[int] = deque(maxlen=240)
        self._source_frames_skipped = 0
        self._latest_upload_ms: float | None = None
        with dpg.texture_registry(show=False, tag=self._texture_registry_tag):
            for texture_tag, texture_buffer in zip(
                self._texture_tags,
                self._texture_buffers,
                strict=True,
            ):
                dpg.add_raw_texture(
                    width=source_width,
                    height=source_height,
                    default_value=texture_buffer.ravel(),
                    format=dpg.mvFormat_Float_rgb,
                    tag=texture_tag,
                )
        with dpg.drawlist(
            width=width,
            height=height,
            parent=parent,
            tag=self._drawlist_tag,
        ):
            dpg.draw_rectangle(
                (0, 0),
                (width, height),
                fill=(7, 9, 10, 255),
                color=(48, 56, 59, 255),
            )
            dpg.draw_image(
                self._texture_tag,
                (0, 0),
                (width, height),
                tag=self._image_tag,
            )
            dpg.add_draw_layer(tag=self._overlay_tag)

    def update_frame(self, frame: LiveFrame | None) -> bool:
        if frame is None:
            return False
        frame_key = (frame.session_id, frame.connection_session_id, frame.frame_index)
        if frame_key == self._latest_frame_key:
            return False
        if frame.width != self._source_width or frame.height != self._source_height:
            self._replace_texture(frame.width, frame.height)
        started_at_ns = time.perf_counter_ns()
        back_texture_index = 1 - self._front_texture_index
        back_texture_buffer = self._texture_buffers[back_texture_index]
        np.multiply(
            frame.image_rgb,
            np.float32(1.0 / 255.0),
            out=back_texture_buffer,
            casting="unsafe",
        )
        self._front_texture_index = back_texture_index
        self._texture_tag = self._texture_tags[self._front_texture_index]
        dpg.configure_item(self._image_tag, texture_tag=self._texture_tag)
        finished_at_ns = time.perf_counter_ns()
        if self._latest_frame_key is not None and frame_key[:2] == self._latest_frame_key[:2]:
            self._source_frames_skipped += max(
                0,
                frame.frame_index - self._latest_frame_key[2] - 1,
            )
        self._latest_frame_key = frame_key
        self._uploaded_frames += 1
        self._uploaded_at_ns.append(finished_at_ns)
        self._latest_upload_ms = (finished_at_ns - started_at_ns) / 1_000_000
        return True

    def update_overlay(self, result: dict[str, object] | None) -> None:
        dpg.delete_item(self._overlay_tag, children_only=True)
        if not result:
            return
        source_width = _positive_number(result.get("source_image_width_px"))
        source_height = _positive_number(result.get("source_image_height_px"))
        hands = result.get("hands")
        if source_width is None or source_height is None or not isinstance(hands, list):
            return
        scale = min(self.width / source_width, self.height / source_height)
        offset_x = (self.width - source_width * scale) / 2
        offset_y = (self.height - source_height * scale) / 2
        for hand in hands:
            if not isinstance(hand, dict):
                continue
            side = hand.get("handedness")
            color = (76, 205, 171, 255) if side == "left" else (245, 183, 83, 255)
            points = _points(hand.get("source_keypoints_2d_px"))
            if len(points) == 21:
                scaled = [
                    (offset_x + point[0] * scale, offset_y + point[1] * scale)
                    for point in points
                ]
                for first, second in HAND_CONNECTIONS:
                    dpg.draw_line(
                        scaled[first],
                        scaled[second],
                        color=color,
                        thickness=2.0,
                        parent=self._overlay_tag,
                    )
                for point in scaled:
                    dpg.draw_circle(
                        point,
                        3.2,
                        color=(235, 244, 241, 255),
                        fill=color,
                        parent=self._overlay_tag,
                    )
            bbox = _bbox(hand.get("source_bbox_xyxy_px"))
            if bbox is not None:
                dpg.draw_rectangle(
                    (offset_x + bbox[0] * scale, offset_y + bbox[1] * scale),
                    (offset_x + bbox[2] * scale, offset_y + bbox[3] * scale),
                    color=color,
                    thickness=2.0,
                    parent=self._overlay_tag,
                )

    def status(self) -> VideoSurfaceStatus:
        recent_cutoff_ns = time.perf_counter_ns() - 2_000_000_000
        recent = tuple(value for value in self._uploaded_at_ns if value >= recent_cutoff_ns)
        recent_upload_fps = 0.0
        if len(recent) > 1 and recent[-1] > recent[0]:
            recent_upload_fps = (
                (len(recent) - 1) * 1_000_000_000 / (recent[-1] - recent[0])
            )
        return VideoSurfaceStatus(
            uploaded_frames=self._uploaded_frames,
            source_frames_skipped=self._source_frames_skipped,
            recent_upload_fps=round(recent_upload_fps, 3),
            latest_upload_ms=(
                round(self._latest_upload_ms, 3)
                if self._latest_upload_ms is not None
                else None
            ),
            latest_frame_index=(
                self._latest_frame_key[2] if self._latest_frame_key is not None else None
            ),
        )

    def _replace_texture(self, width: int, height: int) -> None:
        old_texture_tags = self._texture_tags
        self._source_width = width
        self._source_height = height
        self._texture_buffers = (
            np.zeros((height, width, 3), dtype=np.float32),
            np.zeros((height, width, 3), dtype=np.float32),
        )
        self._texture_tags = (self._next_texture_tag(), self._next_texture_tag())
        self._front_texture_index = 0
        self._texture_tag = self._texture_tags[self._front_texture_index]
        for texture_tag, texture_buffer in zip(
            self._texture_tags,
            self._texture_buffers,
            strict=True,
        ):
            dpg.add_raw_texture(
                width=width,
                height=height,
                default_value=texture_buffer.ravel(),
                format=dpg.mvFormat_Float_rgb,
                tag=texture_tag,
                parent=self._texture_registry_tag,
            )
        dpg.configure_item(self._image_tag, texture_tag=self._texture_tag)
        for old_texture_tag in old_texture_tags:
            dpg.delete_item(old_texture_tag)

    def _next_texture_tag(self) -> str:
        tag = f"main-video-texture-{self._texture_generation}"
        self._texture_generation += 1
        return tag


def _positive_number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None


def _points(value: object) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []
    result: list[tuple[float, float]] = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            return []
        x = _positive_number(point[0])
        y = _positive_number(point[1])
        if x is None or y is None:
            if point == [0, 0]:
                result.append((0.0, 0.0))
                continue
            return []
        result.append((x, y))
    return result


def _bbox(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        return None
    x1, y1, x2, y2 = (float(item) for item in value)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2
