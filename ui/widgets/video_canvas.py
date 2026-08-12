from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PyQt6.QtCore import QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget


class VideoFrameLike(Protocol):
    """Minimal frame contract shared by live preview and recording replay."""

    frame_index: int
    image_rgb: np.ndarray

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...


@dataclass(frozen=True, slots=True)
class FittedImageGeometry:
    minimum: tuple[float, float]
    maximum: tuple[float, float]
    scale: float

    @property
    def width(self) -> float:
        return self.maximum[0] - self.minimum[0]

    @property
    def height(self) -> float:
        return self.maximum[1] - self.minimum[1]


@dataclass(frozen=True, slots=True)
class VideoCanvasStatus:
    presented_frames: int
    source_frames_skipped: int
    recent_presentation_fps: float
    latest_paint_ms: float | None
    latest_frame_index: int | None


def fit_image_geometry(
    container_width: float,
    container_height: float,
    source_width: float,
    source_height: float,
) -> FittedImageGeometry:
    if min(container_width, container_height, source_width, source_height) <= 0:
        raise ValueError("image and container dimensions must be positive")
    scale = min(container_width / source_width, container_height / source_height)
    rendered_width = source_width * scale
    rendered_height = source_height * scale
    offset_x = (container_width - rendered_width) / 2
    offset_y = (container_height - rendered_height) / 2
    return FittedImageGeometry(
        minimum=(offset_x, offset_y),
        maximum=(offset_x + rendered_width, offset_y + rendered_height),
        scale=scale,
    )


class VideoCanvas(QWidget):
    """Paint one immutable RGB frame without algorithm overlays."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("videoCanvas")
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._frame: VideoFrameLike | None = None
        self._image: QImage | None = None
        self._latest_frame_key: tuple[str, int] | None = None
        self._pending_presentation = False
        self._presented_frames = 0
        self._source_frames_skipped = 0
        self._presented_at_ns: deque[int] = deque(maxlen=240)
        self._latest_paint_ms: float | None = None

    def sizeHint(self) -> QSize:
        return QSize(960, 720)

    def heightForWidth(self, width: int) -> int:
        return round(width * 3 / 4)

    def hasHeightForWidth(self) -> bool:
        return True

    def canvas_geometry(self) -> FittedImageGeometry:
        return fit_image_geometry(self.width(), self.height(), 4, 3)

    def set_frame(self, frame: VideoFrameLike | None) -> bool:
        if frame is None:
            return False
        frame_key = (_frame_stream_key(frame), int(frame.frame_index))
        if frame_key == self._latest_frame_key:
            return False
        image_rgb = frame.image_rgb
        if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("VideoCanvas requires an HxWx3 uint8 RGB frame")
        if not image_rgb.flags.c_contiguous:
            raise ValueError("VideoCanvas requires a contiguous RGB frame")
        if self._latest_frame_key is not None:
            same_stream = frame_key[0] == self._latest_frame_key[0]
            if same_stream:
                self._source_frames_skipped += max(
                    0,
                    frame_key[1] - self._latest_frame_key[1] - 1,
                )
        self._frame = frame
        self._image = QImage(
            image_rgb.data,
            int(frame.width),
            int(frame.height),
            int(image_rgb.strides[0]),
            QImage.Format.Format_RGB888,
        )
        self._latest_frame_key = frame_key
        self._pending_presentation = True
        self.update()
        return True

    def clear(self) -> None:
        self._frame = None
        self._image = None
        self._latest_frame_key = None
        self.update()

    def status(self, now_ns: int | None = None) -> VideoCanvasStatus:
        current_ns = time.perf_counter_ns() if now_ns is None else now_ns
        cutoff_ns = current_ns - 2_000_000_000
        recent = tuple(value for value in self._presented_at_ns if value >= cutoff_ns)
        fps = 0.0
        if len(recent) > 1 and recent[-1] > recent[0]:
            fps = (len(recent) - 1) * 1_000_000_000 / (recent[-1] - recent[0])
        return VideoCanvasStatus(
            presented_frames=self._presented_frames,
            source_frames_skipped=self._source_frames_skipped,
            recent_presentation_fps=round(fps, 3),
            latest_paint_ms=(
                round(self._latest_paint_ms, 3)
                if self._latest_paint_ms is not None
                else None
            ),
            latest_frame_index=(
                self._latest_frame_key[1] if self._latest_frame_key is not None else None
            ),
        )

    def paintEvent(self, _event: object) -> None:
        started_ns = time.perf_counter_ns()
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#f5f7fb"))

        canvas = self.canvas_geometry()
        canvas_rect = _rect(canvas)
        painter.setPen(QPen(QColor("#d8dee9"), 1))
        painter.setBrush(QColor("#090d12"))
        painter.drawRoundedRect(canvas_rect, 5, 5)

        if self._image is None or self._frame is None:
            painter.setPen(QColor("#8a95a5"))
            painter.drawText(canvas_rect, Qt.AlignmentFlag.AlignCenter, "等待视频帧")
        else:
            image_geometry = _fit_inside(
                canvas,
                int(self._frame.width),
                int(self._frame.height),
            )
            painter.drawImage(_rect(image_geometry), self._image)
        painter.end()

        finished_ns = time.perf_counter_ns()
        self._latest_paint_ms = (finished_ns - started_ns) / 1_000_000
        if self._pending_presentation:
            self._pending_presentation = False
            self._presented_frames += 1
            self._presented_at_ns.append(finished_ns)


def _frame_stream_key(frame: VideoFrameLike) -> str:
    for name in ("recording_id", "connection_session_id", "stream_id"):
        value = getattr(frame, name, None)
        if isinstance(value, str) and value:
            return value
    return "video"


def _fit_inside(
    container: FittedImageGeometry,
    source_width: float,
    source_height: float,
) -> FittedImageGeometry:
    fitted = fit_image_geometry(
        container.width,
        container.height,
        source_width,
        source_height,
    )
    return FittedImageGeometry(
        minimum=(
            container.minimum[0] + fitted.minimum[0],
            container.minimum[1] + fitted.minimum[1],
        ),
        maximum=(
            container.minimum[0] + fitted.maximum[0],
            container.minimum[1] + fitted.maximum[1],
        ),
        scale=fitted.scale,
    )


def _rect(geometry: FittedImageGeometry) -> QRectF:
    return QRectF(
        geometry.minimum[0],
        geometry.minimum[1],
        geometry.width,
        geometry.height,
    )
