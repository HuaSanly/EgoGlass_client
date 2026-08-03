from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from PyQt6.QtCore import QLineF, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from ingest_gateway.live_frames import LiveFrame
from ui.replay.player import PlaybackFrame

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
    latest_overlay_frame_index: int | None
    overlay_frame_age: int | None
    overlay_visible: bool


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
    """Paint one immutable RGB frame and a recent same-stream hand overlay."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        maximum_overlay_age_frames: int = 18,
    ) -> None:
        super().__init__(parent)
        if maximum_overlay_age_frames < 0:
            raise ValueError("maximum overlay age must be non-negative")
        self.setObjectName("videoCanvas")
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._maximum_overlay_age_frames = maximum_overlay_age_frames
        self._frame: LiveFrame | PlaybackFrame | None = None
        self._image: QImage | None = None
        self._overlay: dict[str, object] | None = None
        self._comparison_overlay: dict[str, object] | None = None
        self._latest_frame_key: tuple[str, str, int] | None = None
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

    def set_frame(self, frame: LiveFrame | PlaybackFrame | None) -> bool:
        if frame is None:
            return False
        frame_key = (frame.session_id, _frame_stream_id(frame), frame.frame_index)
        if frame_key == self._latest_frame_key:
            return False
        image_rgb = frame.image_rgb
        if image_rgb.dtype.name != "uint8" or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("VideoCanvas requires an HxWx3 uint8 RGB frame")
        if not image_rgb.flags.c_contiguous:
            raise ValueError("VideoCanvas requires a contiguous RGB frame")
        stream_changed = (
            self._latest_frame_key is not None
            and frame_key[:2] != self._latest_frame_key[:2]
        )
        moved_backwards = (
            self._latest_frame_key is not None
            and frame_key[:2] == self._latest_frame_key[:2]
            and frame_key[2] < self._latest_frame_key[2]
        )
        if stream_changed or moved_backwards:
            self._overlay = None
            self._comparison_overlay = None
        if self._latest_frame_key is not None and frame_key[:2] == self._latest_frame_key[:2]:
            self._source_frames_skipped += max(
                0,
                frame.frame_index - self._latest_frame_key[2] - 1,
            )
        self._frame = frame
        self._image = QImage(
            image_rgb.data,
            frame.width,
            frame.height,
            int(image_rgb.strides[0]),
            QImage.Format.Format_RGB888,
        )
        self._latest_frame_key = frame_key
        self._pending_presentation = True
        self.update()
        return True

    def set_overlay(self, result: dict[str, object] | None) -> bool:
        return self._set_result_overlay("primary", result)

    def set_comparison_overlay(self, result: dict[str, object] | None) -> bool:
        return self._set_result_overlay("comparison", result)

    def _set_result_overlay(
        self,
        layer: str,
        result: dict[str, object] | None,
    ) -> bool:
        attribute = "_overlay" if layer == "primary" else "_comparison_overlay"
        current = getattr(self, attribute)
        if result is None:
            changed = current is not None
            setattr(self, attribute, None)
            if changed:
                self.update()
            return changed
        result_key = _result_key(result)
        if result_key is None:
            return False
        if (
            self._latest_frame_key is not None
            and result_key[:2] != self._latest_frame_key[:2]
        ):
            return False
        current_key = _result_key(current) if current is not None else None
        if (
            current_key is not None
            and current_key[:2] == result_key[:2]
            and result_key[2] <= current_key[2]
        ):
            return False
        setattr(self, attribute, result)
        self.update()
        return True

    def clear(self) -> None:
        self._frame = None
        self._image = None
        self._overlay = None
        self._comparison_overlay = None
        self._latest_frame_key = None
        self.update()

    def status(self, now_ns: int | None = None) -> VideoCanvasStatus:
        current_ns = time.perf_counter_ns() if now_ns is None else now_ns
        cutoff_ns = current_ns - 2_000_000_000
        recent = tuple(value for value in self._presented_at_ns if value >= cutoff_ns)
        fps = 0.0
        if len(recent) > 1 and recent[-1] > recent[0]:
            fps = (len(recent) - 1) * 1_000_000_000 / (recent[-1] - recent[0])
        active_overlay = self._active_overlay()
        overlay_key = _result_key(self._overlay) if self._overlay is not None else None
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
                self._latest_frame_key[2] if self._latest_frame_key is not None else None
            ),
            latest_overlay_frame_index=overlay_key[2] if overlay_key is not None else None,
            overlay_frame_age=active_overlay[1] if active_overlay is not None else None,
            overlay_visible=(
                active_overlay is not None and _has_drawable_hands(active_overlay[0])
            ),
        )

    def paintEvent(self, _event: object) -> None:
        started_ns = time.perf_counter_ns()
        painter = QPainter(self)
        painter.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        painter.fillRect(self.rect(), QColor("#eef2f7"))

        canvas = self.canvas_geometry()
        canvas_rect = _rect(canvas)
        painter.setPen(QPen(QColor("#d7dde7"), 1))
        painter.setBrush(QColor("#080b10"))
        painter.drawRoundedRect(canvas_rect, 5, 5)

        if self._image is None or self._frame is None:
            painter.setPen(QColor("#86909c"))
            painter.drawText(canvas_rect, Qt.AlignmentFlag.AlignCenter, "等待视频帧")
        else:
            image_geometry = _fit_inside(canvas, self._frame.width, self._frame.height)
            painter.drawImage(_rect(image_geometry), self._image)
            self._paint_overlays(painter, canvas)
        painter.end()

        finished_ns = time.perf_counter_ns()
        self._latest_paint_ms = (finished_ns - started_ns) / 1_000_000
        if self._pending_presentation:
            self._pending_presentation = False
            self._presented_frames += 1
            self._presented_at_ns.append(finished_ns)

    def _paint_overlays(self, painter: QPainter, canvas: FittedImageGeometry) -> None:
        primary = self._active_overlay(self._overlay)
        comparison = self._active_overlay(self._comparison_overlay)
        if comparison is None:
            if primary is not None:
                self._paint_result_overlay(painter, canvas, primary[0])
            return
        midpoint = canvas.minimum[0] + canvas.width / 2
        if primary is not None:
            painter.save()
            painter.setClipRect(
                QRectF(canvas.minimum[0], canvas.minimum[1], canvas.width / 2, canvas.height)
            )
            self._paint_result_overlay(painter, canvas, primary[0])
            painter.restore()
        painter.save()
        painter.setClipRect(
            QRectF(midpoint, canvas.minimum[1], canvas.width / 2, canvas.height)
        )
        self._paint_result_overlay(painter, canvas, comparison[0])
        painter.restore()
        painter.setPen(QPen(QColor("#f8fafc"), 1))
        painter.drawLine(QLineF(midpoint, canvas.minimum[1], midpoint, canvas.maximum[1]))

    def _paint_result_overlay(
        self,
        painter: QPainter,
        canvas: FittedImageGeometry,
        result: dict[str, object],
    ) -> None:
        source_width = _positive_number(result.get("source_image_width_px"))
        source_height = _positive_number(result.get("source_image_height_px"))
        hands = result.get("hands")
        if source_width is None or source_height is None or not isinstance(hands, list):
            return
        geometry = _fit_inside(canvas, source_width, source_height)
        offset_x, offset_y = geometry.minimum
        for hand in hands:
            if not isinstance(hand, dict):
                continue
            color = QColor("#16a085") if hand.get("handedness") == "left" else QColor("#f59e0b")
            points = _points(hand.get("source_keypoints_2d_px"))
            if len(points) == 21:
                scaled = [
                    (offset_x + point[0] * geometry.scale, offset_y + point[1] * geometry.scale)
                    for point in points
                ]
                painter.setPen(QPen(color, 2.2))
                for first, second in HAND_CONNECTIONS:
                    painter.drawLine(QLineF(*scaled[first], *scaled[second]))
                painter.setPen(QPen(QColor("#f7fafc"), 1))
                painter.setBrush(color)
                for point in scaled:
                    painter.drawEllipse(QRectF(point[0] - 3, point[1] - 3, 6, 6))
            bbox = _bbox(hand.get("source_bbox_xyxy_px"))
            if bbox is not None:
                painter.setPen(QPen(color, 2.2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(
                    QRectF(
                        offset_x + bbox[0] * geometry.scale,
                        offset_y + bbox[1] * geometry.scale,
                        (bbox[2] - bbox[0]) * geometry.scale,
                        (bbox[3] - bbox[1]) * geometry.scale,
                    ),
                    4,
                    4,
                )

    def _active_overlay(
        self,
        result: dict[str, object] | None = None,
    ) -> tuple[dict[str, object], int] | None:
        if result is None:
            result = self._overlay
        result_key = _result_key(result) if result is not None else None
        frame_key = self._latest_frame_key
        if result is None or result_key is None or frame_key is None:
            return None
        if result_key[:2] != frame_key[:2]:
            return None
        frame_age = frame_key[2] - result_key[2]
        if frame_age < 0 or frame_age > self._maximum_overlay_age_frames:
            return None
        return result, frame_age


def _fit_inside(
    container: FittedImageGeometry,
    source_width: float,
    source_height: float,
) -> FittedImageGeometry:
    local = fit_image_geometry(container.width, container.height, source_width, source_height)
    return FittedImageGeometry(
        minimum=(
            container.minimum[0] + local.minimum[0],
            container.minimum[1] + local.minimum[1],
        ),
        maximum=(
            container.minimum[0] + local.maximum[0],
            container.minimum[1] + local.maximum[1],
        ),
        scale=local.scale,
    )


def _rect(geometry: FittedImageGeometry) -> QRectF:
    return QRectF(
        geometry.minimum[0],
        geometry.minimum[1],
        geometry.width,
        geometry.height,
    )


def _result_key(result: dict[str, object]) -> tuple[str, str, int] | None:
    session_id = result.get("session_id")
    sequence_id = result.get("sequence_id")
    frame_index = result.get("frame_index")
    if (
        not isinstance(session_id, str)
        or not isinstance(sequence_id, str)
        or not isinstance(frame_index, int)
        or isinstance(frame_index, bool)
    ):
        return None
    return session_id, sequence_id, frame_index


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
        if not all(
            isinstance(item, (int, float)) and not isinstance(item, bool) and item >= 0
            for item in point
        ):
            return []
        result.append((float(point[0]), float(point[1])))
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


def _has_drawable_hands(result: dict[str, object]) -> bool:
    hands = result.get("hands")
    if not isinstance(hands, list):
        return False
    return any(
        isinstance(hand, dict)
        and (
            len(_points(hand.get("source_keypoints_2d_px"))) == 21
            or _bbox(hand.get("source_bbox_xyxy_px")) is not None
        )
        for hand in hands
    )


def _frame_stream_id(frame: LiveFrame | PlaybackFrame) -> str:
    if isinstance(frame, PlaybackFrame):
        return frame.clip_id
    return frame.connection_session_id
