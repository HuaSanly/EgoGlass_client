from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import av
import numpy as np
from PyQt6.QtCore import QRect, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import QHBoxLayout, QStackedWidget, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    FlowLayout,
    FluentIcon,
    ImageLabel,
    InfoBadge,
    MessageBox,
    SmoothScrollArea,
    StrongBodyLabel,
    TitleLabel,
    TransparentToolButton,
)

from schemas.recording import RecordingLibrary, RecordingSummary
from ui.widgets.recording_playback import RecordingPlaybackWidget

if TYPE_CHECKING:
    from ui.application.runtime_host import UnifiedRuntimeHost


class RecordingCard(CardWidget):
    activated = pyqtSignal(str)
    deleteRequested = pyqtSignal(str)

    def __init__(
        self,
        recording: RecordingSummary,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.recording = recording
        self.setObjectName(f"recordingCard-{recording.recording_id}")
        self.setFixedSize(330, 310)
        self.setClickEnabled(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clicked.connect(lambda: self.activated.emit(recording.recording_id))

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 12)
        root.setSpacing(8)
        self.thumbnail = ImageLabel(self)
        self.thumbnail.setFixedSize(310, 186)
        self.thumbnail.setBorderRadius(5, 5, 5, 5)
        self.thumbnail.setScaledContents(True)
        self.thumbnail.setImage(_thumbnail_placeholder())
        root.addWidget(self.thumbnail)

        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        self.title_label = StrongBodyLabel(recording.recording_id[:12], self)
        self.title_label.setToolTip(recording.recording_id)
        title_row.addWidget(self.title_label)
        title_row.addStretch(1)
        self.protocol_badge = (
            InfoBadge.success("协议通过", self)
            if recording.protocol_validated
            else InfoBadge.warning("协议失败", self)
        )
        title_row.addWidget(self.protocol_badge)
        self.delete_button = TransparentToolButton(FluentIcon.DELETE, self)
        self.delete_button.setToolTip("删除该录制")
        self.delete_button.clicked.connect(
            lambda: self.deleteRequested.emit(recording.recording_id)
        )
        title_row.addWidget(self.delete_button)
        root.addLayout(title_row)

        captured = datetime.fromtimestamp(
            recording.recorded_at_unix_ns / 1_000_000_000,
            UTC,
        ).astimezone()
        root.addWidget(
            CaptionLabel(
                f"{captured:%Y-%m-%d %H:%M:%S}  ·  {_clock_ns(recording.duration_ns)}",
                self,
            )
        )
        root.addWidget(
            CaptionLabel(
                f"{recording.width}×{recording.height}  ·  {recording.fps:.1f} FPS  ·  "
                f"{recording.frame_count:,} 帧",
                self,
            )
        )
        root.addWidget(
            CaptionLabel(
                f"IMU {recording.imu_sample_count:,} 行  ·  "
                f"{_file_size(recording.file_size_bytes)}",
                self,
            )
        )

    def set_thumbnail(self, image: QImage) -> None:
        self.thumbnail.setImage(image)


class RecordingLibraryView(QWidget):
    """Recording cards and lightweight capture inspection."""

    def __init__(
        self,
        runtime: UnifiedRuntimeHost,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("recordingLibraryView")
        self.runtime = runtime
        self.cards: dict[str, RecordingCard] = {}
        self._signature: tuple[tuple[object, ...], ...] = ()
        self._opening_future = None
        self._thumbnails = _RecordingThumbnailService()
        self._build()
        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._refresh_state)
        self._timer.start()

    def _build(self) -> None:
        self.stack = QStackedWidget(self)
        self.hall = self._build_hall()
        self.playback = RecordingPlaybackWidget(self)
        self.playback.backRequested.connect(self.show_library)
        self.stack.addWidget(self.hall)
        self.stack.addWidget(self.playback)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.stack)

    def _build_hall(self) -> QWidget:
        hall = QWidget(self)
        hall.setObjectName("recordingLibraryHall")
        root = QVBoxLayout(hall)
        root.setContentsMargins(24, 18, 24, 18)
        root.setSpacing(12)
        header = QHBoxLayout()
        title_column = QVBoxLayout()
        title_column.setSpacing(2)
        title_column.addWidget(TitleLabel("录制库", hall))
        title_column.addWidget(BodyLabel("检查每次独立录制的视频与原始 IMU 数据", hall))
        header.addLayout(title_column)
        header.addStretch(1)
        self.refresh_button = TransparentToolButton(FluentIcon.SYNC, hall)
        self.refresh_button.setToolTip("刷新录制库")
        self.refresh_button.clicked.connect(self._request_refresh)
        header.addWidget(self.refresh_button)
        root.addLayout(header)

        self.scroll = SmoothScrollArea(hall)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet(
            "SmoothScrollArea, QScrollArea, QScrollArea > QWidget > QWidget {"
            " border: none; background: #f5f7fb; }"
        )
        self.scroll.viewport().setStyleSheet("background: #f5f7fb;")
        self.content = QWidget(self.scroll)
        self.content.setObjectName("recordingLibraryContent")
        self.content.setStyleSheet(
            "QWidget#recordingLibraryContent { background: #f5f7fb; }"
        )
        self.flow = FlowLayout(self.content, needAni=False, isTight=True)
        self.flow.setContentsMargins(0, 8, 10, 16)
        self.flow.setHorizontalSpacing(16)
        self.flow.setVerticalSpacing(16)
        self.empty_label = BodyLabel("录制库为空。完成一次录制后可在这里检查。", self.content)
        self.flow.addWidget(self.empty_label)
        self.scroll.setWidget(self.content)
        root.addWidget(self.scroll, 1)
        return hall

    def set_library(self, library: RecordingLibrary) -> None:
        signature = tuple(
            (
                item.recording_id,
                item.duration_ns,
                item.frame_count,
                item.imu_sample_count,
                item.protocol_validated,
            )
            for item in library.recordings
        )
        if signature == self._signature:
            return
        self._signature = signature
        self.flow.takeAllWidgets()
        self.cards.clear()
        self.empty_label = BodyLabel(
            "录制库为空。完成一次录制后可在这里检查。",
            self.content,
        )
        self.empty_label.setVisible(not library.recordings)
        self.flow.addWidget(self.empty_label)
        for recording in library.recordings:
            card = RecordingCard(recording, self.content)
            card.activated.connect(self.open_recording)
            card.deleteRequested.connect(self._confirm_delete)
            self.flow.addWidget(card)
            self.cards[recording.recording_id] = card
            directory = self._recording_directory(recording.recording_id)
            if directory is not None:
                self._thumbnails.request(recording.recording_id, directory / "video.mp4")

    def open_recording(self, recording_id: str) -> None:
        open_reader = getattr(self.runtime, "recording_reader", None)
        if callable(open_reader):
            self.playback.show_loading(recording_id)
            self.stack.setCurrentWidget(self.playback)
            self._opening_future = open_reader(recording_id)
            return
        directory = self._recording_directory(recording_id)
        if directory is None:
            return
        self.playback.open_recording(directory)
        self.stack.setCurrentWidget(self.playback)

    def show_library(self) -> None:
        self.playback.unload()
        self.stack.setCurrentWidget(self.hall)

    def close_resources(self) -> None:
        self._timer.stop()
        self._thumbnails.close()
        self.playback.close_resources()

    def _refresh_state(self) -> None:
        snapshot = self.runtime.snapshot()
        library = getattr(snapshot, "library", None)
        if isinstance(library, RecordingLibrary):
            self.set_library(library)
        future = self._opening_future
        if future is not None and future.done():
            self._opening_future = None
            try:
                self.playback.open_reader(future.result())
            except Exception as error:
                self.playback.show_error(str(error))
        for result in self._thumbnails.take_completed():
            card = self.cards.get(result.recording_id)
            if card is not None and result.image is not None:
                card.set_thumbnail(result.image)

    def _request_refresh(self) -> None:
        request = getattr(self.runtime, "request_library_refresh", None)
        if callable(request):
            request()

    def _recording_directory(self, recording_id: str) -> Path | None:
        media_path = getattr(self.runtime, "recording_media_path", None)
        if callable(media_path):
            value = media_path(recording_id)
            if value is not None:
                return Path(value).parent
        for method_name in ("recording_directory", "recording_path"):
            method = getattr(self.runtime, method_name, None)
            if callable(method):
                value = method(recording_id)
                if value is not None:
                    return Path(value)
        root = getattr(self.runtime, "recordings_root", None)
        if root is not None:
            candidate = Path(root) / recording_id
            if candidate.is_dir() and candidate.parent.resolve() == Path(root).resolve():
                return candidate
        return None

    def _confirm_delete(self, recording_id: str) -> None:
        dialog = MessageBox(
            "删除录制",
            f"将永久删除录制 {recording_id[:12]} 及其 CSV 文件。",
            self.window(),
        )
        dialog.yesButton.setText("删除")
        dialog.cancelButton.setText("取消")
        if not dialog.exec():
            return
        for method_name in (
            "delete_recording",
            "request_recording_delete",
            "request_delete_recording",
        ):
            method = getattr(self.runtime, method_name, None)
            if callable(method):
                method(recording_id)
                return


@dataclass(frozen=True, slots=True)
class _ThumbnailResult:
    recording_id: str
    image: QImage | None


class _RecordingThumbnailService:
    def __init__(self) -> None:
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="recording-thumbnail",
        )
        self._futures: dict[str, concurrent.futures.Future[_ThumbnailResult]] = {}
        self._completed: set[str] = set()

    def request(self, recording_id: str, media_path: Path) -> None:
        if recording_id in self._completed or recording_id in self._futures:
            return
        self._futures[recording_id] = self._executor.submit(
            _decode_thumbnail,
            recording_id,
            media_path,
        )

    def take_completed(self) -> tuple[_ThumbnailResult, ...]:
        completed: list[_ThumbnailResult] = []
        for recording_id, future in tuple(self._futures.items()):
            if not future.done():
                continue
            del self._futures[recording_id]
            self._completed.add(recording_id)
            try:
                completed.append(future.result())
            except Exception:
                completed.append(_ThumbnailResult(recording_id, None))
        return tuple(completed)

    def close(self) -> None:
        for future in self._futures.values():
            future.cancel()
        self._futures.clear()
        self._executor.shutdown(wait=True, cancel_futures=True)


def _decode_thumbnail(recording_id: str, media_path: Path) -> _ThumbnailResult:
    with av.open(str(media_path), mode="r") as container:
        frame = next(container.decode(container.streams.video[0]), None)
        if frame is None:
            return _ThumbnailResult(recording_id, None)
        source = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
        height, width, _channels = source.shape
        image = QImage(
            source.data,
            width,
            height,
            int(source.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
    return _ThumbnailResult(recording_id, _letterbox(image, 310, 186))


def _thumbnail_placeholder() -> QImage:
    image = QImage(310, 186, QImage.Format.Format_RGB888)
    image.fill(QColor("#10151c"))
    return image


def _letterbox(source: QImage, width: int, height: int) -> QImage:
    output = QImage(width, height, QImage.Format.Format_RGB888)
    output.fill(QColor("#10151c"))
    scaled = source.scaled(
        width,
        height,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    painter = QPainter(output)
    painter.drawImage(
        QRect(
            (width - scaled.width()) // 2,
            (height - scaled.height()) // 2,
            scaled.width(),
            scaled.height(),
        ),
        scaled,
    )
    painter.end()
    return output


def _clock_ns(duration_ns: int) -> str:
    total_ms = max(0, duration_ns) // 1_000_000
    minutes, remainder = divmod(total_ms, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _file_size(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"
