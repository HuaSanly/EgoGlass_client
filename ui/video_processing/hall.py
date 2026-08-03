from __future__ import annotations

from datetime import UTC, datetime

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    ElevatedCardWidget,
    FlowLayout,
    FluentIcon,
    ImageLabel,
    InfoBadge,
    InfoLevel,
    SmoothScrollArea,
    StrongBodyLabel,
    TitleLabel,
    TransparentToolButton,
)

from ingest_gateway.recording_models import (
    CaptureSessionState,
    RecordingClip,
    RecordingLibrary,
    RecordingSession,
)


class VideoClipCard(ElevatedCardWidget):
    activated = pyqtSignal(str, str)

    def __init__(
        self,
        session: RecordingSession,
        clip: RecordingClip,
        clip_number: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.session_id = session.session_id
        self.clip_id = clip.clip_id
        self._result_count = 0
        self._processing_state: str | None = None
        self.setObjectName(f"videoClipCard-{clip.clip_id}")
        self.setFixedSize(322, 338)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clicked.connect(lambda: self.activated.emit(self.session_id, self.clip_id))

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 12)
        root.setSpacing(8)
        self.thumbnail = ImageLabel(self)
        self.thumbnail.setObjectName("videoClipThumbnail")
        self.thumbnail.setFixedSize(302, 226)
        self.thumbnail.setBorderRadius(5, 5, 5, 5)
        self.thumbnail.setScaledContents(True)
        root.addWidget(self.thumbnail)

        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        title_row.addWidget(StrongBodyLabel(f"片段 {clip_number:02d}", self))
        title_row.addStretch(1)
        self.result_badge = InfoBadge.info("结果 0", self)
        title_row.addWidget(self.result_badge)
        root.addLayout(title_row)

        captured = datetime.fromtimestamp(clip.recorded_at_unix_ms / 1000, UTC).astimezone()
        self.primary_meta = CaptionLabel(
            f"{captured:%Y-%m-%d %H:%M:%S}  ·  {_clock(clip.duration_ms)}",
            self,
        )
        root.addWidget(self.primary_meta)
        self.secondary_meta = CaptionLabel(
            f"{clip.width}×{clip.height}  ·  {clip.fps} FPS  ·  "
            f"{clip.frame_count} 帧  ·  {_file_size(clip.file_size_bytes)}",
            self,
        )
        root.addWidget(self.secondary_meta)

        complete = session.state is CaptureSessionState.COMPLETE
        self.availability_badge = (
            InfoBadge.success("可回放", self)
            if complete
            else InfoBadge.warning("会话未完成，暂不可用", self)
        )
        status_row = QHBoxLayout()
        status_row.addWidget(self.availability_badge)
        self.processing_badge = InfoBadge.info("未处理", self)
        status_row.addWidget(self.processing_badge)
        status_row.addStretch(1)
        root.addLayout(status_row)
        self.setEnabled(complete)
        if not complete:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def set_thumbnail(self, image: QImage | None) -> None:
        if image is not None:
            self.thumbnail.setImage(image)

    def set_result_count(self, count: int) -> None:
        self._result_count = max(0, count)
        self.result_badge.setText(f"结果 {self._result_count}")
        self._sync_processing_badge()

    def set_processing_state(self, state: str | None) -> None:
        self._processing_state = state
        self._sync_processing_badge()

    def _sync_processing_badge(self) -> None:
        state = self._processing_state
        if state is None:
            self.processing_badge.setText(
                "未处理" if self._result_count == 0 else "已有结果"
            )
            self.processing_badge.setLevel(
                InfoLevel.INFOAMTION
                if self._result_count == 0
                else InfoLevel.SUCCESS
            )
            return
        self.processing_badge.setText(state)
        self.processing_badge.setLevel(
            InfoLevel.ERROR
            if state == "失败"
            else (
                InfoLevel.SUCCESS
                if state == "完成"
                else InfoLevel.ATTENTION
            )
        )


class _SessionSection(QWidget):
    activated = pyqtSignal(str, str)

    def __init__(self, session: RecordingSession, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 10)
        root.setSpacing(10)
        header = QHBoxLayout()
        header.setSpacing(8)
        header.addWidget(StrongBodyLabel(session.display_name or session.session_id[:8], self))
        header.addWidget(CaptionLabel(f"{len(session.clips)} 个片段", self))
        if session.state is not CaptureSessionState.COMPLETE:
            header.addWidget(InfoBadge.warning("未完成", self))
        header.addStretch(1)
        root.addLayout(header)
        self.flow = FlowLayout(needAni=False, isTight=False)
        self.flow.setHorizontalSpacing(14)
        self.flow.setVerticalSpacing(14)
        self.cards: dict[str, VideoClipCard] = {}
        for index, clip in enumerate(session.clips, start=1):
            card = VideoClipCard(session, clip, index, self)
            card.activated.connect(self.activated)
            self.cards[clip.clip_id] = card
            self.flow.addWidget(card)
        root.addLayout(self.flow)


class VideoHall(QWidget):
    refreshRequested = pyqtSignal()
    clipActivated = pyqtSignal(str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("videoHall")
        self.cards: dict[tuple[str, str], VideoClipCard] = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 18)
        root.setSpacing(12)

        header = QHBoxLayout()
        title_column = QVBoxLayout()
        title_column.setSpacing(2)
        title_column.addWidget(TitleLabel("视频处理", self))
        title_column.addWidget(BodyLabel("选择一段录像，查看原始视频与处理结果", self))
        header.addLayout(title_column)
        header.addStretch(1)
        self.refresh_button = TransparentToolButton(FluentIcon.SYNC, self)
        self.refresh_button.setToolTip("刷新录像库")
        self.refresh_button.clicked.connect(self.refreshRequested)
        header.addWidget(self.refresh_button)
        root.addLayout(header)

        self.scroll = SmoothScrollArea(self)
        self.scroll.setObjectName("videoHallScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.content = QWidget(self.scroll)
        self.content.setObjectName("videoHallContent")
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 8, 10, 16)
        self.content_layout.setSpacing(18)
        self.empty_label = BodyLabel("录像库为空。完成一次采集后在这里处理视频。", self.content)
        self.content_layout.addWidget(self.empty_label)
        self.content_layout.addStretch(1)
        self.scroll.setWidget(self.content)
        root.addWidget(self.scroll, 1)

    def set_library(self, library: RecordingLibrary) -> None:
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.cards.clear()
        sessions = [session for session in library.sessions if session.clips]
        self.empty_label = BodyLabel("录像库为空。完成一次采集后在这里处理视频。", self.content)
        self.empty_label.setVisible(not sessions)
        self.content_layout.addWidget(self.empty_label)
        for session in sessions:
            section = _SessionSection(session, self.content)
            section.activated.connect(self.clipActivated)
            self.content_layout.addWidget(section)
            for clip_id, card in section.cards.items():
                self.cards[(session.session_id, clip_id)] = card
        self.content_layout.addStretch(1)

    def set_thumbnail(self, session_id: str, clip_id: str, image: QImage | None) -> None:
        card = self.cards.get((session_id, clip_id))
        if card is not None:
            card.set_thumbnail(image)

    def set_result_counts(self, session_id: str, counts: dict[str, int]) -> None:
        for (card_session, clip_id), card in self.cards.items():
            if card_session == session_id:
                card.set_result_count(counts.get(clip_id, 0))

    def set_processing_states(self, session_id: str, states: dict[str, str]) -> None:
        for (card_session, clip_id), card in self.cards.items():
            if card_session == session_id:
                card.set_processing_state(states.get(clip_id))


def _clock(duration_ms: int) -> str:
    minutes, remainder = divmod(max(0, duration_ms), 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _file_size(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"
