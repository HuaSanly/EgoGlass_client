from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    FluentIcon,
    InfoBadge,
    SimpleCardWidget,
    Slider,
    StrongBodyLabel,
    TitleLabel,
    TransparentToolButton,
)

from schemas.recording import RecordingFrameRow
from ui.gateway.capture_recording import CaptureRecordingReader
from ui.widgets.video_canvas import VideoCanvas


@dataclass(frozen=True, slots=True)
class RecordingPlaybackFrame:
    recording_id: str
    frame_index: int
    recording_time_ns: int
    image_rgb: np.ndarray

    @property
    def width(self) -> int:
        return int(self.image_rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.image_rgb.shape[0])


@dataclass(frozen=True, slots=True)
class ImuCursorSample:
    recording_time_ns: int
    accelerometer: tuple[float, float, float] | None
    gyroscope: tuple[float, float, float] | None


class RecordingReplaySource:
    """Small PyAV decoder indexed by the validated capture CSV contracts."""

    def __init__(self, source: Path | CaptureRecordingReader) -> None:
        self.reader = (
            source
            if isinstance(source, CaptureRecordingReader)
            else CaptureRecordingReader.open(source, verify_hashes=False)
        )
        self.frames = tuple(self.reader.iter_frames())
        self.imu = tuple(self.reader.iter_imu_samples())
        self._imu_times = tuple(row.recording_time_ns for row in self.imu)
        self._container: av.InputContainer | None = None
        self._stream: av.video.stream.VideoStream | None = None
        self._decoder = None
        self._cursor = 0
        self._open_decoder()

    @property
    def recording_id(self) -> str:
        return self.reader.manifest.recording_id

    @property
    def duration_ns(self) -> int:
        return int(self.reader.manifest.duration_ns or 0)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def next_frame(self) -> RecordingPlaybackFrame | None:
        if self._cursor >= len(self.frames):
            return None
        assert self._decoder is not None
        while True:
            decoded = next(self._decoder, None)
            if decoded is None:
                return None
            reference_index = self._find_reference(decoded)
            if reference_index is None or reference_index < self._cursor:
                continue
            self._cursor = reference_index + 1
            return self._frame(decoded, self.frames[reference_index])

    def seek_frame(self, frame_index: int) -> RecordingPlaybackFrame:
        if not 0 <= frame_index < len(self.frames):
            raise IndexError("recording frame index is out of range")
        target = self.frames[frame_index]
        assert self._container is not None and self._stream is not None
        self._container.seek(target.mp4_pts, stream=self._stream, backward=True)
        self._decoder = self._container.decode(self._stream)
        self._cursor = frame_index
        frame = self.next_frame()
        if frame is None:
            raise ValueError("decoder did not return the indexed recording frame")
        return frame

    def imu_cursor(self, recording_time_ns: int) -> ImuCursorSample:
        cursor = bisect_right(self._imu_times, recording_time_ns)
        accelerometer = None
        gyroscope = None
        for row in reversed(self.imu[:cursor]):
            sensor = str(getattr(row.sensor_type, "value", row.sensor_type))
            values = (row.x, row.y, row.z)
            if sensor == "accelerometer" and accelerometer is None:
                accelerometer = values
            elif sensor == "gyroscope" and gyroscope is None:
                gyroscope = values
            if accelerometer is not None and gyroscope is not None:
                break
        return ImuCursorSample(recording_time_ns, accelerometer, gyroscope)

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
        self._container = None
        self._stream = None
        self._decoder = None

    def _open_decoder(self) -> None:
        self.close()
        self._container = av.open(str(self.reader.video_path), mode="r")
        self._stream = self._container.streams.video[0]
        self._decoder = self._container.decode(self._stream)
        self._cursor = 0

    def _find_reference(self, frame: av.VideoFrame) -> int | None:
        if frame.pts is None:
            return None
        for index in range(max(0, self._cursor - 1), len(self.frames)):
            reference = self.frames[index]
            if reference.mp4_pts == frame.pts:
                return index
            if reference.mp4_pts > frame.pts:
                return None
        return None

    def _frame(
        self,
        decoded: av.VideoFrame,
        reference: RecordingFrameRow,
    ) -> RecordingPlaybackFrame:
        image_rgb = np.ascontiguousarray(decoded.to_ndarray(format="rgb24"))
        image_rgb.setflags(write=False)
        return RecordingPlaybackFrame(
            recording_id=self.recording_id,
            frame_index=reference.frame_index,
            recording_time_ns=reference.recording_time_ns,
            image_rgb=image_rgb,
        )


class RecordingPlaybackWidget(QWidget):
    """Lightweight video check with an IMU cursor on the recording timeline."""

    backRequested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("recordingPlayback")
        self._source: RecordingReplaySource | None = None
        self._playing = False
        self._seeking = False
        self._build()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 20)
        root.setSpacing(12)

        header = QHBoxLayout()
        self.back_button = TransparentToolButton(FluentIcon.RETURN, self)
        self.back_button.setToolTip("返回录制库")
        self.back_button.clicked.connect(self.backRequested)
        header.addWidget(self.back_button)
        titles = QVBoxLayout()
        titles.setSpacing(1)
        titles.addWidget(TitleLabel("录制检查", self))
        self.context_label = CaptionLabel("尚未载入录制", self)
        titles.addWidget(self.context_label)
        header.addLayout(titles)
        header.addStretch(1)
        self.hash_badge = InfoBadge.info("未校验", self)
        header.addWidget(self.hash_badge)
        root.addLayout(header)

        content = QHBoxLayout()
        content.setSpacing(14)
        self.canvas = VideoCanvas(self)
        content.addWidget(self.canvas, 1)
        content.addWidget(self._build_cursor_card(), 0)
        root.addLayout(content, 1)
        root.addWidget(self._build_transport())

    def _build_cursor_card(self) -> SimpleCardWidget:
        card = SimpleCardWidget(self)
        card.setObjectName("recordingImuCursor")
        card.setFixedWidth(310)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        header = QHBoxLayout()
        header.addWidget(StrongBodyLabel("IMU 时间游标", card))
        header.addStretch(1)
        self.cursor_badge = InfoBadge.info("00:00.000", card)
        header.addWidget(self.cursor_badge)
        layout.addLayout(header)
        layout.addWidget(CaptionLabel("加速度  m/s²", card))
        self.accelerometer_value = BodyLabel("X --   Y --   Z --", card)
        layout.addWidget(self.accelerometer_value)
        layout.addWidget(CaptionLabel("角速度  rad/s", card))
        self.gyroscope_value = BodyLabel("X --   Y --   Z --", card)
        layout.addWidget(self.gyroscope_value)
        self.cursor_detail = CaptionLabel(
            "游标按 frames.csv 的 recording_time_ns 定位 imu.csv。",
            card,
        )
        self.cursor_detail.setWordWrap(True)
        layout.addWidget(self.cursor_detail)
        layout.addStretch(1)
        return card

    def _build_transport(self) -> SimpleCardWidget:
        card = SimpleCardWidget(self)
        card.setObjectName("recordingPlaybackTransport")
        row = QHBoxLayout(card)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(8)
        self.play_button = TransparentToolButton(FluentIcon.PLAY, card)
        self.play_button.setToolTip("播放或暂停")
        self.play_button.clicked.connect(self.toggle_playback)
        row.addWidget(self.play_button)
        self.restart_button = TransparentToolButton(FluentIcon.SYNC, card)
        self.restart_button.setToolTip("返回第一帧")
        self.restart_button.clicked.connect(lambda: self.seek_frame(0))
        row.addWidget(self.restart_button)
        self.position_slider = Slider(Qt.Orientation.Horizontal, card)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderPressed.connect(self._begin_seek)
        self.position_slider.sliderReleased.connect(self._finish_seek)
        row.addWidget(self.position_slider, 1)
        self.time_label = BodyLabel("00:00.000 / 00:00.000", card)
        row.addWidget(self.time_label)
        return card

    def open_recording(self, recording_directory: Path) -> None:
        self.open_reader(
            CaptureRecordingReader.open(recording_directory, verify_hashes=False)
        )

    def open_reader(self, reader: CaptureRecordingReader) -> None:
        self.unload()
        self._source = RecordingReplaySource(reader)
        self.position_slider.setRange(0, max(0, self._source.frame_count - 1))
        self.context_label.setText(
            f"录制 {self._source.recording_id}  ·  {self._source.frame_count:,} 帧"
        )
        self.hash_badge.setText(
            "哈希已校验" if self._source.reader.hashes_verified else "索引已校验"
        )
        self.seek_frame(0)

    def show_loading(self, recording_id: str) -> None:
        self.unload()
        self.context_label.setText(f"正在载入录制 {recording_id[:12]}…")

    def show_error(self, detail: str) -> None:
        self.unload()
        self.context_label.setText(f"录制载入失败：{detail}")

    def unload(self) -> None:
        self._timer.stop()
        self._playing = False
        if self._source is not None:
            self._source.close()
        self._source = None
        self.canvas.clear()
        self.position_slider.setRange(0, 0)
        self.play_button.setIcon(FluentIcon.PLAY)

    def close_resources(self) -> None:
        self.unload()

    def toggle_playback(self) -> None:
        if self._source is None:
            return
        if (
            not self._playing
            and self.position_slider.value() == self.position_slider.maximum()
        ):
            self.seek_frame(0)
        self._playing = not self._playing
        self.play_button.setIcon(FluentIcon.PAUSE if self._playing else FluentIcon.PLAY)
        if self._playing:
            interval_ms = max(1, round(1000 / self._source.reader.manifest.video_profile.fps))
            self._timer.start(interval_ms)
        else:
            self._timer.stop()

    def seek_frame(self, frame_index: int) -> None:
        if self._source is None:
            return
        frame = self._source.seek_frame(frame_index)
        self._show_frame(frame)

    def _advance(self) -> None:
        if self._source is None:
            return
        frame = self._source.next_frame()
        if frame is None:
            self._timer.stop()
            self._playing = False
            self.play_button.setIcon(FluentIcon.PLAY)
            return
        self._show_frame(frame)

    def _show_frame(self, frame: RecordingPlaybackFrame) -> None:
        assert self._source is not None
        self.canvas.set_frame(frame)
        if not self._seeking:
            self.position_slider.setValue(frame.frame_index)
        cursor = self._source.imu_cursor(frame.recording_time_ns)
        self.cursor_badge.setText(_clock_ns(cursor.recording_time_ns))
        self.accelerometer_value.setText(_vector_text(cursor.accelerometer))
        self.gyroscope_value.setText(_vector_text(cursor.gyroscope))
        self.time_label.setText(
            f"{_clock_ns(frame.recording_time_ns)} / {_clock_ns(self._source.duration_ns)}"
        )

    def _begin_seek(self) -> None:
        self._seeking = True

    def _finish_seek(self) -> None:
        self._seeking = False
        self.seek_frame(self.position_slider.value())


def _vector_text(values: tuple[float, float, float] | None) -> str:
    if values is None:
        return "X --   Y --   Z --"
    return f"X {values[0]:+.3f}   Y {values[1]:+.3f}   Z {values[2]:+.3f}"


def _clock_ns(duration_ns: int) -> str:
    total_ms = max(0, duration_ns) // 1_000_000
    minutes, remainder = divmod(total_ms, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
