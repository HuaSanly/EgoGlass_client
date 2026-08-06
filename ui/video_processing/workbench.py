from __future__ import annotations

from datetime import UTC, datetime

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CheckBox,
    ComboBox,
    FluentIcon,
    InfoBadge,
    PrimaryPushButton,
    PushButton,
    SimpleCardWidget,
    Slider,
    StrongBodyLabel,
    TitleLabel,
    TogglePushButton,
    TransparentToolButton,
)

from schemas import VioPose
from ui.presentation import SpatialReferenceFrame, build_spatial_scene_state
from ui.processing import ProcessingRunInfo, VioRunInfo
from ui.replay.player import PlaybackClipSpan, ReplayPlayer, ReplaySnapshot, ReplayState
from ui.widgets.spatial_sync_canvas import SpatialSyncCanvas
from ui.widgets.video_canvas import VideoCanvas


class ClipTimelineStrip(SimpleCardWidget):
    seekRequested = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sessionClipTimeline")
        self._signature: tuple[PlaybackClipSpan, ...] = ()
        self._buttons: dict[str, TogglePushButton] = {}
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(12, 8, 12, 8)
        self._layout.setSpacing(5)

    def set_clips(self, clips: tuple[PlaybackClipSpan, ...]) -> None:
        if clips == self._signature:
            return
        self._signature = clips
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._buttons.clear()
        if not clips:
            self._layout.addWidget(CaptionLabel("等待片段时间索引", self))
            return
        for index, span in enumerate(clips, start=1):
            button = TogglePushButton(self)
            button.setText(f"片段 {index:02d}  {_duration(span.end_seconds - span.start_seconds)}")
            button.setToolTip(f"{span.frame_count} 帧 · 点击定位")
            button.clicked.connect(
                lambda _checked=False, seconds=span.start_seconds: self.seekRequested.emit(seconds)
            )
            self._buttons[span.clip_id] = button
            stretch = max(1, round((span.end_seconds - span.start_seconds) * 10))
            self._layout.addWidget(button, stretch)

    def set_active_clip(self, clip_id: str | None) -> None:
        for key, button in self._buttons.items():
            button.blockSignals(True)
            button.setChecked(key == clip_id)
            button.blockSignals(False)


class _ReadonlySlicePanel(SimpleCardWidget):
    def __init__(self, title: str, detail: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("readonlySlicePlaceholder")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(5)
        header = QHBoxLayout()
        header.addWidget(StrongBodyLabel(title, self))
        header.addStretch(1)
        header.addWidget(InfoBadge.info("只读占位", self))
        layout.addLayout(header)
        label = CaptionLabel(detail, self)
        label.setWordWrap(True)
        layout.addWidget(label)


class ProcessingWorkbench(QWidget):
    backRequested = pyqtSignal()
    processRequested = pyqtSignal()
    exportRequested = pyqtSignal()
    resultSelectionChanged = pyqtSignal()
    comparisonSelectionChanged = pyqtSignal()

    RAW_RESULT = "__raw__"
    NO_COMPARISON = "__none__"

    def __init__(self, replay: ReplayPlayer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("processingWorkbench")
        self.replay = replay
        self._seeking = False
        self._session_id: str | None = None
        self._initial_clip_id: str | None = None
        self._runs: tuple[ProcessingRunInfo, ...] = ()
        self._vio_runs: tuple[VioRunInfo, ...] = ()
        self._spatial_imu_pose = None
        self._spatial_hand_result: dict[str, object] | None = None
        self._spatial_vio_pose: VioPose | None = None
        self._build_ui()

    @property
    def primary_run_id(self) -> str | None:
        data = self.result_combo.currentData()
        return data if isinstance(data, str) and data != self.RAW_RESULT else None

    @property
    def comparison_run_id(self) -> str | None:
        data = self.comparison_combo.currentData()
        return data if isinstance(data, str) and data != self.NO_COMPARISON else None

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 14, 24, 18)
        root.setSpacing(10)
        root.addLayout(self._build_header())

        content = QHBoxLayout()
        content.setSpacing(12)
        content.addWidget(self._build_video_column(), 1)
        content.addWidget(self._build_space_column(), 0)
        root.addLayout(content, 1)

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        self.back_button = TransparentToolButton(FluentIcon.RETURN, self)
        self.back_button.setToolTip("返回视频大厅")
        self.back_button.clicked.connect(self.backRequested)
        row.addWidget(self.back_button)
        title_column = QVBoxLayout()
        title_column.setSpacing(0)
        self.title_label = TitleLabel("处理工作台", self)
        self.context_label = CaptionLabel("尚未载入视频", self)
        title_column.addWidget(self.title_label)
        title_column.addWidget(self.context_label)
        row.addLayout(title_column)
        row.addStretch(1)
        self.process_button = PrimaryPushButton("处理当前视频", self)
        self.process_button.setIcon(FluentIcon.PLAY)
        self.process_button.setToolTip("逐帧运行手部追踪，并在同一离线任务中运行 SLAM/VIO")
        self.process_button.clicked.connect(self.processRequested)
        row.addWidget(self.process_button)
        row.addWidget(CaptionLabel("结果版本", self))
        self.result_combo = ComboBox(self)
        self.result_combo.setMinimumWidth(250)
        self.result_combo.currentIndexChanged.connect(self._primary_result_changed)
        row.addWidget(self.result_combo)
        self.export_button = PushButton("导出", self)
        self.export_button.setIcon(FluentIcon.SAVE)
        self.export_button.clicked.connect(self.exportRequested)
        row.addWidget(self.export_button)
        return row

    def _build_video_column(self) -> QWidget:
        column = QWidget(self)
        column.setObjectName("workbenchVideoColumn")
        column.setMinimumWidth(560)
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        frame_stack = QWidget(column)
        stack = QGridLayout(frame_stack)
        stack.setContentsMargins(0, 0, 0, 0)
        self.canvas = VideoCanvas(frame_stack)
        stack.addWidget(self.canvas, 0, 0)
        preview = self._build_preview_options(frame_stack)
        stack.addWidget(
            preview,
            0,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )
        slices = _ReadonlySlicePanel(
            "切片候选",
            "切片算法尚未接入；这里不会生成标注或写入文件。",
            frame_stack,
        )
        slices.setFixedWidth(245)
        stack.addWidget(
            slices,
            0,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
        )
        layout.addWidget(frame_stack, 1)
        self.clip_timeline = ClipTimelineStrip(column)
        self.clip_timeline.seekRequested.connect(self.replay.seek)
        layout.addWidget(self.clip_timeline)
        layout.addWidget(self._build_transport(column))
        return column

    def _build_preview_options(self, parent: QWidget) -> SimpleCardWidget:
        card = SimpleCardWidget(parent)
        card.setObjectName("previewOptions")
        card.setFixedWidth(250)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        layout.addWidget(StrongBodyLabel("预览选项", card))
        self.overlay_check = CheckBox("显示手部覆盖层", card)
        self.overlay_check.setChecked(True)
        self.overlay_check.toggled.connect(self.canvas_overlay_enabled)
        layout.addWidget(self.overlay_check)
        row = QHBoxLayout()
        row.addWidget(CaptionLabel("A/B", card))
        self.comparison_combo = ComboBox(card)
        self.comparison_combo.setMinimumWidth(174)
        self.comparison_combo.currentIndexChanged.connect(
            lambda _index: self.comparisonSelectionChanged.emit()
        )
        row.addWidget(self.comparison_combo, 1)
        layout.addLayout(row)
        return card

    def _build_transport(self, parent: QWidget) -> SimpleCardWidget:
        card = SimpleCardWidget(parent)
        card.setObjectName("processingTransport")
        row = QHBoxLayout(card)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(8)
        self.play_button = TransparentToolButton(FluentIcon.PLAY, card)
        self.play_button.setToolTip("播放或暂停")
        self.play_button.clicked.connect(self._toggle_playback)
        row.addWidget(self.play_button)
        self.step_button = TransparentToolButton(FluentIcon.RIGHT_ARROW, card)
        self.step_button.setToolTip("前进一帧")
        self.step_button.clicked.connect(self.replay.step)
        row.addWidget(self.step_button)
        self.position_slider = Slider(Qt.Orientation.Horizontal, card)
        self.position_slider.setRange(0, 1)
        self.position_slider.sliderPressed.connect(self._begin_seek)
        self.position_slider.sliderReleased.connect(self._finish_seek)
        row.addWidget(self.position_slider, 1)
        self.time_label = BodyLabel("00:00.000 / 00:00.000", card)
        row.addWidget(self.time_label)
        self.rate_combo = ComboBox(card)
        for value in ("0.25x", "0.5x", "1.0x", "1.5x", "2.0x"):
            self.rate_combo.addItem(value)
        self.rate_combo.setCurrentText("1.0x")
        self.rate_combo.currentTextChanged.connect(self._set_rate)
        self.rate_combo.setFixedWidth(82)
        row.addWidget(self.rate_combo)
        return card

    def _build_space_column(self) -> QWidget:
        column = QWidget(self)
        column.setObjectName("workbenchSpaceColumn")
        column.setFixedWidth(370)
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.spatial_canvas = SpatialSyncCanvas(column)
        self.spatial_canvas.setMinimumSize(370, 278)
        self.spatial_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.spatial_canvas.reset_pose_button.setVisible(False)
        self.spatial_canvas.set_reference_frame(SpatialReferenceFrame.WORLD)
        self.spatial_canvas.reference_frame_changed.connect(
            lambda _frame: self._refresh_spatial_scene()
        )
        layout.addWidget(self.spatial_canvas, 1)
        self.slice_markers = _ReadonlySlicePanel(
            "切片标记",
            "当前版本仅呈现占位状态，不支持编辑、保存或导出切片。",
            column,
        )
        self.slice_markers.setMinimumHeight(118)
        layout.addWidget(self.slice_markers)
        self.result_detail = CaptionLabel("当前显示原始视频", column)
        self.result_detail.setWordWrap(True)
        layout.addWidget(self.result_detail)
        self.vio_detail = CaptionLabel("SLAM/VIO：未加载离线轨迹", column)
        self.vio_detail.setWordWrap(True)
        layout.addWidget(self.vio_detail)
        return column

    def set_context(
        self,
        session_id: str,
        display_name: str,
        clip_id: str,
    ) -> None:
        self._session_id = session_id
        self._initial_clip_id = clip_id
        self.title_label.setText(display_name)
        self.context_label.setText(f"会话 {session_id[:8]}  ·  当前片段 {clip_id[:8]}")

    def set_runs(self, runs: tuple[ProcessingRunInfo, ...], clip_id: str) -> None:
        self._runs = tuple(
            sorted(
                (run for run in runs if run.covers_clip(clip_id)),
                key=lambda run: run.completed_at_unix_ns or 0,
                reverse=True,
            )
        )
        self.result_combo.blockSignals(True)
        self.result_combo.clear()
        self.result_combo.addItem("原始视频", userData=self.RAW_RESULT)
        for run in self._runs:
            self.result_combo.addItem(_run_label(run), userData=run.run_id)
        self.result_combo.setCurrentIndex(1 if self._runs else 0)
        self.result_combo.blockSignals(False)
        self._rebuild_comparison(emit=False)
        self._update_result_detail()
        self.export_button.setEnabled(self.primary_run_id is not None)
        self.resultSelectionChanged.emit()

    @property
    def selected_vio_run(self) -> VioRunInfo | None:
        clip_id = self._initial_clip_id
        return next(
            (
                run
                for run in self._vio_runs
                if run.is_viewable and (clip_id is None or run.covers_clip(clip_id))
            ),
            None,
        )

    def set_vio_runs(self, runs: tuple[VioRunInfo, ...]) -> None:
        self._vio_runs = tuple(
            sorted(
                (run for run in runs if run.is_viewable),
                key=lambda run: run.completed_at_unix_ns or 0,
                reverse=True,
            )
        )
        selected = self.selected_vio_run
        self._spatial_vio_pose = None
        self._refresh_spatial_scene()
        if selected is None:
            self.vio_detail.setText("SLAM/VIO：未加载离线轨迹")
        else:
            self.vio_detail.setText(
                f"SLAM/VIO：{selected.run_id[-8:]}  ·  {selected.pose_count} 个位姿"
            )

    def set_vio_pose(self, pose: VioPose | None) -> None:
        self._spatial_vio_pose = pose
        self._refresh_spatial_scene()

    def set_imu_pose(self, pose: object | None) -> None:
        self._spatial_imu_pose = pose
        self._refresh_spatial_scene()

    def set_hand_result(self, result: dict[str, object] | None) -> None:
        self._spatial_hand_result = result
        self._refresh_spatial_scene()

    def _refresh_spatial_scene(self) -> None:
        selected = self.selected_vio_run
        transform = selected.transform_camera_to_imu if selected is not None else None
        self.spatial_canvas.set_scene_state(
            build_spatial_scene_state(
                self.spatial_canvas.reference_frame,
                hand_result=self._spatial_hand_result,
                imu_pose=self._spatial_imu_pose,
                vio_pose=self._spatial_vio_pose,
                vio_first_pose=selected.first_pose if selected is not None else None,
                transform_camera_to_imu=transform
                if transform is not None
                else (
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
            )
        )

    def set_vio_status(self, text: str) -> None:
        """Update the read-only VIO stage status without changing the canvas."""

        self.vio_detail.setText(text)

    def set_replay(self, snapshot: ReplaySnapshot) -> None:
        self.play_button.setIcon(
            FluentIcon.PAUSE if snapshot.state is ReplayState.PLAYING else FluentIcon.PLAY
        )
        maximum = max(1, round(snapshot.duration_seconds * 1000))
        value = min(maximum, round(snapshot.position_seconds * 1000))
        if not self._seeking:
            self.position_slider.blockSignals(True)
            self.position_slider.setRange(0, maximum)
            self.position_slider.setValue(value)
            self.position_slider.blockSignals(False)
        self.time_label.setText(
            f"{_clock(snapshot.position_seconds)} / {_clock(snapshot.duration_seconds)}"
        )
        self.clip_timeline.set_clips(snapshot.clips)
        self.clip_timeline.set_active_clip(snapshot.clip_id)

    def canvas_overlay_enabled(self, enabled: bool) -> None:
        self.canvas.set_primary_overlay_enabled(enabled)
        self.canvas.set_comparison_overlay_enabled(enabled)

    def clear_media(self) -> None:
        self.canvas.clear()
        self._vio_runs = ()
        self._spatial_imu_pose = None
        self._spatial_vio_pose = None
        self._spatial_hand_result = None
        self.spatial_canvas.set_scene_state(None)
        self.clip_timeline.set_clips(())
        self.result_combo.clear()
        self.comparison_combo.clear()
        self.result_detail.setText("当前显示原始视频")
        self.vio_detail.setText("SLAM/VIO：未加载离线轨迹")

    def _primary_result_changed(self, _index: int) -> None:
        self._rebuild_comparison(emit=False)
        self._update_result_detail()
        self.export_button.setEnabled(self.primary_run_id is not None)
        self.resultSelectionChanged.emit()

    def _rebuild_comparison(self, *, emit: bool = True) -> None:
        primary = self.primary_run_id
        self.comparison_combo.blockSignals(True)
        self.comparison_combo.clear()
        self.comparison_combo.addItem("关闭对比", userData=self.NO_COMPARISON)
        for run in self._runs:
            if run.run_id != primary:
                self.comparison_combo.addItem(_short_run_label(run), userData=run.run_id)
        # A/B is an explicit inspection mode. Do not carry a previous
        # comparison into a newly loaded clip or refreshed result list.
        self.comparison_combo.setCurrentIndex(0)
        self.comparison_combo.blockSignals(False)
        if emit:
            self.comparisonSelectionChanged.emit()

    def _update_result_detail(self) -> None:
        run_id = self.primary_run_id
        run = next((item for item in self._runs if item.run_id == run_id), None)
        if run is None:
            self.result_detail.setText("当前显示原始视频；视频只解码一次。")
            return
        completed = (
            datetime.fromtimestamp(run.completed_at_unix_ns / 1_000_000_000, UTC)
            .astimezone()
            .strftime("%Y-%m-%d %H:%M:%S")
            if run.completed_at_unix_ns is not None
            else "--"
        )
        scope = "完整会话" if run.clip_id is None else "当前片段"
        self.result_detail.setText(
            f"{run.preset.display_name}  ·  {scope}\n"
            f"完成 {completed}  ·  推理 {run.inferred_frame_count} 帧  ·  "
            f"检测 {run.detected_hand_count} 只手  ·  {run.run_id}"
        )

    def _toggle_playback(self) -> None:
        if self.replay.snapshot().state is ReplayState.PLAYING:
            self.replay.pause()
        else:
            self.replay.play()

    def _begin_seek(self) -> None:
        self._seeking = True

    def _finish_seek(self) -> None:
        self._seeking = False
        self.replay.seek(self.position_slider.value() / 1000)

    def _set_rate(self, value: str) -> None:
        if value:
            self.replay.set_playback_rate(float(value.removesuffix("x")))


def _run_label(run: ProcessingRunInfo) -> str:
    scope = "会话" if run.clip_id is None else "片段"
    return f"{run.preset.display_name} · {scope} · {run.run_id[-8:]}"


def _short_run_label(run: ProcessingRunInfo) -> str:
    return f"{run.preset.display_name} · {run.run_id[-8:]}"


def _clock(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    minutes, remainder = divmod(milliseconds, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    minutes, remainder = divmod(total, 60)
    return f"{minutes:02d}:{remainder:02d}"
