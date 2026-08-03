from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QSizePolicy,
    QStackedWidget,
    QTableWidgetItem,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    Action,
    BodyLabel,
    CaptionLabel,
    ComboBox,
    CommandBar,
    FluentIcon,
    HeaderCardWidget,
    InfoBar,
    Pivot,
    SimpleCardWidget,
    Slider,
    StrongBodyLabel,
    SwitchButton,
    TableWidget,
    TitleLabel,
    TransparentToolButton,
    TreeWidget,
)

from ingest_gateway.recording_models import CaptureSessionState, RecordingLibrary
from perception.video_processing import ProcessingJob, ProcessingJobState, ProcessingRunInfo
from ui.replay.player import PlaybackFrame, ReplayPlayer, ReplaySnapshot, ReplayState
from ui.runtime import UnifiedRuntimeHost
from ui.widgets.spatial_sync_canvas import SpatialSyncCanvas
from ui.widgets.status_indicator import InfoLevel, StatusIndicator
from ui.widgets.video_canvas import VideoCanvas


@dataclass(frozen=True, slots=True)
class _Selection:
    session_id: str
    clip_id: str | None


class VideoProcessingView(QWidget):
    """Primary stored-session processing and synchronized inspection workspace."""

    def __init__(
        self,
        runtime: UnifiedRuntimeHost,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("videoProcessingView")
        self.runtime = runtime
        self.replay = ReplayPlayer()
        self._selection: _Selection | None = None
        self._library_signature: tuple[tuple[str, tuple[str, ...]], ...] = ()
        self._last_frame_key: tuple[str, str, int, int] | None = None
        self._last_processing_revision = -1
        self._run_future: concurrent.futures.Future[tuple[ProcessingRunInfo, ...]] | None = None
        self._result_futures: dict[
            str,
            tuple[
                tuple[str, str, int, int],
                concurrent.futures.Future[dict[str, object] | None],
            ],
        ] = {}
        self._run_ids: dict[str, str] = {}
        self._task_row_ids: list[str] = []
        self._seeking = False

        self._build_ui()
        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(16)
        self._frame_timer.timeout.connect(self._update_frame)
        self._frame_timer.start()
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(100)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start()

    def close_resources(self) -> None:
        self._frame_timer.stop()
        self._status_timer.stop()
        self.replay.close()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 18)
        root.setSpacing(12)

        heading = QHBoxLayout()
        heading.addWidget(TitleLabel("视频处理", self))
        heading.addStretch(1)
        self.workspace_status = StatusIndicator(
            "等待会话",
            FluentIcon.MOVIE,
            self,
            level=InfoLevel.INFOAMTION,
        )
        heading.addWidget(self.workspace_status)
        root.addLayout(heading)
        root.addWidget(self._build_command_bar())

        workspace = QHBoxLayout()
        workspace.setSpacing(12)
        workspace.addWidget(self._build_session_browser(), 0)
        workspace.addWidget(self._build_video_column(), 1)
        workspace.addWidget(self._build_inspection_column(), 0)
        root.addLayout(workspace, 1)
        root.addWidget(self._build_task_panel())

    def _build_command_bar(self) -> CommandBar:
        bar = CommandBar(self)
        self.refresh_action = Action(
            FluentIcon.SYNC,
            "刷新会话",
            triggered=self.runtime.request_library_refresh,
        )
        self.start_action = Action(FluentIcon.PLAY, "开始处理", triggered=self._start_processing)
        self.cancel_action = Action(
            FluentIcon.CANCEL,
            "取消任务",
            triggered=self._cancel_processing,
        )
        self.retry_action = Action(FluentIcon.SYNC, "重试", triggered=self._retry_processing)
        self.export_action = Action(FluentIcon.SAVE, "导出标注视频", triggered=self._export_result)
        bar.addAction(self.refresh_action)
        bar.addSeparator()
        bar.addAction(self.start_action)
        bar.addAction(self.cancel_action)
        bar.addAction(self.retry_action)
        bar.addSeparator()
        bar.addAction(self.export_action)
        self.preset_combo = ComboBox(bar)
        self.preset_combo.setMinimumWidth(176)
        for text in ("手部追踪 · 质量优先", "手部追踪 · 均衡", "手部追踪 · 快速预览"):
            self.preset_combo.addItem(text)
        bar.addWidget(self.preset_combo)
        return bar

    def _build_session_browser(self) -> QWidget:
        card = HeaderCardWidget("会话与片段", self)
        card.setFixedWidth(230)
        layout = QVBoxLayout()
        layout.setSpacing(8)
        self.session_tree = TreeWidget(card)
        self.session_tree.setHeaderHidden(True)
        self.session_tree.setMinimumHeight(420)
        self.session_tree.currentItemChanged.connect(self._select_tree_item)
        layout.addWidget(self.session_tree, 1)
        self.session_hint = CaptionLabel("启动和手动刷新时扫描录像库", card)
        self.session_hint.setWordWrap(True)
        layout.addWidget(self.session_hint)
        card.viewLayout.addLayout(layout)
        return card

    def _build_video_column(self) -> QWidget:
        column = QWidget(self)
        column.setObjectName("processingVideoColumn")
        column.setMinimumWidth(500)
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.canvas = VideoCanvas(column)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self._build_transport())
        return column

    def _build_transport(self) -> SimpleCardWidget:
        card = SimpleCardWidget(self)
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

    def _build_inspection_column(self) -> QWidget:
        column = QWidget(self)
        column.setObjectName("processingInspectionColumn")
        column.setFixedWidth(360)
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.spatial_canvas = SpatialSyncCanvas(column)
        self.spatial_canvas.setMinimumSize(360, 270)
        self.spatial_canvas.setMaximumHeight(330)
        self.spatial_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.spatial_canvas.reset_pose_button.setVisible(False)
        layout.addWidget(self.spatial_canvas, 1)
        layout.addWidget(self._build_inspector(), 1)
        return column

    def _build_inspector(self) -> HeaderCardWidget:
        card = HeaderCardWidget("处理检查器", self)
        root = QVBoxLayout()
        self.inspector_pivot = Pivot(card)
        root.addWidget(self.inspector_pivot)
        self.inspector_stack = QStackedWidget(card)
        self.inspector_pages: dict[str, QWidget] = {}
        for key, text, builder in (
            ("preset", "处理方案", self._preset_page),
            ("layers", "结果图层", self._layers_page),
            ("frame", "当前帧", self._frame_page),
        ):
            page = builder(card)
            page.setObjectName(f"processingInspector-{key}")
            self.inspector_pages[key] = page
            self.inspector_pivot.addItem(key, text)
            self.inspector_stack.addWidget(page)
        root.addWidget(self.inspector_stack, 1)
        self.inspector_pivot.currentItemChanged.connect(self._show_inspector_page)
        self.inspector_pivot.setCurrentItem("preset")
        card.viewLayout.addLayout(root)
        return card

    def _preset_page(self, parent: QWidget) -> QWidget:
        page = QWidget(parent)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(10)
        self.preset_detail = BodyLabel("逐帧执行传感器预处理与手部识别。", page)
        self.preset_detail.setWordWrap(True)
        layout.addWidget(self.preset_detail)
        auto_row = QHBoxLayout()
        auto_row.addWidget(BodyLabel("会话完成后自动入队", page))
        auto_row.addStretch(1)
        self.auto_queue_switch = SwitchButton(page)
        self.auto_queue_switch.setChecked(False)
        self.auto_queue_switch.checkedChanged.connect(
            self.runtime.request_processing_auto_queue
        )
        auto_row.addWidget(self.auto_queue_switch)
        layout.addLayout(auto_row)
        layout.addStretch(1)
        return page

    def _layers_page(self, parent: QWidget) -> QWidget:
        page = QWidget(parent)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(CaptionLabel("主结果", page))
        self.primary_run_combo = ComboBox(page)
        self.primary_run_combo.currentTextChanged.connect(self._clear_result_queries)
        layout.addWidget(self.primary_run_combo)
        layout.addWidget(CaptionLabel("A/B 对比", page))
        self.comparison_run_combo = ComboBox(page)
        self.comparison_run_combo.addItem("关闭对比")
        self.comparison_run_combo.currentTextChanged.connect(self._clear_result_queries)
        layout.addWidget(self.comparison_run_combo)
        self.layer_status = CaptionLabel("当前会话没有处理结果", page)
        self.layer_status.setWordWrap(True)
        layout.addWidget(self.layer_status)
        layout.addStretch(1)
        return page

    def _frame_page(self, parent: QWidget) -> QWidget:
        page = QWidget(parent)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(7)
        self.frame_identity = StrongBodyLabel("未载入帧", page)
        self.frame_timing = BodyLabel("PTS --\n会话时间 --", page)
        self.frame_timing.setWordWrap(True)
        self.frame_result = CaptionLabel("未查询到结构化结果", page)
        self.frame_result.setWordWrap(True)
        self.frame_imu = CaptionLabel("IMU --", page)
        self.frame_imu.setWordWrap(True)
        layout.addWidget(self.frame_identity)
        layout.addWidget(self.frame_timing)
        layout.addWidget(self.frame_imu)
        layout.addWidget(self.frame_result)
        layout.addStretch(1)
        return page

    def _build_task_panel(self) -> SimpleCardWidget:
        card = SimpleCardWidget(self)
        card.setObjectName("processingTaskPanel")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        header = QHBoxLayout()
        header.addWidget(StrongBodyLabel("处理任务", card))
        header.addStretch(1)
        self.task_summary = CaptionLabel("暂无任务", card)
        header.addWidget(self.task_summary)
        self.task_toggle = TransparentToolButton(FluentIcon.DOWN, card)
        self.task_toggle.setToolTip("展开或折叠任务列表")
        self.task_toggle.clicked.connect(self._toggle_tasks)
        header.addWidget(self.task_toggle)
        layout.addLayout(header)
        self.task_table = TableWidget(card)
        self.task_table.setColumnCount(7)
        self.task_table.setHorizontalHeaderLabels(
            ["会话", "范围", "方案", "状态", "进度", "耗时", "说明 / 失败原因"]
        )
        self.task_table.verticalHeader().setVisible(False)
        self.task_table.setSelectionBehavior(TableWidget.SelectionBehavior.SelectRows)
        self.task_table.setEditTriggers(TableWidget.EditTrigger.NoEditTriggers)
        self.task_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.task_table.setMaximumHeight(180)
        self.task_table.setVisible(False)
        layout.addWidget(self.task_table)
        return card

    def _update_frame(self) -> None:
        snapshot = self.replay.snapshot()
        self._sync_transport(snapshot)
        frame = snapshot.frame
        if frame is None:
            return
        key = (frame.session_id, frame.clip_id, frame.frame_index, frame.session_time_ns)
        if key != self._last_frame_key:
            self._last_frame_key = key
            self.canvas.set_frame(frame)
            self.spatial_canvas.set_pose(snapshot.imu_pose)
            self._query_results(frame)
            self._update_frame_inspector(frame, snapshot)
        self._resolve_result_queries(key)

    def _update_status(self) -> None:
        snapshot = self.runtime.snapshot()
        if snapshot.library is not None:
            self._refresh_tree(snapshot.library)
        processing = snapshot.processing
        if processing is not None and processing.revision != self._last_processing_revision:
            self._last_processing_revision = processing.revision
            self.auto_queue_switch.blockSignals(True)
            self.auto_queue_switch.setChecked(
                processing.auto_enqueue_on_session_complete
            )
            self.auto_queue_switch.blockSignals(False)
            self._update_tasks(processing.jobs)
        self._resolve_runs()
        if self.isVisible():
            self._drain_command_results()

    def _refresh_tree(self, library: RecordingLibrary) -> None:
        sessions = tuple(
            session
            for session in library.sessions
            if session.state is CaptureSessionState.COMPLETE and session.clips
        )
        signature = tuple(
            (session.session_id, tuple(clip.clip_id for clip in session.clips))
            for session in sessions
        )
        if signature == self._library_signature:
            return
        self._library_signature = signature
        selected = self._selection
        self.session_tree.clear()
        restore: QTreeWidgetItem | None = None
        for session in sessions:
            title = session.display_name or session.session_id[:8]
            item = QTreeWidgetItem([title])
            item.setData(0, Qt.ItemDataRole.UserRole, (session.session_id, None))
            item.setToolTip(0, session.session_id)
            self.session_tree.addTopLevelItem(item)
            for index, clip in enumerate(session.clips, start=1):
                child = QTreeWidgetItem([f"片段 {index} · {clip.duration_ms / 1000:.1f}s"])
                child.setData(0, Qt.ItemDataRole.UserRole, (session.session_id, clip.clip_id))
                item.addChild(child)
                if selected == _Selection(session.session_id, clip.clip_id):
                    restore = child
            if selected == _Selection(session.session_id, None):
                restore = item
            item.setExpanded(True)
        if restore is None and self.session_tree.topLevelItemCount() > 0:
            restore = self.session_tree.topLevelItem(0)
        if restore is not None:
            self.session_tree.setCurrentItem(restore)
        else:
            self._selection = None
            self.workspace_status.setText("没有可处理的完整会话")
            self.workspace_status.setLevel(InfoLevel.INFOAMTION)

    def _select_tree_item(
        self,
        item: QTreeWidgetItem | None,
        _previous: QTreeWidgetItem | None,
    ) -> None:
        if item is None:
            return
        value = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(value, tuple) or len(value) != 2:
            return
        session_id, clip_id = value
        if not isinstance(session_id, str) or (
            clip_id is not None and not isinstance(clip_id, str)
        ):
            return
        self._selection = _Selection(session_id, clip_id)
        try:
            session_path = self.runtime.session_directory(session_id)
            self.replay.open_session(session_path, clip_id)
        except Exception as error:
            self._show_error(str(error))
            return
        self.workspace_status.setText(
            f"{session_id[:8]} · {'完整会话' if clip_id is None else clip_id[:8]}"
        )
        self.workspace_status.setLevel(InfoLevel.SUCCESS)
        self._run_future = self.runtime.processing_runs(session_id)
        self._clear_result_queries()

    def _start_processing(self) -> None:
        if self._selection is None:
            self._show_error("请先选择一个会话或片段")
            return
        preset_ids = (
            "hand-tracking-quality",
            "hand-tracking-balanced",
            "hand-tracking-preview",
        )
        self.runtime.request_processing(
            self._selection.session_id,
            clip_id=self._selection.clip_id,
            preset_id=preset_ids[max(0, self.preset_combo.currentIndex())],
        )

    def _cancel_processing(self) -> None:
        job_id = self._selected_job_id(active_fallback=True)
        if job_id is not None:
            self.runtime.request_processing_cancel(job_id)

    def _retry_processing(self) -> None:
        job_id = self._selected_job_id(active_fallback=False)
        if job_id is not None:
            self.runtime.request_processing_retry(job_id)

    def _export_result(self) -> None:
        frame = self.replay.snapshot().frame
        selection = self._selection
        run_id = self._run_ids.get(self.primary_run_combo.currentText())
        if frame is None or selection is None or run_id is None:
            self._show_error("请先选择已完成的处理结果和视频片段")
            return
        self.runtime.request_processing_export(
            selection.session_id,
            run_id,
            frame.clip_id,
        )

    def _selected_job_id(self, *, active_fallback: bool) -> str | None:
        row = self.task_table.currentRow()
        if 0 <= row < len(self._task_row_ids):
            return self._task_row_ids[row]
        processing = self.runtime.snapshot().processing
        return processing.active_job_id if active_fallback and processing is not None else None

    def _update_tasks(self, jobs: tuple[ProcessingJob, ...]) -> None:
        self._task_row_ids = [job.job_id for job in jobs]
        self.task_table.setRowCount(len(jobs))
        for row, job in enumerate(jobs):
            values = (
                job.session_id[:8],
                "完整会话" if job.clip_id is None else job.clip_id[:8],
                job.preset.display_name,
                _job_state_text(job.state),
                f"{job.progress_current}/{job.progress_total}",
                _duration_text(job.elapsed_seconds),
                job.detail,
            )
            for column, value in enumerate(values):
                self.task_table.setItem(row, column, QTableWidgetItem(value))
        active = next(
            (job for job in jobs if job.state in {
                ProcessingJobState.PREPARING,
                ProcessingJobState.RUNNING,
                ProcessingJobState.CANCELING,
            }),
            None,
        )
        self.task_summary.setText(
            "暂无任务"
            if not jobs
            else (
                f"{_job_state_text(active.state)} · "
                f"{active.progress_current}/{active.progress_total}"
                if active is not None
                else f"共 {len(jobs)} 个历史任务"
            )
        )
    def _query_results(self, frame: PlaybackFrame) -> None:
        selection = self._selection
        if selection is None:
            return
        for layer, combo in (
            ("primary", self.primary_run_combo),
            ("comparison", self.comparison_run_combo),
        ):
            label = combo.currentText()
            run_id = self._run_ids.get(label)
            if run_id is None or (layer == "comparison" and label == "关闭对比"):
                continue
            key = (frame.session_id, frame.clip_id, frame.frame_index, frame.session_time_ns)
            self._result_futures[layer] = (
                key,
                self.runtime.processing_result(
                    frame.session_id,
                    run_id,
                    frame.clip_id,
                    frame.frame_index,
                    frame.session_time_ns,
                ),
            )

    def _resolve_result_queries(self, frame_key: tuple[str, str, int, int]) -> None:
        for layer, item in tuple(self._result_futures.items()):
            key, future = item
            if not future.done():
                continue
            del self._result_futures[layer]
            try:
                result = future.result()
            except Exception as error:
                self.layer_status.setText(str(error))
                continue
            if key != frame_key:
                continue
            if layer == "primary":
                self.canvas.set_overlay(result)
                self.spatial_canvas.set_hand_result(result)
                self.frame_result.setText(_result_summary(result))
            else:
                self.canvas.set_comparison_overlay(result)

    def _clear_result_queries(self, _value: str = "") -> None:
        self._result_futures.clear()
        self.canvas.set_overlay(None)
        self.canvas.set_comparison_overlay(None)
        self.spatial_canvas.set_hand_result(None)
        if self._last_frame_key is not None:
            frame = self.replay.snapshot().frame
            if frame is not None:
                self._query_results(frame)

    def _resolve_runs(self) -> None:
        future = self._run_future
        if future is None or not future.done():
            return
        self._run_future = None
        try:
            runs = future.result()
        except Exception as error:
            self.layer_status.setText(str(error))
            return
        self._run_ids.clear()
        self.primary_run_combo.clear()
        self.comparison_run_combo.clear()
        self.comparison_run_combo.addItem("关闭对比")
        for run in runs:
            if not run.is_viewable:
                continue
            label = f"{run.preset.display_name} · {run.run_id[-8:]}"
            self._run_ids[label] = run.run_id
            self.primary_run_combo.addItem(label)
            self.comparison_run_combo.addItem(label)
        self.layer_status.setText(
            f"可用运行 {len(self._run_ids)} 个" if self._run_ids else "当前会话没有处理结果"
        )
        self._clear_result_queries()

    def _update_frame_inspector(
        self,
        frame: PlaybackFrame,
        snapshot: ReplaySnapshot,
    ) -> None:
        self.frame_identity.setText(f"{frame.clip_id[:8]} · F{frame.frame_index}")
        self.frame_timing.setText(
            f"PTS {frame.pts_ns / 1_000_000:.3f} ms\n"
            f"会话时间 {frame.session_time_ns / 1_000_000_000:.6f} s"
        )
        pose = snapshot.imu_pose
        self.frame_imu.setText(
            "IMU --"
            if pose is None
            else (
                f"IMU {pose.recent_rate_hz:.1f} Hz · "
                f"R {pose.roll_degrees:.1f}° · P {pose.pitch_degrees:.1f}° · "
                f"Y {pose.yaw_degrees:.1f}°"
            )
        )
    def _sync_transport(self, snapshot: ReplaySnapshot) -> None:
        replay = snapshot
        self.play_button.setIcon(
            FluentIcon.PAUSE if replay.state is ReplayState.PLAYING else FluentIcon.PLAY
        )
        maximum = max(1, round(replay.duration_seconds * 1000))
        value = min(maximum, round(replay.position_seconds * 1000))
        if not self._seeking:
            self.position_slider.blockSignals(True)
            self.position_slider.setRange(0, maximum)
            self.position_slider.setValue(value)
            self.position_slider.blockSignals(False)
        self.time_label.setText(
            f"{_clock(replay.position_seconds)} / {_clock(replay.duration_seconds)}"
        )
        if replay.state is ReplayState.ERROR and replay.error:
            self.workspace_status.setText("回放失败")
            self.workspace_status.setLevel(InfoLevel.ERROR)

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

    def _show_inspector_page(self, key: str) -> None:
        page = self.inspector_pages.get(key)
        if page is not None:
            self.inspector_stack.setCurrentWidget(page)

    def _toggle_tasks(self) -> None:
        visible = not self.task_table.isVisible()
        self.task_table.setVisible(visible)
        self.task_toggle.setIcon(FluentIcon.UP if visible else FluentIcon.DOWN)

    def _drain_command_results(self) -> None:
        for result in self.runtime.command_results():
            if result.succeeded:
                InfoBar.success("操作完成", result.detail, duration=1800, parent=self.window())
            else:
                self._show_error(result.detail)

    def _show_error(self, detail: str) -> None:
        InfoBar.error("操作失败", detail, duration=4500, parent=self.window())


def _job_state_text(state: ProcessingJobState) -> str:
    return {
        ProcessingJobState.QUEUED: "等待",
        ProcessingJobState.PREPARING: "校验",
        ProcessingJobState.RUNNING: "处理中",
        ProcessingJobState.CANCELING: "取消中",
        ProcessingJobState.COMPLETED: "完成",
        ProcessingJobState.FAILED: "失败",
        ProcessingJobState.INTERRUPTED: "中断",
        ProcessingJobState.CANCELED: "已取消",
    }[state]


def _result_summary(result: dict[str, object] | None) -> str:
    if result is None:
        return "当前帧没有手部结果"
    hands = result.get("hands")
    count = len(hands) if isinstance(hands, list) else 0
    duration = result.get("inference_duration_ns")
    duration_text = (
        f"{duration / 1_000_000:.1f} ms"
        if isinstance(duration, int) and not isinstance(duration, bool)
        else "--"
    )
    return f"检测手部 {count} · 推理 {duration_text}"


def _clock(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    minutes, remainder = divmod(milliseconds, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def _duration_text(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(round(seconds), 60)
    return f"{minutes:d}m {remainder:02d}s"
