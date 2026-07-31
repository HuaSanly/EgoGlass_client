from __future__ import annotations

import concurrent.futures
from enum import StrEnum
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    ComboBox,
    FluentIcon,
    HeaderCardWidget,
    InfoBadge,
    InfoBar,
    InfoLevel,
    PrimaryPushButton,
    ProgressRing,
    PushButton,
    SegmentedWidget,
    Slider,
    SmoothScrollArea,
    StateToolTip,
    StrongBodyLabel,
    TitleLabel,
    TransparentToolButton,
)

from ingest_gateway.recording_models import RecordingState
from ingest_gateway.webrtc_models import StreamControlAction, StreamControlState
from ui.replay.player import ReplayPlayer, ReplayState
from ui.runtime import UnifiedRuntimeHost
from ui.state import RuntimeSnapshot
from ui.widgets.spatial_sync_canvas import SpatialSyncCanvas
from ui.widgets.video_canvas import VideoCanvas


class ViewerMode(StrEnum):
    LIVE = "live"
    REPLAY = "replay"


class HomeView(QWidget):
    """Single 4:3 workspace shared by live display and stored replay."""

    def __init__(self, runtime: UnifiedRuntimeHost, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("homeView")
        self.setStyleSheet("#homeView { background: #f5f7fb; }")
        self.runtime = runtime
        self.replay = ReplayPlayer()
        self.viewer_mode = ViewerMode.LIVE
        self._closed = False
        self._live_session_id: str | None = None
        self._last_snapshot_revision = -1
        self._library_identity: int | None = None
        self._session_labels: dict[str, str] = {}
        self._clip_labels: dict[str, tuple[str, str]] = {}
        self._media_future: concurrent.futures.Future[Path | None] | None = None
        self._result_media_future: concurrent.futures.Future[Path] | None = None
        self._pending_replay_target: tuple[str, str] | None = None
        self._active_replay_target: tuple[str, str] | None = None
        self._last_replay_error: str | None = None
        self._last_replay_job_state = "idle"
        self._replay_state_tooltip: StateToolTip | None = None

        self._build()
        self._frame_timer = QTimer(self)
        self._frame_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._frame_timer.setInterval(16)
        self._frame_timer.timeout.connect(self._update_frame)
        self._frame_timer.start()
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(100)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start()

    def close_resources(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._frame_timer.stop()
        self._status_timer.stop()
        if self._replay_state_tooltip is not None:
            self._replay_state_tooltip.close()
            self._replay_state_tooltip = None
        self.replay.close()

    def set_viewer_mode(self, mode: ViewerMode) -> None:
        if mode is self.viewer_mode:
            return
        self.viewer_mode = mode
        self.mode_selector.setCurrentItem(mode.value)
        replay_mode = mode is ViewerMode.REPLAY
        self.replay_controls.setVisible(replay_mode)
        self.right_stack.setCurrentIndex(1 if replay_mode else 0)
        self._sync_context_badges()
        self.canvas.set_overlay(None)

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 22)
        root.setSpacing(14)
        root.addLayout(self._build_header())
        root.addLayout(self._build_mode_bar())

        content = QHBoxLayout()
        content.setSpacing(16)
        content.addWidget(self._build_video_column(), 1)
        self.sidebar = self._build_sidebar()
        content.addWidget(self.sidebar)
        root.addLayout(content, 1)

    def _build_header(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        title_column = QVBoxLayout()
        title_column.setSpacing(0)
        title_column.addWidget(TitleLabel("EgoGlass 感知工作台"))
        title_column.addWidget(CaptionLabel("眼镜实时感知、采集与离线回放"))
        layout.addLayout(title_column)
        layout.addStretch(1)

        self.connection_badge = InfoBadge("设备未连接", level=InfoLevel.INFOAMTION)
        self.resolution_badge = InfoBadge("-- × --", level=InfoLevel.INFOAMTION)
        self.fps_badge = InfoBadge("0.0 FPS", level=InfoLevel.INFOAMTION)
        self.inference_badge = InfoBadge("推理 --", level=InfoLevel.INFOAMTION)
        for badge in (
            self.connection_badge,
            self.resolution_badge,
            self.fps_badge,
            self.inference_badge,
        ):
            badge.setMinimumHeight(24)
            layout.addWidget(badge, 0, Qt.AlignmentFlag.AlignVCenter)
        return layout

    def _build_mode_bar(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(10)
        self.mode_selector = SegmentedWidget(self)
        self.mode_selector.addItem(
            ViewerMode.LIVE.value,
            "实时",
            lambda: self.set_viewer_mode(ViewerMode.LIVE),
            FluentIcon.CAMERA,
        )
        self.mode_selector.addItem(
            ViewerMode.REPLAY.value,
            "回放",
            lambda: self.set_viewer_mode(ViewerMode.REPLAY),
            FluentIcon.MOVIE,
        )
        self.mode_selector.setCurrentItem(ViewerMode.LIVE.value)
        layout.addWidget(self.mode_selector)
        self.mode_badge = InfoBadge("来源 · 实时", level=InfoLevel.SUCCESS)
        layout.addWidget(self.mode_badge, 0, Qt.AlignmentFlag.AlignVCenter)
        self.session_badge = InfoBadge("会话 · --", level=InfoLevel.INFOAMTION)
        layout.addWidget(self.session_badge, 0, Qt.AlignmentFlag.AlignVCenter)
        self.frame_detail = CaptionLabel("等待首帧")
        layout.addWidget(self.frame_detail, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addStretch(1)

        self.stream_button = _compact_button("开始视频", FluentIcon.PLAY, primary=True)
        self.stream_button.setObjectName("streamControlButton")
        self.stream_button.setProperty("action", "start")
        self.stream_button.clicked.connect(self._toggle_stream)
        layout.addWidget(self.stream_button)
        self.recording_button = _compact_button("开始录制", FluentIcon.VIDEO)
        self.recording_button.setObjectName("recordingControlButton")
        self.recording_button.setProperty("action", "start")
        self.recording_button.clicked.connect(self._toggle_recording)
        layout.addWidget(self.recording_button)
        self.session_button = _compact_button("新建会话", FluentIcon.ADD)
        self.session_button.setObjectName("sessionControlButton")
        self.session_button.clicked.connect(lambda: self.runtime.request_session("new"))
        layout.addWidget(self.session_button)
        self.recording_badge = InfoBadge("未录制", level=InfoLevel.INFOAMTION)
        self.recording_badge.setMinimumHeight(24)
        layout.addWidget(self.recording_badge, 0, Qt.AlignmentFlag.AlignVCenter)
        return layout

    def _build_video_column(self) -> QWidget:
        column = QWidget(self)
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.canvas = VideoCanvas(column)
        layout.addWidget(self.canvas, 1)

        self.replay_controls = QWidget(column)
        replay_layout = QHBoxLayout(self.replay_controls)
        replay_layout.setContentsMargins(4, 0, 4, 0)
        replay_layout.setSpacing(8)
        self.play_button = _tool_button(FluentIcon.PLAY, "播放或暂停")
        self.play_button.clicked.connect(self._toggle_replay)
        replay_layout.addWidget(self.play_button)
        self.step_button = _tool_button(FluentIcon.PAGE_RIGHT, "前进一帧")
        self.step_button.clicked.connect(self.replay.step)
        replay_layout.addWidget(self.step_button)
        self.replay_slider = Slider(Qt.Orientation.Horizontal, self.replay_controls)
        self.replay_slider.setRange(0, 1)
        self.replay_slider.sliderReleased.connect(self._seek_replay)
        replay_layout.addWidget(self.replay_slider, 1)
        self.replay_time = CaptionLabel("00:00 / 00:00")
        replay_layout.addWidget(self.replay_time)
        self.rate_combo = ComboBox(self.replay_controls)
        self.rate_combo.addItems(["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"])
        self.rate_combo.setCurrentText("1.0x")
        self.rate_combo.currentTextChanged.connect(self._set_replay_rate)
        self.rate_combo.setFixedWidth(92)
        replay_layout.addWidget(self.rate_combo)
        self.replay_controls.setVisible(False)
        layout.addWidget(self.replay_controls)
        return column

    def _build_sidebar(self) -> SmoothScrollArea:
        scroll = SmoothScrollArea(self)
        scroll.setObjectName("homeSidebar")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(380)
        scroll.setStyleSheet("#homeSidebar { border: none; background: transparent; }")
        scroll.viewport().setStyleSheet("background: transparent;")

        self.right_stack = QStackedWidget(scroll)
        self.right_stack.setStyleSheet("background: transparent;")
        self.right_stack.addWidget(self._build_live_sidebar())
        self.right_stack.addWidget(self._build_replay_sidebar())
        scroll.setWidget(self.right_stack)
        return scroll

    def _build_live_sidebar(self) -> QWidget:
        panel = QWidget(self)
        panel.setObjectName("liveSpatialPanel")
        panel.setStyleSheet(
            """
            QWidget#liveSpatialPanel {
                background: transparent;
            }
            QWidget#syncMetricBlock {
                background: #ffffff;
                border: 1px solid #e6ebf3;
                border-radius: 8px;
            }
            CaptionLabel#syncMetricTitle {
                color: #64748b;
                font-size: 11px;
            }
            BodyLabel#syncMetricBody {
                color: #1f2937;
                font-size: 12px;
            }
            StrongBodyLabel#syncSummaryTitle {
                color: #0f172a;
                font-size: 13px;
            }
            CaptionLabel#syncSummaryDetail {
                color: #64748b;
                font-size: 11px;
            }
            """
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(12)

        self.spatial_canvas = SpatialSyncCanvas(panel)
        layout.addWidget(self.spatial_canvas)

        data_card = HeaderCardWidget("同步数据", panel)
        data_card.setObjectName("spatialSyncDataCard")
        data_layout = QVBoxLayout()
        data_layout.setSpacing(10)

        self.perception_state = StrongBodyLabel("等待模型")
        self.perception_state.setObjectName("syncSummaryTitle")
        self.perception_detail = CaptionLabel("尚未收到识别结果")
        self.perception_detail.setObjectName("syncSummaryDetail")
        self.perception_detail.setWordWrap(True)

        summary_row = QVBoxLayout()
        summary_row.setSpacing(2)
        summary_row.addWidget(self.perception_state)
        summary_row.addWidget(self.perception_detail)
        data_layout.addLayout(summary_row)

        self.imu_sync_badge = InfoBadge("--", level=InfoLevel.INFOAMTION)
        self.left_pose_badge = InfoBadge("--", level=InfoLevel.INFOAMTION)
        self.right_pose_badge = InfoBadge("--", level=InfoLevel.INFOAMTION)
        self.link_sync_badge = InfoBadge("--", level=InfoLevel.INFOAMTION)
        self.imu_detail = BodyLabel("等待 IMU")
        self.left_confidence = BodyLabel("未检测到")
        self.right_confidence = BodyLabel("未检测到")
        self.frame_metrics = BodyLabel("等待视频链路")
        for detail in (
            self.imu_detail,
            self.left_confidence,
            self.right_confidence,
            self.frame_metrics,
        ):
            detail.setObjectName("syncMetricBody")
            detail.setWordWrap(True)

        metric_grid = QGridLayout()
        metric_grid.setHorizontalSpacing(8)
        metric_grid.setVerticalSpacing(8)
        metric_grid.addWidget(
            _sync_metric_block("IMU 姿态", self.imu_sync_badge, self.imu_detail),
            0,
            0,
        )
        metric_grid.addWidget(
            _sync_metric_block("左手位姿", self.left_pose_badge, self.left_confidence),
            0,
            1,
        )
        metric_grid.addWidget(
            _sync_metric_block("右手位姿", self.right_pose_badge, self.right_confidence),
            1,
            0,
        )
        metric_grid.addWidget(
            _sync_metric_block("帧链路", self.link_sync_badge, self.frame_metrics),
            1,
            1,
        )
        data_layout.addLayout(metric_grid)

        data_card.viewLayout.addLayout(data_layout)
        layout.addWidget(data_card)
        layout.addStretch(1)
        return panel

    def _build_replay_sidebar(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(12)

        resource_card = HeaderCardWidget("回放资源", panel)
        resource_layout = QVBoxLayout()
        resource_layout.setSpacing(10)
        resource_layout.addWidget(CaptionLabel("采集会话"))
        self.session_combo = ComboBox(resource_card)
        self.session_combo.currentTextChanged.connect(self._select_replay_session)
        resource_layout.addWidget(self.session_combo)
        resource_layout.addWidget(CaptionLabel("视频片段"))
        self.clip_combo = ComboBox(resource_card)
        resource_layout.addWidget(self.clip_combo)
        action_row = QHBoxLayout()
        self.refresh_button = _push_button("刷新", FluentIcon.SYNC)
        self.refresh_button.clicked.connect(self.runtime.request_library_refresh)
        action_row.addWidget(self.refresh_button)
        self.open_clip_button = _push_button("打开片段", FluentIcon.FOLDER)
        self.open_clip_button.clicked.connect(self._open_selected_clip)
        action_row.addWidget(self.open_clip_button)
        resource_layout.addLayout(action_row)
        resource_card.viewLayout.addLayout(resource_layout)
        layout.addWidget(resource_card)

        result_card = HeaderCardWidget("识别回放", panel)
        result_layout = QVBoxLayout()
        result_layout.setSpacing(10)
        self.replay_job_detail = BodyLabel("尚未生成识别回放")
        self.replay_job_detail.setWordWrap(True)
        progress_row = QHBoxLayout()
        self.replay_progress = ProgressRing(result_card)
        self.replay_progress.setFixedSize(30, 30)
        self.replay_progress.setRange(0, 1)
        self.replay_progress.setVisible(False)
        progress_row.addWidget(self.replay_progress)
        progress_row.addWidget(self.replay_job_detail, 1)
        result_layout.addLayout(progress_row)
        self.generate_button = _push_button(
            "生成识别回放",
            FluentIcon.ROBOT,
            primary=True,
        )
        self.generate_button.clicked.connect(self._generate_replay)
        result_layout.addWidget(self.generate_button)
        self.open_result_button = _push_button("播放识别结果", FluentIcon.MOVIE)
        self.open_result_button.clicked.connect(self._open_result_replay)
        self.open_result_button.setEnabled(False)
        result_layout.addWidget(self.open_result_button)
        result_card.viewLayout.addLayout(result_layout)
        layout.addWidget(result_card)

        playback_card = HeaderCardWidget("播放信息", panel)
        playback_layout = QVBoxLayout()
        self.playback_detail = BodyLabel("未打开视频")
        self.playback_detail.setWordWrap(True)
        playback_layout.addWidget(self.playback_detail)
        playback_card.viewLayout.addLayout(playback_layout)
        layout.addWidget(playback_card)
        layout.addStretch(1)
        return panel

    def _update_frame(self) -> None:
        self._resolve_media_futures()
        replay_snapshot = self.replay.snapshot()
        frame = (
            self.runtime.latest_frame()
            if self.viewer_mode is ViewerMode.LIVE
            else replay_snapshot.frame
        )
        if self.canvas.set_frame(frame) and frame is not None:
            self.frame_detail.setText(
                f"帧 {frame.frame_index:,} · RGB {frame.width}×{frame.height}"
            )
        if self.viewer_mode is ViewerMode.REPLAY:
            self._update_replay_controls(replay_snapshot)

    def _update_status(self) -> None:
        snapshot = self.runtime.snapshot()
        self._drain_command_results()
        self._refresh_replay_library(snapshot)
        self._update_replay_job(snapshot)
        if snapshot.revision == self._last_snapshot_revision:
            return
        self._last_snapshot_revision = snapshot.revision
        self._update_connection(snapshot)
        self._update_recording(snapshot)
        self._update_perception(snapshot)
        self._update_metrics(snapshot)
        self._update_imu(snapshot)

    def _update_connection(self, snapshot: RuntimeSnapshot) -> None:
        webrtc = snapshot.webrtc
        if webrtc is not None:
            phase = webrtc.phase.value
            connected = phase in {"connected", "streaming"}
            self.connection_badge.setText("设备已连接" if connected else f"连接 {phase}")
            self.connection_badge.setLevel(
                InfoLevel.SUCCESS
                if connected
                else InfoLevel.ERROR
                if phase == "failed"
                else InfoLevel.INFOAMTION
            )
            self.resolution_badge.setText(
                f"{webrtc.width or '--'} × {webrtc.height or '--'}"
            )
        control = snapshot.stream_control
        if control is None:
            return
        streaming = control.state in {
            StreamControlState.STARTING,
            StreamControlState.STREAMING,
        }
        self.stream_button.setText("停止视频" if streaming else "开始视频")
        self.stream_button.setIcon(FluentIcon.PAUSE if streaming else FluentIcon.PLAY)
        self.stream_button.setProperty("action", "stop" if streaming else "start")
        self.stream_button.setEnabled(control.state is not StreamControlState.UNAVAILABLE)

    def _update_recording(self, snapshot: RuntimeSnapshot) -> None:
        recording = snapshot.recording
        if recording is None:
            return
        active = recording.state in {RecordingState.COUNTDOWN, RecordingState.RECORDING}
        busy = recording.state is RecordingState.FINALIZING
        self.recording_button.setText("停止录制" if active else "开始录制")
        self.recording_button.setProperty("action", "stop" if active else "start")
        self.recording_button.setEnabled(
            recording.state is not RecordingState.UNAVAILABLE and not busy
        )
        self.recording_badge.setText(
            f"录制 {recording.recording_duration_ms / 1000:.1f}s" if active else "未录制"
        )
        self.recording_badge.setLevel(
            InfoLevel.ERROR if active else InfoLevel.INFOAMTION
        )
        self._live_session_id = recording.session_id
        if self.viewer_mode is ViewerMode.LIVE:
            self._sync_context_badges()

    def _update_perception(self, snapshot: RuntimeSnapshot) -> None:
        perception = snapshot.perception
        state = str(perception.get("state", "idle"))
        detail = str(perception.get("detail", "等待识别"))
        self.perception_state.setText(state.upper())
        self.perception_detail.setText(detail)
        latest = perception.get("latest_result")
        if not isinstance(latest, dict):
            self.spatial_canvas.set_hand_result(None)
            self.left_pose_badge.setText("--")
            self.left_pose_badge.setLevel(InfoLevel.INFOAMTION)
            self.right_pose_badge.setText("--")
            self.right_pose_badge.setLevel(InfoLevel.INFOAMTION)
            self.left_confidence.setText("未检测到")
            self.right_confidence.setText("未检测到")
            return
        if self.viewer_mode is ViewerMode.LIVE:
            self.canvas.set_overlay(latest)
        self.spatial_canvas.set_hand_result(latest)
        hands = latest.get("hands")
        left = _hand_for_side(hands, "left")
        right = _hand_for_side(hands, "right")
        self.left_confidence.setText(_confidence_body(left))
        self.right_confidence.setText(_confidence_body(right))
        self.left_pose_badge.setText("已同步" if left else "--")
        self.left_pose_badge.setLevel(InfoLevel.SUCCESS if left else InfoLevel.INFOAMTION)
        self.right_pose_badge.setText("已同步" if right else "--")
        self.right_pose_badge.setLevel(InfoLevel.SUCCESS if right else InfoLevel.INFOAMTION)
        hand_count = len(hands) if isinstance(hands, list) else 0
        self.perception_detail.setText(
            f"结果帧 {latest.get('frame_index', '--')} · 检出 {hand_count} 只手"
        )
        duration_ns = latest.get("inference_duration_ns")
        if isinstance(duration_ns, int):
            self.inference_badge.setText(f"推理 {duration_ns / 1_000_000:.1f} ms")

    def _update_metrics(self, snapshot: RuntimeSnapshot) -> None:
        display = snapshot.display
        if display is None:
            return
        canvas = self.canvas.status()
        self.fps_badge.setText(f"{canvas.recent_presentation_fps:.1f} FPS")
        self.link_sync_badge.setText(f"{display.presentation_fps:.1f} FPS")
        self.link_sync_badge.setLevel(
            InfoLevel.SUCCESS if display.presentation_fps >= 28 else InfoLevel.WARNING
        )
        self.frame_metrics.setText(
            "\n".join(
                (
                    f"接收 {display.frames_received:,} · RGB {display.frames_converted:,}",
                    f"呈现 {canvas.presented_frames:,} · "
                    f"{display.presentation_fps:.1f} FPS",
                    f"转换 {display.latest_conversion_ms or 0:.2f} ms · "
                    f"绘制 {canvas.latest_paint_ms or 0:.2f} ms",
                    f"队列 {display.presentation_queue_depth} · "
                    f"丢帧 {display.presentation_frames_dropped} · "
                    f"跳帧 "
                    f"{display.pending_frames_overwritten + canvas.source_frames_skipped}",
                )
            )
        )

    def _update_imu(self, snapshot: RuntimeSnapshot) -> None:
        pose = snapshot.imu_pose
        self.spatial_canvas.set_pose(pose)
        if pose is None or pose.samples_received == 0:
            self.imu_detail.setText("等待 IMU")
            self.imu_sync_badge.setText("--")
            self.imu_sync_badge.setLevel(InfoLevel.INFOAMTION)
            return
        self.imu_sync_badge.setText(f"{pose.recent_rate_hz:.1f} Hz")
        self.imu_sync_badge.setLevel(
            InfoLevel.SUCCESS if pose.recent_rate_hz >= 80 else InfoLevel.WARNING
        )
        self.imu_detail.setText(
            f"{pose.recent_rate_hz:.1f} Hz · "
            f"R {pose.roll_degrees:.1f}° · P {pose.pitch_degrees:.1f}° · "
            f"Y {pose.yaw_degrees:.1f}°"
        )

    def _refresh_replay_library(self, snapshot: RuntimeSnapshot) -> None:
        library = snapshot.library
        if library is None or id(library) == self._library_identity:
            return
        self._library_identity = id(library)
        current = self.session_combo.currentText()
        self._session_labels = {
            f"{session.display_name or '未命名会话'} · {session.session_id[:8]}": session.session_id
            for session in library.sessions
            if session.clips
        }
        labels = list(self._session_labels)
        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        self.session_combo.addItems(labels)
        self.session_combo.setCurrentText(current if current in self._session_labels else "")
        if labels and not self.session_combo.currentText():
            self.session_combo.setCurrentIndex(0)
        self.session_combo.blockSignals(False)
        self._select_replay_session()

    def _select_replay_session(self, _label: str = "") -> None:
        session_id = self._selected_session_id()
        library = self.runtime.snapshot().library
        self._clip_labels = {}
        if session_id is not None and library is not None:
            session = next(
                (item for item in library.sessions if item.session_id == session_id),
                None,
            )
            if session is not None:
                self._clip_labels = {
                    f"片段 {index + 1:02d} · {clip.width}×{clip.height}": (
                        session_id,
                        clip.clip_id,
                    )
                    for index, clip in enumerate(session.clips)
                }
        self.clip_combo.clear()
        self.clip_combo.addItems(list(self._clip_labels))
        if self._clip_labels:
            self.clip_combo.setCurrentIndex(0)
        if self.viewer_mode is ViewerMode.REPLAY:
            self._sync_context_badges()

    def _open_selected_clip(self) -> None:
        target = self._clip_labels.get(self.clip_combo.currentText())
        if target is None or self._media_future is not None:
            return
        self._pending_replay_target = target
        self._media_future = self.runtime.media_path(*target)

    def _generate_replay(self) -> None:
        session_id = self._selected_session_id()
        if session_id is not None:
            self.runtime.request_replay_generation(session_id)

    def _open_result_replay(self) -> None:
        if self._result_media_future is not None:
            return
        replay = self.runtime.snapshot().perception.get("replay", {})
        report = replay.get("report") if isinstance(replay, dict) else None
        videos = report.get("videos") if isinstance(report, dict) else None
        first_video = videos[0] if isinstance(videos, list) and videos else None
        if not isinstance(first_video, dict) or not isinstance(report, dict):
            return
        values = (report.get("session_id"), report.get("run_id"), first_video.get("clip_id"))
        if all(isinstance(value, str) for value in values):
            self._result_media_future = self.runtime.replay_video_path(*values)

    def _update_replay_job(self, snapshot: RuntimeSnapshot) -> None:
        replay = snapshot.perception.get("replay", {})
        if not isinstance(replay, dict):
            return
        state = str(replay.get("state", "idle"))
        detail = str(replay.get("detail", "尚未生成识别回放"))
        processed = replay.get("frames_processed", 0)
        total = replay.get("frame_total", 0)
        running = state == "running"
        self.replay_progress.setVisible(running)
        self.generate_button.setEnabled(not running)
        if isinstance(processed, int) and isinstance(total, int) and total > 0:
            self.replay_progress.setRange(0, total)
            self.replay_progress.setValue(min(processed, total))
            detail = f"{detail} · {processed}/{total}"
        self.replay_job_detail.setText(detail)
        self.open_result_button.setEnabled(isinstance(replay.get("report"), dict))
        self._update_replay_state_tooltip(state, detail)

    def _resolve_media_futures(self) -> None:
        if self._media_future is not None and self._media_future.done():
            future, self._media_future = self._media_future, None
            try:
                path = future.result()
                if path is not None:
                    self.replay.open(path)
                    self._active_replay_target = self._pending_replay_target
                    self.set_viewer_mode(ViewerMode.REPLAY)
                self._pending_replay_target = None
            except Exception as error:
                self._pending_replay_target = None
                self._show_error(str(error))
        if self._result_media_future is not None and self._result_media_future.done():
            future, self._result_media_future = self._result_media_future, None
            try:
                self.replay.open(future.result())
                self._active_replay_target = None
                self.set_viewer_mode(ViewerMode.REPLAY)
            except Exception as error:
                self._show_error(str(error))

    def _update_replay_controls(self, snapshot: object) -> None:
        if not hasattr(snapshot, "state"):
            return
        replay = self.replay.snapshot()
        self.play_button.setIcon(
            FluentIcon.PAUSE if replay.state is ReplayState.PLAYING else FluentIcon.PLAY
        )
        maximum = max(1, round(replay.duration_seconds * 1000))
        value = min(maximum, round(replay.position_seconds * 1000))
        self.replay_slider.blockSignals(True)
        self.replay_slider.setRange(0, maximum)
        self.replay_slider.setValue(value)
        self.replay_slider.blockSignals(False)
        self.replay_time.setText(
            f"{_clock(replay.position_seconds)} / {_clock(replay.duration_seconds)}"
        )
        path_name = replay.path.name if replay.path is not None else "未打开视频"
        self.playback_detail.setText(
            f"{path_name}\n状态：{replay.state.value}\n"
            f"位置：{replay.position_seconds:.2f}s / {replay.duration_seconds:.2f}s"
        )
        if replay.state is ReplayState.ERROR and replay.error != self._last_replay_error:
            self._last_replay_error = replay.error
            self._show_error(replay.error or "回放失败")

    def _toggle_replay(self) -> None:
        if self.replay.snapshot().state is ReplayState.PLAYING:
            self.replay.pause()
        else:
            self.replay.play()

    def _seek_replay(self) -> None:
        self.replay.seek(self.replay_slider.value() / 1000)

    def _set_replay_rate(self, value: str) -> None:
        if value:
            self.replay.set_playback_rate(float(value.removesuffix("x")))

    def _toggle_stream(self) -> None:
        action = str(self.stream_button.property("action") or "start")
        self.runtime.request_stream(StreamControlAction(action))

    def _toggle_recording(self) -> None:
        action = str(self.recording_button.property("action") or "start")
        self.runtime.request_recording(action)

    def _drain_command_results(self) -> None:
        for result in self.runtime.command_results():
            if result.succeeded:
                if result.name == "generate-replay":
                    continue
                InfoBar.success(
                    "操作完成",
                    result.detail,
                    duration=2200,
                    parent=self.window(),
                )
            else:
                self._show_error(result.detail)

    def _show_error(self, detail: str) -> None:
        InfoBar.error("操作失败", detail, duration=4500, parent=self.window())

    def _selected_session_id(self) -> str | None:
        return self._session_labels.get(self.session_combo.currentText())

    def _sync_context_badges(self) -> None:
        replay_mode = self.viewer_mode is ViewerMode.REPLAY
        self.mode_badge.setText("来源 · 回放" if replay_mode else "来源 · 实时")
        self.mode_badge.setLevel(InfoLevel.INFOAMTION if replay_mode else InfoLevel.SUCCESS)
        session_id = self._selected_session_id() if replay_mode else self._live_session_id
        self.session_badge.setText(f"会话 · {session_id[:8]}" if session_id else "会话 · --")

    def _update_replay_state_tooltip(self, state: str, detail: str) -> None:
        previous = self._last_replay_job_state
        self._last_replay_job_state = state
        if state == "running":
            if self._replay_state_tooltip is None:
                tooltip = StateToolTip("正在生成识别回放", detail, self.window())
                tooltip.move(tooltip.getSuitablePos())
                tooltip.show()
                self._replay_state_tooltip = tooltip
            else:
                self._replay_state_tooltip.setContent(detail)
            return
        if previous != "running":
            return
        tooltip, self._replay_state_tooltip = self._replay_state_tooltip, None
        if state == "complete":
            if tooltip is not None:
                tooltip.setContent(detail)
                tooltip.setState(True)
            InfoBar.success("识别回放已生成", detail, duration=3000, parent=self.window())
        elif state == "error":
            if tooltip is not None:
                tooltip.close()
            self._show_error(detail)


def _push_button(
    text: str,
    icon: FluentIcon,
    *,
    primary: bool = False,
) -> PushButton:
    button = PrimaryPushButton() if primary else PushButton()
    button.setText(text)
    button.setIcon(icon)
    button.setMinimumHeight(36)
    button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return button


def _sync_metric_block(title: str, badge: InfoBadge, detail: BodyLabel) -> QWidget:
    block = QWidget()
    block.setObjectName("syncMetricBlock")
    block.setMinimumHeight(118)
    block_layout = QVBoxLayout(block)
    block_layout.setContentsMargins(10, 8, 10, 9)
    block_layout.setSpacing(5)

    header = QHBoxLayout()
    header.setContentsMargins(0, 0, 0, 0)
    header.setSpacing(6)
    label = CaptionLabel(title)
    label.setObjectName("syncMetricTitle")
    header.addWidget(label)
    header.addStretch(1)
    badge.setMinimumHeight(22)
    header.addWidget(badge, 0, Qt.AlignmentFlag.AlignRight)
    block_layout.addLayout(header)
    block_layout.addWidget(detail, 1)
    return block


def _compact_button(
    text: str,
    icon: FluentIcon,
    *,
    primary: bool = False,
) -> PushButton:
    button = PrimaryPushButton() if primary else PushButton()
    button.setText(text)
    button.setIcon(icon)
    button.setMinimumHeight(34)
    button.setMaximumHeight(34)
    button.setMinimumWidth(104)
    button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return button


def _tool_button(icon: FluentIcon, tooltip: str) -> TransparentToolButton:
    button = TransparentToolButton()
    button.setIcon(icon)
    button.setToolTip(tooltip)
    button.setFixedSize(36, 36)
    return button


def _hand_for_side(hands: object, side: str) -> dict[str, object] | None:
    if not isinstance(hands, list):
        return None
    return next(
        (
            hand
            for hand in hands
            if isinstance(hand, dict) and hand.get("handedness") == side
        ),
        None,
    )


def _confidence_text(label: str, hand: dict[str, object] | None) -> str:
    if hand is None:
        return f"{label}\n未检测到"
    fields = (
        ("检测", "detector_confidence"),
        ("重建", "reconstruction_quality"),
        ("深度", "depth_score"),
        ("覆盖", "coverage_score"),
        ("紧致", "compactness_score"),
        ("最终", "final_confidence"),
    )
    values = []
    for name, field in fields:
        value = hand.get(field)
        values.append(
            f"{name} {float(value):.2f}"
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else f"{name} --"
        )
    return f"{label}\n" + "  ·  ".join(values)


def _confidence_body(hand: dict[str, object] | None) -> str:
    if hand is None:
        return "未检测到\n等待 3D keypoints"
    fields = (
        ("检测", "detector_confidence"),
        ("重建", "reconstruction_quality"),
        ("深度", "depth_score"),
        ("覆盖", "coverage_score"),
        ("紧致", "compactness_score"),
        ("最终", "final_confidence"),
    )
    values = []
    for name, field in fields:
        value = hand.get(field)
        values.append(
            f"{name} {float(value):.2f}"
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else f"{name} --"
        )
    return "\n".join((" · ".join(values[:3]), " · ".join(values[3:])))


def _clock(seconds: float) -> str:
    total = max(0, round(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"
