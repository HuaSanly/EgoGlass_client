from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CaptionLabel,
    FluentIcon,
    HeaderCardWidget,
    InfoBar,
    InfoLevel,
    PrimaryPushButton,
    PushButton,
    SimpleCardWidget,
    SmoothScrollArea,
    StrongBodyLabel,
    TitleLabel,
    ToggleToolButton,
)

from ingest_gateway.recording_models import RecordingState
from ingest_gateway.webrtc_models import StreamControlAction, StreamControlState
from ui.runtime import UnifiedRuntimeHost
from ui.state import RuntimeSnapshot
from ui.widgets.spatial_sync_canvas import SpatialSyncCanvas
from ui.widgets.status_indicator import StatusIndicator
from ui.widgets.video_canvas import VideoCanvas


class HomeView(QWidget):
    """Live capture, optional inference, and device synchronization view."""

    def __init__(self, runtime: UnifiedRuntimeHost, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("homeView")
        self.setStyleSheet("#homeView { background: #f5f7fb; }")
        self.runtime = runtime
        self._closed = False
        self._live_session_id: str | None = None
        self._last_snapshot_revision = -1

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
        layout.addWidget(TitleLabel("EgoGlass"))
        layout.addStretch(1)

        self.connection_badge = StatusIndicator(
            "设备未连接", FluentIcon.CONNECT, self
        )
        self.resolution_badge = StatusIndicator("-- × --", FluentIcon.FIT_PAGE, self)
        self.fps_badge = StatusIndicator("0.0 FPS", FluentIcon.SPEED_HIGH, self)
        self.inference_badge = StatusIndicator("推理 --", FluentIcon.ROBOT, self)
        for indicator in (
            self.connection_badge,
            self.resolution_badge,
            self.fps_badge,
            self.inference_badge,
        ):
            layout.addWidget(indicator, 0, Qt.AlignmentFlag.AlignVCenter)
        return layout

    def _build_mode_bar(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(10)
        layout.addWidget(StrongBodyLabel("实时采集", self))
        self.mode_badge = StatusIndicator(
            "来源 · 实时",
            FluentIcon.CAMERA,
            self,
            level=InfoLevel.SUCCESS,
        )
        layout.addWidget(self.mode_badge, 0, Qt.AlignmentFlag.AlignVCenter)
        self.session_badge = StatusIndicator("会话 · --", FluentIcon.LIBRARY, self)
        layout.addWidget(self.session_badge, 0, Qt.AlignmentFlag.AlignVCenter)
        self.frame_detail = CaptionLabel("等待首帧")
        layout.addWidget(self.frame_detail, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addStretch(1)

        self.live_inference_button = ToggleToolButton(FluentIcon.ROBOT, self)
        self.live_inference_button.setToolTip("开启实时手部推理")
        self.live_inference_button.setFixedSize(36, 36)
        self.live_inference_button.toggled.connect(self.runtime.request_live_inference)
        layout.addWidget(self.live_inference_button)

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
        self.recording_badge = StatusIndicator("未录制", FluentIcon.VIDEO, self)
        layout.addWidget(self.recording_badge, 0, Qt.AlignmentFlag.AlignVCenter)
        return layout

    def _build_video_column(self) -> QWidget:
        column = QWidget(self)
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.canvas = VideoCanvas(column)
        layout.addWidget(self.canvas, 1)

        self.frame_link_strip = self._build_frame_link_strip(column)
        layout.addWidget(self.frame_link_strip)
        return column

    def _build_frame_link_strip(self, parent: QWidget) -> SimpleCardWidget:
        card = SimpleCardWidget(parent)
        card.setObjectName("frameLinkStrip")
        card.setFixedHeight(78)
        layout = QHBoxLayout(card)
        layout.setContentsMargins(14, 9, 14, 9)
        layout.setSpacing(12)

        heading = QVBoxLayout()
        heading.setSpacing(3)
        heading.addWidget(StrongBodyLabel("帧链路", card))
        heading.addWidget(CaptionLabel("实时传输", card))
        heading.addStretch(1)
        layout.addLayout(heading)
        layout.addWidget(_vertical_separator(card))

        self.frame_input_value = StrongBodyLabel("0 / 0", card)
        self.frame_input_detail = CaptionLabel("接收帧 / RGB 帧", card)
        self.frame_present_value = StrongBodyLabel("0.0 FPS", card)
        self.frame_present_detail = CaptionLabel("呈现 0 帧", card)
        self.frame_latency_value = StrongBodyLabel("0.00 / 0.00 ms", card)
        self.frame_latency_detail = CaptionLabel("转换 / 绘制", card)
        self.frame_buffer_value = StrongBodyLabel("0 / 0 / 0", card)
        self.frame_buffer_detail = CaptionLabel("队列 / 丢帧 / 跳帧 · 覆盖 --", card)
        for title, value, detail in (
            ("输入", self.frame_input_value, self.frame_input_detail),
            ("呈现", self.frame_present_value, self.frame_present_detail),
            ("时延", self.frame_latency_value, self.frame_latency_detail),
            ("缓冲", self.frame_buffer_value, self.frame_buffer_detail),
        ):
            layout.addLayout(_frame_metric_cell(title, value, detail, card), 1)
        return card

    def _build_sidebar(self) -> SmoothScrollArea:
        scroll = SmoothScrollArea(self)
        scroll.setObjectName("homeSidebar")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(380)
        scroll.setStyleSheet("#homeSidebar { border: none; background: transparent; }")
        scroll.viewport().setStyleSheet("background: transparent;")

        scroll.setWidget(self._build_live_sidebar())
        return scroll

    def _build_live_sidebar(self) -> QWidget:
        panel = QWidget(self)
        panel.setObjectName("liveSpatialPanel")
        panel.setStyleSheet("QWidget#liveSpatialPanel { background: transparent; }")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.spatial_canvas = SpatialSyncCanvas(panel)
        self.spatial_canvas.reset_pose_requested.connect(self.runtime.request_imu_pose_reset)
        layout.addWidget(self.spatial_canvas)

        self.sync_data_card = HeaderCardWidget("同步数据", panel)
        self.sync_data_card.setObjectName("spatialSyncDataCard")
        self.sync_data_card.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        data_layout = QVBoxLayout()
        data_layout.setSpacing(8)

        summary_header = QHBoxLayout()
        summary_header.setContentsMargins(0, 0, 0, 0)
        summary_header.addWidget(StrongBodyLabel("感知模型", self.sync_data_card))
        summary_header.addStretch(1)
        self.perception_state = StatusIndicator(
            "等待", FluentIcon.ROBOT, self.sync_data_card
        )
        summary_header.addWidget(self.perception_state)
        data_layout.addLayout(summary_header)
        self.perception_detail = CaptionLabel("尚未收到识别结果")
        self.perception_detail.setWordWrap(True)
        data_layout.addWidget(self.perception_detail)
        data_layout.addWidget(_horizontal_separator(self.sync_data_card))

        self.imu_sync_badge = StatusIndicator(
            "--", FluentIcon.ROTATE, self.sync_data_card
        )
        self.left_pose_badge = StatusIndicator(
            "--", FluentIcon.FINGERPRINT, self.sync_data_card
        )
        self.right_pose_badge = StatusIndicator(
            "--", FluentIcon.FINGERPRINT, self.sync_data_card
        )
        self.imu_detail = CaptionLabel("等待 IMU")
        self.left_confidence = CaptionLabel("未检测到 · 等待 3D 关键点")
        self.right_confidence = CaptionLabel("未检测到 · 等待 3D 关键点")
        for detail in (
            self.imu_detail,
            self.left_confidence,
            self.right_confidence,
        ):
            detail.setWordWrap(True)

        self.imu_status_row = _sync_status_row(
            "IMU 姿态",
            self.imu_sync_badge,
            self.imu_detail,
            self.sync_data_card,
        )
        self.left_status_row = _sync_status_row(
            "左手位姿",
            self.left_pose_badge,
            self.left_confidence,
            self.sync_data_card,
        )
        self.right_status_row = _sync_status_row(
            "右手位姿",
            self.right_pose_badge,
            self.right_confidence,
            self.sync_data_card,
        )
        data_layout.addWidget(self.imu_status_row)
        data_layout.addWidget(_horizontal_separator(self.sync_data_card))
        data_layout.addWidget(self.left_status_row)
        data_layout.addWidget(_horizontal_separator(self.sync_data_card))
        data_layout.addWidget(self.right_status_row)

        self.sync_data_card.viewLayout.addLayout(data_layout)
        layout.addWidget(self.sync_data_card)
        layout.addStretch(1)
        return panel

    def _update_frame(self) -> None:
        frame = self.runtime.latest_frame()
        if self.canvas.set_frame(frame) and frame is not None:
            self.frame_detail.setText(
                f"帧 {frame.frame_index:,} · RGB {frame.width}×{frame.height}"
            )
        result = self.runtime.take_latest_perception_result()
        if result is not None:
            self.canvas.set_overlay(result)

    def _update_status(self) -> None:
        snapshot = self.runtime.snapshot()
        if self.isVisible():
            self._drain_command_results()
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
        self._sync_context_badges()

    def _update_perception(self, snapshot: RuntimeSnapshot) -> None:
        perception = snapshot.perception
        state = str(perception.get("state", "idle"))
        detail = str(perception.get("detail", "等待识别"))
        live_enabled = perception.get("live_enabled") is True
        offline_processing = perception.get("offline_processing") is True
        self.live_inference_button.blockSignals(True)
        self.live_inference_button.setChecked(live_enabled)
        self.live_inference_button.blockSignals(False)
        self.live_inference_button.setEnabled(not offline_processing)
        self.live_inference_button.setToolTip(
            "离线处理正在独占 GPU"
            if offline_processing
            else ("关闭实时手部推理" if live_enabled else "开启实时手部推理")
        )
        state_text, state_level = _perception_state_presentation(state)
        self.perception_state.setText(state_text)
        self.perception_state.setLevel(state_level)
        self.perception_detail.setText(detail)
        latest = perception.get("latest_result")
        if not isinstance(latest, dict):
            self.spatial_canvas.set_hand_result(None)
            self.left_pose_badge.setText("--")
            self.left_pose_badge.setLevel(InfoLevel.INFOAMTION)
            self.right_pose_badge.setText("--")
            self.right_pose_badge.setLevel(InfoLevel.INFOAMTION)
            self.left_confidence.setText(_confidence_body(None))
            self.right_confidence.setText(_confidence_body(None))
            return
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
        self.frame_input_value.setText(
            f"{display.frames_received:,} / {display.frames_converted:,}"
        )
        self.frame_present_value.setText(f"{display.presentation_fps:.1f} FPS")
        self.frame_present_detail.setText(f"呈现 {canvas.presented_frames:,} 帧")
        self.frame_latency_value.setText(
            f"{display.latest_conversion_ms or 0:.2f} / "
            f"{canvas.latest_paint_ms or 0:.2f} ms"
        )
        skipped = display.pending_frames_overwritten + canvas.source_frames_skipped
        self.frame_buffer_value.setText(
            f"{display.presentation_queue_depth} / "
            f"{display.presentation_frames_dropped} / {skipped}"
        )
        overlay = (
            f"Δ{canvas.overlay_frame_age} 帧"
            if canvas.overlay_visible
            else "--"
        )
        self.frame_buffer_detail.setText(f"队列 / 丢帧 / 跳帧 · 覆盖 {overlay}")

    def _update_imu(self, snapshot: RuntimeSnapshot) -> None:
        pose = snapshot.imu_pose
        self.spatial_canvas.set_pose(pose)
        if pose is None or pose.samples_received == 0:
            self.imu_detail.setText("等待 IMU")
            self.imu_sync_badge.setText("--")
            self.imu_sync_badge.setLevel(InfoLevel.INFOAMTION)
            return
        self.imu_sync_badge.setText("已同步")
        self.imu_sync_badge.setLevel(
            InfoLevel.SUCCESS if pose.recent_rate_hz >= 80 else InfoLevel.WARNING
        )
        self.imu_detail.setText(
            f"{pose.recent_rate_hz:.1f} Hz · "
            f"R {pose.roll_degrees:.1f}° · P {pose.pitch_degrees:.1f}° · "
            f"Y {pose.yaw_degrees:.1f}°"
        )

    def _toggle_stream(self) -> None:
        action = str(self.stream_button.property("action") or "start")
        self.runtime.request_stream(StreamControlAction(action))

    def _toggle_recording(self) -> None:
        action = str(self.recording_button.property("action") or "start")
        self.runtime.request_recording(action)

    def _drain_command_results(self) -> None:
        for result in self.runtime.command_results():
            if result.succeeded:
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

    def _sync_context_badges(self) -> None:
        session_id = self._live_session_id
        self.session_badge.setText(f"会话 · {session_id[:8]}" if session_id else "会话 · --")


def _vertical_separator(parent: QWidget) -> QFrame:
    separator = QFrame(parent)
    separator.setFrameShape(QFrame.Shape.VLine)
    separator.setFrameShadow(QFrame.Shadow.Plain)
    separator.setStyleSheet("color: #e5e7eb;")
    return separator


def _horizontal_separator(parent: QWidget) -> QFrame:
    separator = QFrame(parent)
    separator.setFrameShape(QFrame.Shape.HLine)
    separator.setFrameShadow(QFrame.Shadow.Plain)
    separator.setStyleSheet("color: #e5e7eb;")
    return separator


def _frame_metric_cell(
    title: str,
    value: StrongBodyLabel,
    detail: CaptionLabel,
    parent: QWidget,
) -> QVBoxLayout:
    layout = QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(1)
    layout.addWidget(CaptionLabel(title, parent))
    layout.addWidget(value)
    layout.addWidget(detail)
    return layout


def _sync_status_row(
    title: str,
    badge: StatusIndicator,
    detail: CaptionLabel,
    parent: QWidget,
) -> QWidget:
    row = QWidget(parent)
    row.setObjectName("syncStatusRow")
    row.setMinimumHeight(58)
    row_layout = QVBoxLayout(row)
    row_layout.setContentsMargins(0, 2, 0, 2)
    row_layout.setSpacing(4)
    header = QHBoxLayout()
    header.setContentsMargins(0, 0, 0, 0)
    header.setSpacing(6)
    header.addWidget(StrongBodyLabel(title, row))
    header.addStretch(1)
    header.addWidget(badge, 0, Qt.AlignmentFlag.AlignRight)
    row_layout.addLayout(header)
    row_layout.addWidget(detail)
    return row


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


def _perception_state_presentation(state: str) -> tuple[str, InfoLevel]:
    return {
        "disabled": ("已停用", InfoLevel.INFOAMTION),
        "idle": ("等待", InfoLevel.INFOAMTION),
        "loading": ("加载中", InfoLevel.WARNING),
        "ready": ("就绪", InfoLevel.SUCCESS),
        "error": ("异常", InfoLevel.ERROR),
    }.get(state.lower(), (state.upper(), InfoLevel.INFOAMTION))


def _confidence_body(hand: dict[str, object] | None) -> str:
    if hand is None:
        return "未检测到 · 等待 3D 关键点"
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
