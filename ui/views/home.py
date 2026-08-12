from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    CaptionLabel,
    FluentIcon,
    InfoBar,
    InfoLevel,
    PrimaryPushButton,
    PushButton,
    SimpleCardWidget,
    StrongBodyLabel,
    TitleLabel,
)

from ui.gateway.webrtc_models import StreamControlAction
from ui.widgets.imu_monitor import ImuChartSample, ImuMonitorStats, ImuMonitorWidget
from ui.widgets.status_indicator import StatusIndicator
from ui.widgets.video_canvas import VideoCanvas

if TYPE_CHECKING:
    from ui.application.runtime_host import UnifiedRuntimeHost


class HomeView(QWidget):
    """Recording console for live RGB and raw IMU capture."""

    def __init__(
        self,
        runtime: UnifiedRuntimeHost,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("homeView")
        self.setStyleSheet("#homeView { background: #f5f7fb; }")
        self.runtime = runtime
        self._closed = False
        self._last_snapshot_revision = -1
        self._last_imu_sequence = 0
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
        layout.addWidget(TitleLabel("EgoGlass 录制台", self))
        layout.addStretch(1)

        self.connection_badge = StatusIndicator(
            "设备未连接",
            FluentIcon.CONNECT,
            self,
        )
        self.resolution_badge = StatusIndicator("-- × --", FluentIcon.FIT_PAGE, self)
        self.fps_badge = StatusIndicator("0.0 FPS", FluentIcon.SPEED_HIGH, self)
        self.recording_badge = StatusIndicator("未录制", FluentIcon.VIDEO, self)
        for indicator in (
            self.connection_badge,
            self.resolution_badge,
            self.fps_badge,
            self.recording_badge,
        ):
            layout.addWidget(indicator, 0, Qt.AlignmentFlag.AlignVCenter)
        return layout

    def _build_mode_bar(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(10)
        layout.addWidget(StrongBodyLabel("实时采集", self))
        self.recording_id_badge = StatusIndicator(
            "录制 --",
            FluentIcon.LIBRARY,
            self,
        )
        layout.addWidget(self.recording_id_badge, 0, Qt.AlignmentFlag.AlignVCenter)
        self.recording_detail = CaptionLabel("等待开始录制", self)
        layout.addWidget(self.recording_detail, 0, Qt.AlignmentFlag.AlignVCenter)
        self.frame_detail = CaptionLabel("等待首帧", self)
        layout.addWidget(self.frame_detail, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addStretch(1)

        self.stream_button = _compact_button(
            "开始视频",
            FluentIcon.PLAY,
            primary=True,
        )
        self.stream_button.setObjectName("streamControlButton")
        self.stream_button.setProperty("action", "start")
        self.stream_button.clicked.connect(self._toggle_stream)
        layout.addWidget(self.stream_button)

        self.recording_button = _compact_button("开始录制", FluentIcon.VIDEO)
        self.recording_button.setObjectName("recordingControlButton")
        self.recording_button.setProperty("action", "start")
        self.recording_button.clicked.connect(self._toggle_recording)
        layout.addWidget(self.recording_button)
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
        heading.addWidget(StrongBodyLabel("视频链路", card))
        heading.addWidget(CaptionLabel("实时传输", card))
        heading.addStretch(1)
        layout.addLayout(heading)

        self.frame_input_value = StrongBodyLabel("0 / 0", card)
        self.frame_present_value = StrongBodyLabel("0.0 FPS", card)
        self.frame_latency_value = StrongBodyLabel("0.00 / 0.00 ms", card)
        self.frame_buffer_value = StrongBodyLabel("0 / 0 / 0", card)
        for title, value, detail in (
            ("输入", self.frame_input_value, "接收帧 / RGB 帧"),
            ("呈现", self.frame_present_value, "画布帧率"),
            ("时延", self.frame_latency_value, "转换 / 绘制"),
            ("缓冲", self.frame_buffer_value, "队列 / 丢帧 / 跳帧"),
        ):
            cell = QVBoxLayout()
            cell.setSpacing(2)
            cell.addWidget(CaptionLabel(title, card))
            cell.addWidget(value)
            cell.addWidget(CaptionLabel(detail, card))
            layout.addLayout(cell, 1)
        return card

    def _build_sidebar(self) -> ImuMonitorWidget:
        self.imu_monitor = ImuMonitorWidget(self)
        self.imu_monitor.setObjectName("homeImuMonitor")
        self.imu_monitor.setFixedWidth(420)
        return self.imu_monitor

    def _update_frame(self) -> None:
        frame = self.runtime.latest_frame()
        if self.canvas.set_frame(frame) and frame is not None:
            self.frame_detail.setText(
                f"帧 {frame.frame_index:,}  ·  RGB {frame.width}×{frame.height}"
            )

    def _update_status(self) -> None:
        snapshot = self.runtime.snapshot()
        if self.isVisible():
            self._drain_command_results()
        self._drain_imu_samples(snapshot)
        revision = int(getattr(snapshot, "revision", 0))
        if revision == self._last_snapshot_revision:
            return
        self._last_snapshot_revision = revision
        self._update_connection(snapshot)
        self._update_recording(snapshot)
        self._update_metrics(snapshot)
        self._update_imu(snapshot)

    def _update_connection(self, snapshot: object) -> None:
        webrtc = getattr(snapshot, "webrtc", None)
        if webrtc is not None:
            phase = _enum_value(getattr(webrtc, "phase", "idle"))
            connected = phase in {"connected", "streaming"}
            self.connection_badge.setText(
                "设备已连接" if connected else f"连接 {phase}"
            )
            self.connection_badge.setLevel(
                InfoLevel.SUCCESS
                if connected
                else InfoLevel.ERROR
                if phase == "failed"
                else InfoLevel.INFOAMTION
            )
            width = getattr(webrtc, "width", None) or "--"
            height = getattr(webrtc, "height", None) or "--"
            self.resolution_badge.setText(f"{width} × {height}")

        control = getattr(snapshot, "stream_control", None)
        if control is None:
            return
        state = _enum_value(getattr(control, "state", "unavailable"))
        streaming = state in {"starting", "streaming"}
        self.stream_button.setText("停止视频" if streaming else "开始视频")
        self.stream_button.setIcon(FluentIcon.PAUSE if streaming else FluentIcon.PLAY)
        self.stream_button.setProperty("action", "stop" if streaming else "start")
        self.stream_button.setEnabled(state != "unavailable")

    def _update_recording(self, snapshot: object) -> None:
        recording = getattr(snapshot, "recording", None)
        if recording is None:
            return
        state = _enum_value(getattr(recording, "state", "unavailable"))
        active = state in {"countdown", "recording"}
        busy = state == "finalizing"
        recording_id = getattr(recording, "recording_id", None)
        duration_ms = int(
            _first_attr(recording, "duration_ms", "recording_duration_ms", default=0)
        )
        frame_count = int(getattr(recording, "frame_count", 0) or 0)
        imu_rows = int(
            _first_attr(recording, "imu_row_count", "imu_sample_count", default=0)
        )

        self.recording_button.setText("停止录制" if active else "开始录制")
        self.recording_button.setProperty("action", "stop" if active else "start")
        self.recording_button.setEnabled(state != "unavailable" and not busy)
        self.recording_badge.setText(
            f"录制 {_clock_ms(duration_ms)}" if active else "正在完成" if busy else "未录制"
        )
        self.recording_badge.setLevel(
            InfoLevel.ERROR if active else InfoLevel.ATTENTION if busy else InfoLevel.INFOAMTION
        )
        self.recording_id_badge.setText(
            f"录制 {str(recording_id)[:8]}" if recording_id else "录制 --"
        )
        self.recording_detail.setText(
            f"{_clock_ms(duration_ms)}  ·  {frame_count:,} 帧  ·  IMU {imu_rows:,} 行"
            if active or busy
            else "等待开始录制"
        )

    def _update_metrics(self, snapshot: object) -> None:
        display = getattr(snapshot, "display", None)
        if display is None:
            return
        canvas = self.canvas.status()
        self.fps_badge.setText(f"{canvas.recent_presentation_fps:.1f} FPS")
        self.frame_input_value.setText(
            f"{int(getattr(display, 'frames_received', 0)):,} / "
            f"{int(getattr(display, 'frames_converted', 0)):,}"
        )
        self.frame_present_value.setText(
            f"{float(getattr(display, 'presentation_fps', 0.0)):.1f} FPS"
        )
        conversion_ms = getattr(display, "latest_conversion_ms", None) or 0.0
        self.frame_latency_value.setText(
            f"{float(conversion_ms):.2f} / {float(canvas.latest_paint_ms or 0):.2f} ms"
        )
        queue_depth = int(getattr(display, "presentation_queue_depth", 0))
        dropped = int(getattr(display, "presentation_frames_dropped", 0))
        overwritten = int(getattr(display, "pending_frames_overwritten", 0))
        skipped = overwritten + canvas.source_frames_skipped
        self.frame_buffer_value.setText(f"{queue_depth} / {dropped} / {skipped}")

    def _update_imu(self, snapshot: object) -> None:
        status = getattr(snapshot, "imu", None)
        telemetry = getattr(snapshot, "imu_telemetry", None)
        recording = getattr(snapshot, "recording", None)
        sensors = getattr(status, "sensors", {}) if status is not None else {}
        sensor_statuses = tuple(sensors.values()) if hasattr(sensors, "values") else ()
        rates = [
            float(value)
            for sensor in sensor_statuses
            if (value := getattr(sensor, "observed_rate_hz", None)) is not None
        ]
        latencies_ns = [
            int(value)
            for sensor in sensor_statuses
            if (value := getattr(sensor, "last_event_to_callback_delta_ns", None)) is not None
            and int(value) >= 0
        ]
        stats = ImuMonitorStats(
            sample_rate_hz=float(
                max(
                    float(getattr(telemetry, "accelerometer_rate_hz", 0.0)),
                    float(getattr(telemetry, "gyroscope_rate_hz", 0.0)),
                    max(rates, default=0.0),
                )
            ),
            latest_latency_ms=(
                getattr(telemetry, "latest_callback_latency_ms", None)
                if telemetry is not None
                else max(latencies_ns) / 1_000_000
                if latencies_ns
                else None
            ),
            sequence_gaps=int(
                getattr(telemetry, "sequence_gap_count", 0)
                if telemetry is not None
                else sum(
                    int(getattr(sensor, "sequence_gaps", 0))
                    for sensor in sensor_statuses
                )
            ),
            duplicate_samples=int(
                getattr(telemetry, "duplicate_count", 0)
                if telemetry is not None
                else _first_attr(
                    status,
                    "duplicate_samples",
                    "duplicate_count",
                    default=0,
                )
            ),
            out_of_order_samples=int(
                getattr(telemetry, "out_of_order_count", 0)
                if telemetry is not None
                else sum(
                    int(getattr(sensor, "out_of_order_samples", 0))
                    for sensor in sensor_statuses
                )
            ),
            queue_overflows=int(
                _first_attr(
                    recording,
                    "telemetry_queue_overflow_count",
                    default=0,
                )
            ),
            recording_window_ns=int(
                _first_attr(
                    recording,
                    "duration_ns",
                    default=int(
                        _first_attr(
                            recording,
                            "duration_ms",
                            "recording_duration_ms",
                            default=0,
                        )
                    )
                    * 1_000_000,
                )
                if recording is not None
                else 0
            ),
            csv_state=_recording_csv_state(recording),
        )
        self.imu_monitor.set_stats(stats)

    def _drain_imu_samples(self, snapshot: object) -> None:
        take_samples = getattr(self.runtime, "imu_telemetry_samples", None)
        raw_samples: Iterable[object]
        if callable(take_samples):
            raw_samples = tuple(take_samples(self._last_imu_sequence))
            self._last_imu_sequence = max(
                (
                    int(getattr(sample, "sequence", self._last_imu_sequence))
                    for sample in raw_samples
                ),
                default=self._last_imu_sequence,
            )
        else:
            telemetry = getattr(snapshot, "imu_telemetry", None)
            if telemetry is not None:
                raw_samples = (
                    *tuple(getattr(telemetry, "accelerometer", ())),
                    *tuple(getattr(telemetry, "gyroscope", ())),
                )
            else:
                status = getattr(snapshot, "imu", None)
                sensors = getattr(status, "sensors", {}) if status is not None else {}
                raw_samples = tuple(
                    sample
                    for sensor_status in (
                        sensors.values() if hasattr(sensors, "values") else ()
                    )
                    if (sample := getattr(sensor_status, "last_sample", None)) is not None
                )
        samples = tuple(
            converted
            for raw in raw_samples
            if (converted := _chart_sample(raw)) is not None
        )
        self.imu_monitor.append_samples(samples)

    def _toggle_stream(self) -> None:
        action = str(self.stream_button.property("action") or "start")
        self.runtime.request_stream(StreamControlAction(action))

    def _toggle_recording(self) -> None:
        action = str(self.recording_button.property("action") or "start")
        self.runtime.request_recording(action)

    def _drain_command_results(self) -> None:
        provider = getattr(self.runtime, "command_results", None)
        if not callable(provider):
            return
        for result in provider():
            succeeded = bool(getattr(result, "succeeded", False))
            name = str(getattr(result, "name", "操作"))
            detail = str(getattr(result, "detail", ""))
            if succeeded:
                InfoBar.success(
                    title=name,
                    content=detail,
                    parent=self,
                    duration=2500,
                )
            else:
                InfoBar.error(
                    title=f"{name}失败",
                    content=detail,
                    parent=self,
                    duration=5000,
                )


def _chart_sample(sample: object) -> ImuChartSample | None:
    values = getattr(sample, "values", None)
    if not isinstance(values, (tuple, list)) or len(values) != 3:
        return None
    sensor_type = _enum_value(getattr(sample, "sensor_type", ""))
    sequence_number = _first_attr(sample, "sequence_number", "sequence")
    if not sensor_type or not isinstance(sequence_number, int):
        return None
    recording_time_ns = int(
        _first_attr(
            sample,
            "recording_time_ns",
            "sensor_event_monotonic_ns",
            default=0,
        )
    )
    received_at_ns = int(
        _first_attr(
            sample,
            "received_at_client_monotonic_ns",
            "received_at_elapsed_realtime_ns",
            default=recording_time_ns,
        )
    )
    return ImuChartSample(
        sensor_type=sensor_type,
        sequence_number=sequence_number,
        recording_time_ns=recording_time_ns,
        received_at_client_monotonic_ns=received_at_ns,
        values=(float(values[0]), float(values[1]), float(values[2])),
    )


def _compact_button(
    text: str,
    icon: FluentIcon,
    *,
    primary: bool = False,
) -> PushButton:
    button: PushButton = PrimaryPushButton(text) if primary else PushButton(text)
    button.setIcon(icon)
    button.setFixedHeight(34)
    button.setMinimumWidth(104)
    return button


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value)).lower()


def _first_attr(value: object, *names: str, default: Any = None) -> Any:
    if value is None:
        return default
    for name in names:
        item = getattr(value, name, None)
        if item is not None:
            return item
    return default


def _clock_ms(duration_ms: int) -> str:
    minutes, remainder = divmod(max(0, duration_ms), 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _recording_csv_state(recording: object | None) -> str:
    if recording is None:
        return "idle"
    explicit = _first_attr(recording, "csv_state", "writer_state")
    if explicit is not None:
        return str(explicit)
    state = _enum_value(getattr(recording, "state", "idle"))
    return {
        "countdown": "writing",
        "recording": "writing",
        "finalizing": "finalizing",
        "ready": "idle",
        "unavailable": "idle",
        "error": "error",
    }.get(state, state)
