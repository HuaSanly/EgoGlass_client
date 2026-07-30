from __future__ import annotations

import dearpygui.dearpygui as dpg

from ingest_gateway.webrtc_models import ImuSensorType
from ui.state import RuntimeSnapshot


class DiagnosticsView:
    """Dense native diagnostics for transport, display, IMU, recording, and inference."""

    def __init__(self, parent: int | str) -> None:
        self._last_revision = -1
        with dpg.group(parent=parent):
            dpg.add_text("运行时诊断", color=(232, 236, 235))
            dpg.add_separator()
            with dpg.group(horizontal=True):
                self._build_panel("接入与视频", "diagnostics-video")
                self._build_panel("IMU", "diagnostics-imu")
                self._build_panel("识别与录制", "diagnostics-perception")
            dpg.add_spacer(height=8)
            dpg.add_text("最近事件", color=(171, 180, 179))
            dpg.add_input_text(
                multiline=True,
                readonly=True,
                width=-1,
                height=360,
                tag="diagnostics-events",
            )

    def update(self, snapshot: RuntimeSnapshot) -> None:
        if snapshot.revision == self._last_revision:
            return
        self._last_revision = snapshot.revision
        self._update_video(snapshot)
        self._update_imu(snapshot)
        self._update_perception(snapshot)
        events = list(snapshot.recent_events)
        if snapshot.last_error:
            events.insert(0, f"状态采集失败: {snapshot.last_error}")
        dpg.set_value("diagnostics-events", "\n".join(events) or "暂无运行事件")

    @staticmethod
    def _build_panel(title: str, tag: str) -> None:
        with dpg.child_window(width=440, height=255, border=True):
            dpg.add_text(title, color=(171, 180, 179))
            dpg.add_separator()
            dpg.add_text("等待状态", tag=tag, wrap=410)

    @staticmethod
    def _update_video(snapshot: RuntimeSnapshot) -> None:
        webrtc = snapshot.webrtc
        display = snapshot.display
        if webrtc is None:
            return
        lines = [
            f"WebRTC: {webrtc.phase.value} / {webrtc.connection_state or '--'}",
            f"设备会话: {webrtc.device_session_id or '--'}",
            f"接收: {webrtc.frames_received:,} 帧  {webrtc.average_fps or 0:.1f} FPS",
            f"画面: {webrtc.width or '--'}x{webrtc.height or '--'}  {webrtc.video_codec or '--'}",
            (
                f"RTP: 收到 {webrtc.rtp_packets_received:,}  "
                f"丢包 {webrtc.rtp_packets_lost:,} ({webrtc.rtp_packet_loss_percent:.2f}%)"
            ),
            (
                f"RTP 抖动: {webrtc.rtp_jitter_ms:.2f} ms  "
                f"损坏帧丢弃: {webrtc.corrupt_frames_dropped}"
            ),
            f"元数据: {webrtc.metadata_matched:,}/{webrtc.metadata_received:,}",
            f"待匹配: 帧 {webrtc.pending_frames} / 元数据 {webrtc.pending_metadata}",
            f"时钟跳变: {webrtc.sdk_clock_discontinuities}",
        ]
        if display is not None:
            lines.extend(
                (
                    f"RGB 转换: {display.frames_converted:,} 帧  {display.recent_fps:.1f} FPS",
                    (
                        f"RGB 分发: {display.rgb_frames_forwarded:,}  "
                        f"错误 {display.rgb_sink_failures:,}"
                    ),
                    f"显示覆盖: {display.pending_frames_overwritten:,}",
                )
            )
        if webrtc.last_error:
            lines.append(f"错误: {webrtc.last_error}")
        dpg.set_value("diagnostics-video", "\n".join(lines))

    @staticmethod
    def _update_imu(snapshot: RuntimeSnapshot) -> None:
        imu = snapshot.imu
        pose = snapshot.imu_pose
        if imu is None:
            return
        accelerometer = imu.sensors.get(ImuSensorType.ACCELEROMETER)
        gyroscope = imu.sensors.get(ImuSensorType.GYROSCOPE)
        lines = [
            f"通道: {imu.channel_state.value}",
            f"消息: {imu.messages_received:,}  样本: {imu.samples_received:,}",
            f"格式错误: {imu.malformed_messages}",
            _sensor_line("ACC", accelerometer),
            _sensor_line("GYRO", gyroscope),
        ]
        if pose is not None:
            lines.extend(
                (
                    f"姿态预览: {pose.recent_rate_hz:.1f} Hz",
                    f"姿态队列溢出: {pose.queue_overflow_count}",
                    f"样本年龄: {pose.latest_sample_age_ms or 0:.1f} ms",
                )
            )
        dpg.set_value("diagnostics-imu", "\n".join(lines))

    @staticmethod
    def _update_perception(snapshot: RuntimeSnapshot) -> None:
        perception = snapshot.perception
        recording = snapshot.recording
        lines = [
            f"识别: {perception.get('state', '--')}",
            f"识别详情: {perception.get('detail', '--')}",
            f"输入帧: {perception.get('live_frames_received', 0):,}",
            f"推理次数: {perception.get('live_inferences', 0):,}",
            f"推理丢帧: {perception.get('live_frames_dropped', 0):,}",
        ]
        latest = perception.get("latest_result")
        if isinstance(latest, dict):
            duration_ns = latest.get("inference_duration_ns")
            if isinstance(duration_ns, int):
                lines.append(f"最近推理: {duration_ns / 1_000_000:.1f} ms")
        if recording is not None:
            lines.extend(
                (
                    f"录制: {recording.state.value}",
                    f"会话: {recording.session_id or '--'}",
                    f"时长: {recording.recording_duration_ms / 1000:.1f} s",
                )
            )
        error = perception.get("last_error")
        if error:
            lines.append(f"识别错误: {error}")
        dpg.set_value("diagnostics-perception", "\n".join(lines))


def _sensor_line(label: str, sensor: object) -> str:
    if sensor is None:
        return f"{label}: --"
    return (
        f"{label}: {sensor.sample_count:,}  {sensor.observed_rate_hz or 0:.1f} Hz  "
        f"缺口 {sensor.sequence_gaps}  乱序 {sensor.out_of_order_samples}"
    )
