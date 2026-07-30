from __future__ import annotations

import concurrent.futures
from pathlib import Path

import dearpygui.dearpygui as dpg

from ingest_gateway.recording_models import RecordingState
from ingest_gateway.webrtc_models import StreamControlAction, StreamControlState
from ui.replay.player import ReplayPlayer, ReplayState
from ui.runtime import UnifiedRuntimeHost
from ui.state import RuntimeSnapshot
from ui.widgets.imu_pose import ImuPoseWidget
from ui.widgets.video_surface import VideoSurface


class LiveView:
    """One large surface shared by live RGB and PTS-driven replay."""

    def __init__(self, runtime: UnifiedRuntimeHost, parent: int | str) -> None:
        self.runtime = runtime
        self.replay = ReplayPlayer()
        self._viewer_mode = "live"
        self._last_snapshot_revision = -1
        self._last_result_key: tuple[str, str, int] | None = None
        self._library_identity: int | None = None
        self._session_labels: dict[str, str] = {}
        self._clip_labels: dict[str, tuple[str, str]] = {}
        self._media_future: concurrent.futures.Future[Path | None] | None = None
        self._result_media_future: concurrent.futures.Future[Path] | None = None
        self._pending_replay_target: tuple[str, str] | None = None
        self._active_replay_target: tuple[str, str] | None = None
        with dpg.group(parent=parent):
            self._build_header()
            self._build_source_controls()
            with dpg.group(horizontal=True):
                self._build_video_column()
                self._build_control_column()

    def _build_header(self) -> None:
        with dpg.group(horizontal=True):
            dpg.add_text("设备未连接", tag="live-connection-state")
            dpg.add_text("|", color=(88, 97, 98))
            dpg.add_text("等待视频", tag="live-video-state", color=(156, 165, 165))
            dpg.add_spacer(width=20)
            dpg.add_text("显示 0.0 FPS", tag="live-display-fps")
            dpg.add_text("推理 --", tag="live-inference-latency")
        dpg.add_separator()

    def _build_source_controls(self) -> None:
        with dpg.group(horizontal=True):
            dpg.add_radio_button(
                items=("实时", "回放"),
                default_value="实时",
                horizontal=True,
                tag="viewer-mode",
                callback=self._change_viewer_mode,
            )
            dpg.add_combo(
                items=(),
                width=220,
                tag="replay-session-select",
                callback=self._select_replay_session,
            )
            dpg.add_combo(
                items=(),
                width=150,
                tag="replay-clip-select",
            )
            dpg.add_button(label="打开", callback=self._open_selected_clip)
            dpg.add_button(label="生成识别回放", callback=self._generate_replay)
            dpg.add_button(
                label="播放识别结果",
                tag="open-result-replay-button",
                enabled=False,
                callback=self._open_result_replay,
            )

    def _build_video_column(self) -> None:
        with dpg.child_window(width=984, height=636, border=False):
            self.video = VideoSurface(parent=dpg.last_item(), width=960, height=540)
            with dpg.group(horizontal=True):
                dpg.add_text("LIVE", tag="viewer-mode-label", color=(93, 199, 164))
                dpg.add_text("等待首帧", tag="live-frame-detail")
                dpg.add_spacer(width=18)
                dpg.add_text("未收到识别结果", tag="live-perception-detail")
            with dpg.group(horizontal=True, show=False, tag="replay-controls"):
                dpg.add_button(
                    label="播放",
                    tag="replay-play-button",
                    callback=self._toggle_replay,
                )
                dpg.add_button(label="单帧", callback=lambda: self.replay.step())
                dpg.add_slider_float(
                    min_value=0.0,
                    max_value=1.0,
                    width=610,
                    tag="replay-position",
                    callback=self._seek_replay,
                )
                dpg.add_combo(
                    items=("0.25x", "0.5x", "1.0x", "1.5x", "2.0x"),
                    default_value="1.0x",
                    width=90,
                    callback=self._set_replay_rate,
                )
                dpg.add_text("00:00 / 00:00", tag="replay-timecode")

    def _build_control_column(self) -> None:
        with dpg.child_window(width=360, height=636, border=True):
            dpg.add_text("采集控制", color=(171, 180, 179))
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="开始视频",
                    width=156,
                    tag="stream-toggle-button",
                    callback=self._toggle_stream,
                )
                dpg.add_button(
                    label="开始录制",
                    width=156,
                    tag="recording-toggle-button",
                    callback=self._toggle_recording,
                )
            dpg.add_button(
                label="新建采集会话",
                width=320,
                callback=lambda: self.runtime.request_session("new"),
            )
            dpg.add_separator()
            dpg.add_text("当前会话", color=(171, 180, 179))
            dpg.add_text("尚未创建", tag="live-session-name")
            dpg.add_text("录制服务未就绪", tag="live-recording-detail", wrap=320)
            dpg.add_separator()
            dpg.add_text("手部感知", color=(171, 180, 179))
            dpg.add_text("等待模型", tag="live-hand-state")
            dpg.add_text("左手 --", tag="live-left-confidence")
            dpg.add_text("右手 --", tag="live-right-confidence")
            dpg.add_separator()
            dpg.add_text("帧链路", color=(171, 180, 179))
            dpg.add_text("接收 --", tag="live-ingest-metric")
            dpg.add_text("转换 --", tag="live-convert-metric")
            dpg.add_text("RGB 分发 --", tag="live-rgb-fanout-metric")
            dpg.add_text("上传 --", tag="live-upload-metric")
            dpg.add_text("覆盖 --", tag="live-drop-metric")
            dpg.add_separator()
            with dpg.group() as imu_group:
                dpg.add_text("IMU 姿态预览", color=(171, 180, 179))
                self.imu_pose = ImuPoseWidget(parent=imu_group, width=320, height=120)
            dpg.add_text("等待 IMU", tag="live-imu-detail")

    def update(self, snapshot: RuntimeSnapshot) -> None:
        self._refresh_replay_library(snapshot)
        self._resolve_media_futures()
        replay = self.replay.snapshot()
        frame = self.runtime.latest_frame() if self._viewer_mode == "live" else replay.frame
        if self.video.update_frame(frame):
            status = self.video.status()
            dpg.set_value(
                "live-frame-detail",
                f"帧 {status.latest_frame_index}  RGB {frame.width}x{frame.height}",
            )
        if self._viewer_mode == "replay":
            self._update_replay_controls(replay)
        if snapshot.revision == self._last_snapshot_revision:
            return
        self._last_snapshot_revision = snapshot.revision
        self._update_connection(snapshot)
        self._update_recording(snapshot)
        self._update_perception(snapshot)
        self._update_metrics(snapshot)
        self._update_imu(snapshot)

    def close(self) -> None:
        self.replay.close()

    def open_clip(self, session_id: str, clip_id: str) -> None:
        """Open a library/annotation clip on this view's only replay surface."""

        if self._media_future is not None:
            return
        self._pending_replay_target = (session_id, clip_id)
        self._media_future = self.runtime.media_path(session_id, clip_id)

    def replay_position_seconds(self) -> float:
        return self.replay.snapshot().position_seconds

    def replay_target(self) -> tuple[str, str] | None:
        return self._active_replay_target

    def seek_replay(self, seconds: float) -> None:
        self.replay.seek(seconds)

    def show_live_tab(self) -> None:
        dpg.set_value("main-navigation", "live-tab")

    def _update_connection(self, snapshot: RuntimeSnapshot) -> None:
        webrtc = snapshot.webrtc
        if webrtc is None:
            return
        dpg.set_value("live-connection-state", f"连接 {webrtc.phase.value.upper()}")
        dpg.set_value(
            "live-video-state",
            f"{webrtc.width or '--'}x{webrtc.height or '--'}  {webrtc.video_codec or '--'}",
        )
        status = snapshot.stream_control
        if status is None:
            return
        streaming = status.state in {
            StreamControlState.STARTING,
            StreamControlState.STREAMING,
        }
        dpg.configure_item(
            "stream-toggle-button",
            label="停止视频" if streaming else "开始视频",
            user_data="stop" if streaming else "start",
            enabled=status.state is not StreamControlState.UNAVAILABLE,
        )

    def _update_recording(self, snapshot: RuntimeSnapshot) -> None:
        recording = snapshot.recording
        if recording is None:
            return
        active = recording.state in {RecordingState.COUNTDOWN, RecordingState.RECORDING}
        busy = recording.state is RecordingState.FINALIZING
        dpg.configure_item(
            "recording-toggle-button",
            label="停止录制" if active else "开始录制",
            user_data="stop" if active else "start",
            enabled=recording.state is not RecordingState.UNAVAILABLE and not busy,
        )
        dpg.set_value("live-recording-detail", recording.detail)
        dpg.set_value("live-session-name", recording.session_id or "尚未创建")

    def _update_perception(self, snapshot: RuntimeSnapshot) -> None:
        perception = snapshot.perception
        state = perception.get("state", "idle")
        detail = perception.get("detail", "等待识别")
        dpg.set_value("live-hand-state", f"{str(state).upper()}  {detail}")
        latest = perception.get("latest_result")
        if not isinstance(latest, dict):
            return
        result_key = _perception_result_key(latest)
        if (
            self._viewer_mode == "live"
            and result_key is not None
            and result_key != self._last_result_key
        ):
            self.video.update_overlay(latest)
            self._last_result_key = result_key
        hands = latest.get("hands")
        left = None
        right = None
        if isinstance(hands, list):
            for hand in hands:
                if not isinstance(hand, dict):
                    continue
                if hand.get("handedness") == "left":
                    left = hand
                elif hand.get("handedness") == "right":
                    right = hand
        dpg.set_value("live-left-confidence", _confidence_text("左手", left))
        dpg.set_value("live-right-confidence", _confidence_text("右手", right))
        duration = latest.get("inference_duration_ns")
        if isinstance(duration, int):
            dpg.set_value("live-inference-latency", f"推理 {duration / 1_000_000:.1f} ms")
        hand_count = len(hands) if isinstance(hands, list) else 0
        dpg.set_value(
            "live-perception-detail",
            f"结果帧 {latest.get('frame_index', '--')}  检出 {hand_count} 只手",
        )

    def _update_metrics(self, snapshot: RuntimeSnapshot) -> None:
        display = snapshot.display
        if display is None:
            return
        surface = self.video.status()
        dpg.set_value("live-display-fps", f"显示 {surface.recent_upload_fps:.1f} FPS")
        dpg.set_value("live-ingest-metric", f"接收 {display.frames_received} 帧")
        dpg.set_value(
            "live-convert-metric",
            f"转换 {display.frames_converted} 帧 / {display.latest_conversion_ms or 0:.2f} ms",
        )
        dpg.set_value(
            "live-rgb-fanout-metric",
            (
                f"RGB 分发 {display.rgb_frames_forwarded} 帧 / "
                f"错误 {display.rgb_sink_failures}"
            ),
        )
        dpg.set_value(
            "live-upload-metric",
            f"上传 {surface.uploaded_frames} 帧 / {surface.latest_upload_ms or 0:.2f} ms",
        )
        dpg.set_value(
            "live-drop-metric",
            f"显示覆盖 {display.pending_frames_overwritten} / 跳过 {surface.source_frames_skipped}",
        )

    def _update_imu(self, snapshot: RuntimeSnapshot) -> None:
        pose = snapshot.imu_pose
        self.imu_pose.update(pose)
        if pose is None or pose.samples_received == 0:
            dpg.set_value("live-imu-detail", "等待 IMU")
            return
        dpg.set_value(
            "live-imu-detail",
            (
                f"{pose.recent_rate_hz:.1f} Hz  "
                f"R {pose.roll_degrees:.1f}  P {pose.pitch_degrees:.1f}  "
                f"Y {pose.yaw_degrees:.1f}"
            ),
        )

    def _refresh_replay_library(self, snapshot: RuntimeSnapshot) -> None:
        library = snapshot.library
        if library is not None and id(library) != self._library_identity:
            self._library_identity = id(library)
            self._session_labels = {
                (session.display_name or session.session_id): session.session_id
                for session in library.sessions
                if session.clips
            }
            labels = tuple(self._session_labels)
            dpg.configure_item("replay-session-select", items=labels)
            if dpg.get_value("replay-session-select") not in self._session_labels:
                dpg.set_value("replay-session-select", labels[0] if labels else "")
            self._select_replay_session()
        replay = snapshot.perception.get("replay", {})
        report = replay.get("report") if isinstance(replay, dict) else None
        dpg.configure_item("open-result-replay-button", enabled=isinstance(report, dict))

    def _select_replay_session(self, *_args: object) -> None:
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
                    f"片段 {index + 1:02d}": (session_id, clip.clip_id)
                    for index, clip in enumerate(session.clips)
                }
        labels = tuple(self._clip_labels)
        dpg.configure_item("replay-clip-select", items=labels)
        dpg.set_value("replay-clip-select", labels[0] if labels else "")

    def _open_selected_clip(self) -> None:
        target = self._clip_labels.get(dpg.get_value("replay-clip-select"))
        if target is not None:
            self.open_clip(*target)

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

    def _resolve_media_futures(self) -> None:
        if self._media_future is not None and self._media_future.done():
            future, self._media_future = self._media_future, None
            try:
                path = future.result()
                if path is not None:
                    self.replay.open(path)
                    self._active_replay_target = self._pending_replay_target
                    self._set_replay_mode()
                    self.show_live_tab()
                self._pending_replay_target = None
            except Exception as error:
                self._pending_replay_target = None
                dpg.set_value("global-command-status", str(error))
        if self._result_media_future is not None and self._result_media_future.done():
            future, self._result_media_future = self._result_media_future, None
            try:
                self.replay.open(future.result())
                self._set_replay_mode()
            except Exception as error:
                dpg.set_value("global-command-status", str(error))

    def _change_viewer_mode(self, _sender: int | str, value: str) -> None:
        self._viewer_mode = "live" if value == "实时" else "replay"
        dpg.configure_item("replay-controls", show=self._viewer_mode == "replay")
        dpg.set_value("viewer-mode-label", "LIVE" if self._viewer_mode == "live" else "REPLAY")
        if self._viewer_mode == "replay":
            self.video.update_overlay(None)
        else:
            self._last_result_key = None

    def _set_replay_mode(self) -> None:
        self._viewer_mode = "replay"
        dpg.set_value("viewer-mode", "回放")
        dpg.set_value("viewer-mode-label", "REPLAY")
        dpg.configure_item("replay-controls", show=True)
        self.video.update_overlay(None)

    def _toggle_replay(self) -> None:
        if self.replay.snapshot().state is ReplayState.PLAYING:
            self.replay.pause()
        else:
            self.replay.play()

    def _seek_replay(self, _sender: int | str, value: float) -> None:
        self.replay.seek(value)

    def _set_replay_rate(self, _sender: int | str, value: str) -> None:
        self.replay.set_playback_rate(float(value.removesuffix("x")))

    def _update_replay_controls(self, replay: object) -> None:
        if not hasattr(replay, "state"):
            return
        snapshot = self.replay.snapshot()
        dpg.configure_item(
            "replay-play-button",
            label="暂停" if snapshot.state is ReplayState.PLAYING else "播放",
        )
        dpg.configure_item(
            "replay-position",
            max_value=max(0.001, snapshot.duration_seconds),
        )
        dpg.set_value("replay-position", snapshot.position_seconds)
        dpg.set_value(
            "replay-timecode",
            f"{_clock(snapshot.position_seconds)} / {_clock(snapshot.duration_seconds)}",
        )

    def _selected_session_id(self) -> str | None:
        return self._session_labels.get(dpg.get_value("replay-session-select"))

    def _toggle_stream(self, sender: int | str) -> None:
        action = dpg.get_item_user_data(sender) or "start"
        self.runtime.request_stream(StreamControlAction(action))

    def _toggle_recording(self, sender: int | str) -> None:
        action = dpg.get_item_user_data(sender) or "start"
        self.runtime.request_recording(str(action))


def _confidence_text(label: str, hand: dict[str, object] | None) -> str:
    if hand is None:
        return f"{label} --"
    values = []
    for name, field in (
        ("det", "detector_confidence"),
        ("rec", "reconstruction_quality"),
        ("final", "final_confidence"),
    ):
        value = hand.get(field)
        if isinstance(value, (int, float)):
            values.append(f"{name} {float(value):.2f}")
        else:
            values.append(f"{name} --")
    return f"{label}  " + "  ".join(values)


def _perception_result_key(result: dict[str, object]) -> tuple[str, str, int] | None:
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


def _clock(seconds: float) -> str:
    total = max(0, round(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"
