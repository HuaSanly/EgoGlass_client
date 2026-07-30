from __future__ import annotations

from datetime import datetime

import dearpygui.dearpygui as dpg

from ingest_gateway.recording_models import RecordingLibrary, RecordingSession
from ui.runtime import UnifiedRuntimeHost
from ui.state import RuntimeSnapshot
from ui.views.live import LiveView


class LibraryView:
    """Native recording library backed directly by ``RecordingRuntime``."""

    def __init__(
        self,
        runtime: UnifiedRuntimeHost,
        live_view: LiveView,
        parent: int | str,
    ) -> None:
        self.runtime = runtime
        self.live_view = live_view
        self._library_identity: int | None = None
        self._sessions: dict[str, str] = {}
        self._clips: dict[str, tuple[str, str]] = {}
        self._delete_target: tuple[str, str | None] | None = None
        with dpg.group(parent=parent):
            self._build_toolbar()
            with dpg.group(horizontal=True):
                self._build_session_panel()
                self._build_detail_panel()
        self._build_confirmation_modal()

    def update(self, snapshot: RuntimeSnapshot) -> None:
        library = snapshot.library
        if library is None or id(library) == self._library_identity:
            return
        self._library_identity = id(library)
        self._refresh_sessions(library)

    def _build_toolbar(self) -> None:
        with dpg.group(horizontal=True):
            dpg.add_text("采集资料库", color=(232, 236, 235))
            dpg.add_spacer(width=18)
            dpg.add_button(label="刷新", callback=self._force_refresh)
            dpg.add_text("会话和片段操作直接作用于本地采集目录", color=(112, 124, 124))
        dpg.add_separator()

    def _build_session_panel(self) -> None:
        with dpg.child_window(width=380, height=700, border=True):
            dpg.add_text("会话", color=(171, 180, 179))
            dpg.add_listbox(
                items=(),
                num_items=18,
                width=350,
                tag="library-session-list",
                callback=self._select_session,
            )
            dpg.add_separator()
            dpg.add_input_text(
                width=350,
                hint="会话名称",
                tag="library-session-name",
            )
            with dpg.group(horizontal=True):
                dpg.add_button(label="重命名", callback=self._rename_session)
                dpg.add_button(label="删除会话", callback=self._ask_delete_session)

    def _build_detail_panel(self) -> None:
        with dpg.child_window(width=-1, height=700, border=False):
            dpg.add_text("未选择会话", tag="library-session-title")
            dpg.add_text("", tag="library-session-meta", color=(156, 165, 165))
            dpg.add_separator()
            with dpg.group(horizontal=True):
                for label, tag in (
                    ("IMU", "library-quality-imu"),
                    ("视频映射", "library-quality-video"),
                    ("时钟", "library-quality-clock"),
                    ("异常", "library-quality-anomaly"),
                ):
                    with dpg.child_window(width=205, height=82, border=True):
                        dpg.add_text(label, color=(112, 124, 124))
                        dpg.add_text("--", tag=tag)
            dpg.add_spacer(height=8)
            dpg.add_text("片段", color=(171, 180, 179))
            dpg.add_listbox(
                items=(),
                num_items=12,
                width=850,
                tag="library-clip-list",
                callback=self._select_clip,
            )
            dpg.add_text("未选择片段", tag="library-clip-meta", color=(156, 165, 165))
            with dpg.group(horizontal=True):
                dpg.add_button(label="在主画面打开", callback=self._open_clip)
                dpg.add_button(label="生成识别回放", callback=self._generate_replay)
                dpg.add_button(label="删除片段", callback=self._ask_delete_clip)

    def _build_confirmation_modal(self) -> None:
        with dpg.window(
            label="确认删除",
            modal=True,
            show=False,
            no_resize=True,
            width=430,
            height=150,
            tag="library-delete-modal",
        ):
            dpg.add_text("", tag="library-delete-message", wrap=390)
            dpg.add_spacer(height=10)
            with dpg.group(horizontal=True):
                dpg.add_button(label="取消", callback=lambda: self._close_delete_modal())
                dpg.add_button(label="确认删除", callback=self._confirm_delete)

    def _refresh_sessions(self, library: RecordingLibrary) -> None:
        self._sessions = {
            self._session_label(session): session.session_id for session in library.sessions
        }
        labels = tuple(self._sessions)
        current = dpg.get_value("library-session-list")
        dpg.configure_item("library-session-list", items=labels)
        dpg.set_value(
            "library-session-list",
            current if current in self._sessions else (labels[0] if labels else ""),
        )
        self._select_session()

    def _select_session(self, *_args: object) -> None:
        session = self._selected_session()
        if session is None:
            self._clear_detail()
            return
        dpg.set_value("library-session-name", session.display_name or session.session_id)
        dpg.set_value("library-session-title", session.display_name or session.session_id)
        dpg.set_value(
            "library-session-meta",
            f"{session.state.value.upper()}  ·  {len(session.clips)} 个片段  ·  "
            f"{_format_datetime(session.started_at_unix_ms)}",
        )
        quality = session.quality
        dpg.set_value("library-quality-imu", f"{quality.imu_sample_count:,} 样本")
        coverage = quality.metadata_match_coverage
        dpg.set_value(
            "library-quality-video",
            "暂无元数据" if coverage is None else f"{coverage * 100:.1f}%",
        )
        dpg.set_value(
            "library-quality-clock",
            f"{quality.timestamp_mapping_segment_count} 段 / "
            f"拒绝 {quality.rejected_clock_mapping_segment_count}",
        )
        anomalies = (
            quality.imu_sequence_gap_count
            + quality.imu_out_of_order_sample_count
            + quality.telemetry_queue_overflow_count
        )
        dpg.set_value("library-quality-anomaly", str(anomalies))
        self._clips = {
            self._clip_label(index, clip): (session.session_id, clip.clip_id)
            for index, clip in enumerate(session.clips)
        }
        labels = tuple(self._clips)
        dpg.configure_item("library-clip-list", items=labels)
        dpg.set_value("library-clip-list", labels[0] if labels else "")
        self._select_clip()

    def _select_clip(self, *_args: object) -> None:
        session = self._selected_session()
        target = self._selected_clip_target()
        if session is None or target is None:
            dpg.set_value("library-clip-meta", "未选择片段")
            return
        clip = next((item for item in session.clips if item.clip_id == target[1]), None)
        if clip is None:
            return
        dpg.set_value(
            "library-clip-meta",
            f"{clip.width}x{clip.height}  ·  {clip.fps} FPS  ·  "
            f"{clip.frame_count:,} 帧  ·  {_format_duration(clip.duration_ms)}  ·  "
            f"{_format_bytes(clip.file_size_bytes)}",
        )

    def _selected_session(self) -> RecordingSession | None:
        session_id = self._sessions.get(dpg.get_value("library-session-list"))
        library = self.runtime.snapshot().library
        if session_id is None or library is None:
            return None
        return next((item for item in library.sessions if item.session_id == session_id), None)

    def _selected_clip_target(self) -> tuple[str, str] | None:
        return self._clips.get(dpg.get_value("library-clip-list"))

    def _open_clip(self) -> None:
        target = self._selected_clip_target()
        if target is not None:
            self.live_view.open_clip(*target)

    def _generate_replay(self) -> None:
        session = self._selected_session()
        if session is not None:
            self.runtime.request_replay_generation(session.session_id)

    def _rename_session(self) -> None:
        session = self._selected_session()
        name = str(dpg.get_value("library-session-name")).strip()
        if session is not None and name:
            self.runtime.rename_session(session.session_id, name)

    def _ask_delete_session(self) -> None:
        session = self._selected_session()
        if session is None:
            return
        self._delete_target = (session.session_id, None)
        display_name = session.display_name or session.session_id
        dpg.set_value(
            "library-delete-message",
            f"删除会话“{display_name}”及其全部本地数据？此操作不可撤销。",
        )
        dpg.configure_item("library-delete-modal", show=True)

    def _ask_delete_clip(self) -> None:
        target = self._selected_clip_target()
        if target is None:
            return
        self._delete_target = target
        dpg.set_value("library-delete-message", "删除所选视频片段及其索引数据？此操作不可撤销。")
        dpg.configure_item("library-delete-modal", show=True)

    def _confirm_delete(self) -> None:
        target = self._delete_target
        self._close_delete_modal()
        if target is None:
            return
        if target[1] is None:
            self.runtime.delete_session(target[0])
        else:
            self.runtime.delete_clip(target[0], target[1])

    def _close_delete_modal(self) -> None:
        self._delete_target = None
        dpg.configure_item("library-delete-modal", show=False)

    def _force_refresh(self) -> None:
        self.runtime.request_library_refresh()

    def _clear_detail(self) -> None:
        self._clips = {}
        dpg.set_value("library-session-title", "未选择会话")
        dpg.set_value("library-session-meta", "")
        dpg.configure_item("library-clip-list", items=())
        dpg.set_value("library-clip-list", "")
        dpg.set_value("library-clip-meta", "未选择片段")

    @staticmethod
    def _session_label(session: RecordingSession) -> str:
        return f"{session.display_name or session.session_id}  [{session.state.value}]"

    @staticmethod
    def _clip_label(index: int, clip: object) -> str:
        return f"片段 {index + 1:02d}  ·  {_format_duration(clip.duration_ms)}"


def _format_datetime(unix_ms: int) -> str:
    return datetime.fromtimestamp(unix_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")


def _format_duration(duration_ms: int) -> str:
    seconds = max(0, round(duration_ms / 1000))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"
