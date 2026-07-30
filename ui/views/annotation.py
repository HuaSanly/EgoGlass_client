from __future__ import annotations

import dearpygui.dearpygui as dpg

from annotation.controller import AnnotationController, AnnotationEditorError
from annotation.models import EpisodeLabels, PhaseKind
from annotation.store import AnnotationValidationError
from ui.runtime import UnifiedRuntimeHost
from ui.state import RuntimeSnapshot
from ui.views.live import LiveView

PHASE_LABELS = {
    "准备": "prepare",
    "接近": "approach",
    "接触": "contact",
    "操作": "manipulate",
    "释放": "release",
    "完成": "complete",
    "其他": "other",
}
HAND_LABELS = {
    "未指定": "unspecified",
    "左手": "left",
    "右手": "right",
    "双手": "both",
    "无手": "none",
}
OUTCOME_LABELS = {
    "未检查": "unreviewed",
    "成功": "success",
    "失败": "failure",
    "中断": "interrupted",
    "无效": "invalid",
}


class AnnotationView:
    """Native episode/phase editor that reuses the application's main replay surface."""

    def __init__(
        self,
        runtime: UnifiedRuntimeHost,
        live_view: LiveView,
        parent: int | str,
    ) -> None:
        self.runtime = runtime
        self.live_view = live_view
        self.controller = AnnotationController(runtime.annotation)
        self._sessions: dict[str, str] = {}
        self._clips: dict[str, str] = {}
        self._episodes: dict[str, str] = {}
        self._phases: dict[str, str] = {}
        self._last_replay_frame = -1
        with dpg.group(parent=parent):
            self._build_toolbar()
            with dpg.group(horizontal=True):
                self._build_timeline_column()
                self._build_inspector()
        self._run(self._refresh_workspace)

    def update(self, _snapshot: RuntimeSnapshot) -> None:
        clip = self.controller.selected_clip
        session_id = self.controller.selected_session_id
        if clip is None or session_id is None:
            return
        if self.live_view.replay_target() != (session_id, clip.clip_id):
            return
        frame = round(self.live_view.replay_position_seconds() * clip.fps)
        frame = self.controller.set_current_frame(frame)
        if frame != self._last_replay_frame:
            self._last_replay_frame = frame
            dpg.set_value("annotation-playhead", frame)
            dpg.set_value("annotation-frame-readout", self._frame_readout())
            self._draw_timeline()

    def _build_toolbar(self) -> None:
        with dpg.group(horizontal=True):
            dpg.add_combo(
                items=(),
                width=260,
                tag="annotation-session-select",
                callback=self._select_session,
            )
            dpg.add_combo(
                items=(),
                width=190,
                tag="annotation-clip-select",
                callback=self._select_clip,
            )
            dpg.add_button(label="刷新", callback=lambda: self._run(self._refresh_workspace))
            dpg.add_button(label="在主画面打开", callback=self._open_clip)
            dpg.add_button(label="撤销", tag="annotation-undo", callback=self._undo)
            dpg.add_button(label="重做", tag="annotation-redo", callback=self._redo)
            dpg.add_button(label="保存草稿", callback=self._save)
            dpg.add_button(label="发布", callback=self._publish)
        dpg.add_text("未选择标注会话", tag="annotation-status", color=(156, 165, 165))
        dpg.add_separator()

    def _build_timeline_column(self) -> None:
        with dpg.child_window(width=900, height=700, border=False):
            dpg.add_text("Episode 时间线", color=(171, 180, 179))
            with dpg.drawlist(width=870, height=112, tag="annotation-timeline"):
                dpg.draw_rectangle(
                    (0, 0),
                    (870, 112),
                    fill=(12, 15, 16, 255),
                    color=(48, 56, 59, 255),
                )
                dpg.add_draw_layer(tag="annotation-timeline-content")
            dpg.add_slider_int(
                min_value=0,
                max_value=1,
                width=870,
                tag="annotation-playhead",
                callback=self._seek_frame,
            )
            with dpg.group(horizontal=True):
                dpg.add_text("帧 0 / 0", tag="annotation-frame-readout")
                dpg.add_spacer(width=12)
                dpg.add_button(label="设 Episode 入点", callback=self._mark_episode_in)
                dpg.add_button(label="以当前帧为出点新建", callback=self._add_episode)
                dpg.add_button(label="拆分", callback=self._split_episode)
                dpg.add_button(label="与后一段合并", callback=self._merge_episode)
                dpg.add_button(label="删除 Episode", callback=self._delete_episode)
            dpg.add_text("Episode 入点未设置", tag="annotation-episode-mark")
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_combo(
                    items=tuple(PHASE_LABELS),
                    default_value="操作",
                    width=110,
                    tag="annotation-phase-kind",
                )
                dpg.add_combo(
                    items=tuple(HAND_LABELS),
                    default_value="未指定",
                    width=110,
                    tag="annotation-phase-hand",
                )
                dpg.add_input_text(width=160, hint="阶段动作", tag="annotation-phase-verb")
                dpg.add_input_text(width=160, hint="阶段对象", tag="annotation-phase-object")
                dpg.add_button(label="设阶段入点", callback=self._mark_phase_in)
                dpg.add_button(label="新建阶段", callback=self._add_phase)
            dpg.add_text("阶段入点未设置", tag="annotation-phase-mark")
            dpg.add_listbox(
                items=(),
                num_items=5,
                width=870,
                tag="annotation-phase-list",
            )
            dpg.add_button(label="删除所选阶段", callback=self._delete_phase)
            dpg.add_separator()
            dpg.add_text("候选分段", color=(171, 180, 179))
            with dpg.group(horizontal=True):
                dpg.add_combo(
                    items=("整段作为 Episode", "固定窗口"),
                    default_value="整段作为 Episode",
                    width=180,
                    tag="annotation-proposal-strategy",
                )
                dpg.add_input_int(
                    default_value=8,
                    min_value=1,
                    max_value=120,
                    width=100,
                    tag="annotation-window-seconds",
                )
                dpg.add_button(label="生成候选", callback=self._generate_proposals)
                dpg.add_button(label="接受并替换当前片段", callback=self._accept_proposals)
            dpg.add_text("尚未生成候选", tag="annotation-proposal-status")

    def _build_inspector(self) -> None:
        with dpg.child_window(width=440, height=700, border=True):
            dpg.add_text("Episode 检查器", color=(171, 180, 179))
            dpg.add_combo(
                items=(),
                width=405,
                tag="annotation-episode-select",
                callback=self._select_episode,
            )
            dpg.add_input_text(width=405, hint="任务 ID", tag="annotation-task-id")
            dpg.add_input_text(width=405, hint="操作指令", tag="annotation-instruction")
            with dpg.group(horizontal=True):
                dpg.add_input_text(width=195, hint="动作", tag="annotation-verb")
                dpg.add_input_text(width=195, hint="对象", tag="annotation-object")
            dpg.add_input_text(width=405, hint="目标", tag="annotation-target")
            with dpg.group(horizontal=True):
                dpg.add_combo(
                    items=tuple(HAND_LABELS),
                    default_value="未指定",
                    width=195,
                    tag="annotation-hand",
                )
                dpg.add_combo(
                    items=tuple(OUTCOME_LABELS),
                    default_value="未检查",
                    width=195,
                    tag="annotation-outcome",
                )
            dpg.add_input_text(
                width=405,
                height=120,
                multiline=True,
                hint="备注",
                tag="annotation-notes",
            )
            dpg.add_button(label="应用标签", width=405, callback=self._apply_labels)
            dpg.add_separator()
            dpg.add_text("发布要求：指令、动作、对象、用手、结果和至少一个阶段。", wrap=405)

    def _refresh_workspace(self) -> None:
        workspace = self.controller.refresh_workspace()
        self._sessions = {
            f"{session.display_name}  [{session.annotation_status}]": session.session_id
            for session in workspace.sessions
        }
        labels = tuple(self._sessions)
        dpg.configure_item("annotation-session-select", items=labels)
        default_session_id = next(
            (
                session.session_id
                for session in workspace.sessions
                if session.clips and session.editable
            ),
            next((session.session_id for session in workspace.sessions if session.clips), None),
        )
        default_label = next(
            (
                label
                for label, session_id in self._sessions.items()
                if session_id == default_session_id
            ),
            labels[0] if labels else "",
        )
        dpg.set_value("annotation-session-select", default_label)
        if default_label:
            self._select_session()
        else:
            self._clear_editor()

    def _select_session(self, *_args: object) -> None:
        session_id = self._sessions.get(dpg.get_value("annotation-session-select"))
        if session_id is None:
            return

        def select() -> None:
            if self.controller.dirty:
                self.controller.save()
            detail = self.controller.select_session(session_id)
            self._clips = {
                f"片段 {index + 1:02d}  ·  {clip.frame_count} 帧": clip.clip_id
                for index, clip in enumerate(detail.session.clips)
            }
            labels = tuple(self._clips)
            dpg.configure_item("annotation-clip-select", items=labels)
            dpg.set_value("annotation-clip-select", labels[0] if labels else "")
            self._refresh_editor()

        self._run(select)

    def _select_clip(self, *_args: object) -> None:
        clip_id = self._clips.get(dpg.get_value("annotation-clip-select"))
        if clip_id is not None:
            self._run(lambda: self.controller.select_clip(clip_id))
            self._refresh_editor()

    def _select_episode(self, *_args: object) -> None:
        episode_id = self._episodes.get(dpg.get_value("annotation-episode-select"))
        self._run(lambda: self.controller.select_episode(episode_id))
        self._refresh_inspector()
        self._draw_timeline()

    def _open_clip(self) -> None:
        clip = self.controller.selected_clip
        session_id = self.controller.selected_session_id
        if clip is not None and session_id is not None:
            self.live_view.open_clip(session_id, clip.clip_id)
            self.live_view.show_live_tab()

    def _seek_frame(self, _sender: int | str, value: int) -> None:
        self.controller.set_current_frame(value)
        clip = self.controller.selected_clip
        if clip is not None:
            self.live_view.seek_replay(value / clip.fps)
        dpg.set_value("annotation-frame-readout", self._frame_readout())
        self._draw_timeline()

    def _mark_episode_in(self) -> None:
        self.controller.mark_episode_in()
        dpg.set_value("annotation-episode-mark", f"Episode 入点：{self.controller.episode_mark_in}")

    def _add_episode(self) -> None:
        self._run(self.controller.add_episode)
        self._refresh_editor()

    def _split_episode(self) -> None:
        self._run(self.controller.split_episode)
        self._refresh_editor()

    def _merge_episode(self) -> None:
        self._run(self.controller.merge_with_next)
        self._refresh_editor()

    def _delete_episode(self) -> None:
        self._run(self.controller.delete_episode)
        self._refresh_editor()

    def _mark_phase_in(self) -> None:
        self.controller.mark_phase_in()
        dpg.set_value("annotation-phase-mark", f"阶段入点：{self.controller.phase_mark_in}")

    def _add_phase(self) -> None:
        kind = PHASE_LABELS[dpg.get_value("annotation-phase-kind")]
        hand = HAND_LABELS[dpg.get_value("annotation-phase-hand")]
        verb = str(dpg.get_value("annotation-phase-verb")).strip() or None
        object_name = str(dpg.get_value("annotation-phase-object")).strip() or None
        self._run(
            lambda: self.controller.add_phase(
                PhaseKind(kind),
                active_hand=hand,
                action_verb=verb,
                object_name=object_name,
            )
        )
        self._refresh_editor()

    def _delete_phase(self) -> None:
        phase_id = self._phases.get(dpg.get_value("annotation-phase-list"))
        if phase_id is not None:
            self._run(lambda: self.controller.delete_phase(phase_id))
            self._refresh_editor()

    def _apply_labels(self) -> None:
        labels = EpisodeLabels(
            task_id=str(dpg.get_value("annotation-task-id")).strip() or None,
            instruction=str(dpg.get_value("annotation-instruction")),
            verb=str(dpg.get_value("annotation-verb")),
            object=str(dpg.get_value("annotation-object")),
            target=str(dpg.get_value("annotation-target")).strip() or None,
            hand=HAND_LABELS[dpg.get_value("annotation-hand")],
            outcome=OUTCOME_LABELS[dpg.get_value("annotation-outcome")],
            notes=str(dpg.get_value("annotation-notes")),
        )
        self._run(lambda: self.controller.update_labels(labels))
        self._refresh_editor()

    def _generate_proposals(self) -> None:
        fixed = dpg.get_value("annotation-proposal-strategy") == "固定窗口"
        seconds = int(dpg.get_value("annotation-window-seconds"))
        batch = self._run(
            lambda: self.controller.generate_proposals(
                "fixed_window" if fixed else "clip_as_episode",
                window_duration_ms=seconds * 1000,
                stride_duration_ms=seconds * 1000,
            )
        )
        if batch is not None:
            dpg.set_value("annotation-proposal-status", f"{len(batch.proposals)} 个候选待接受")
            self._draw_timeline()

    def _accept_proposals(self) -> None:
        count = self._run(self.controller.accept_proposals)
        if isinstance(count, int):
            dpg.set_value("annotation-proposal-status", f"已接受 {count} 个候选")
        self._refresh_editor()

    def _save(self) -> None:
        draft = self._run(self.controller.save)
        if draft is not None:
            self._set_status(f"草稿已保存，修订 {draft.draft_revision}", True)
        self._refresh_editor()

    def _publish(self) -> None:
        revision = self._run(self.controller.publish)
        if revision is not None:
            self._set_status(
                f"已发布 {revision.quality.episode_count} 个 Episode / "
                f"{revision.quality.phase_count} 个阶段",
                True,
            )
        self._refresh_editor()

    def _undo(self) -> None:
        self._run(self.controller.undo)
        self._refresh_editor()

    def _redo(self) -> None:
        self._run(self.controller.redo)
        self._refresh_editor()

    def _refresh_editor(self) -> None:
        detail = self.controller.detail
        clip = self.controller.selected_clip
        if detail is None or clip is None:
            self._clear_editor()
            return
        self._episodes = {
            f"{index + 1:02d}  {episode.start_frame_index}-{episode.end_frame_index_exclusive}  "
            f"{episode.labels.instruction or '未标注'}": episode.episode_id
            for index, episode in enumerate(
                item for item in detail.draft.episodes if item.clip_id == clip.clip_id
            )
        }
        labels = tuple(self._episodes)
        dpg.configure_item("annotation-episode-select", items=labels)
        selected_label = next(
            (
                label
                for label, episode_id in self._episodes.items()
                if episode_id == self.controller.selected_episode_id
            ),
            labels[0] if labels else "",
        )
        dpg.set_value("annotation-episode-select", selected_label)
        if selected_label and self.controller.selected_episode_id is None:
            self.controller.select_episode(self._episodes[selected_label])
        dpg.configure_item("annotation-playhead", max_value=max(1, clip.frame_count - 1))
        dpg.set_value("annotation-playhead", self.controller.current_frame_index)
        state = "有未保存修改" if self.controller.dirty else "已保存"
        status = (
            f"{detail.session.display_name}  ·  "
            f"{len(detail.draft.episodes)} 个 Episode  ·  {state}"
        )
        self._set_status(
            status,
            not self.controller.dirty,
        )
        dpg.configure_item("annotation-undo", enabled=self.controller.can_undo)
        dpg.configure_item("annotation-redo", enabled=self.controller.can_redo)
        self._refresh_inspector()
        self._draw_timeline()

    def _refresh_inspector(self) -> None:
        episode = self.controller.selected_episode
        if episode is None:
            for tag in (
                "annotation-task-id",
                "annotation-instruction",
                "annotation-verb",
                "annotation-object",
                "annotation-target",
                "annotation-notes",
            ):
                dpg.set_value(tag, "")
            self._phases = {}
            dpg.configure_item("annotation-phase-list", items=())
            return
        labels = episode.labels
        dpg.set_value("annotation-task-id", labels.task_id or "")
        dpg.set_value("annotation-instruction", labels.instruction)
        dpg.set_value("annotation-verb", labels.verb)
        dpg.set_value("annotation-object", labels.object)
        dpg.set_value("annotation-target", labels.target or "")
        dpg.set_value("annotation-hand", _label_for(HAND_LABELS, labels.hand.value))
        dpg.set_value("annotation-outcome", _label_for(OUTCOME_LABELS, labels.outcome.value))
        dpg.set_value("annotation-notes", labels.notes)
        self._phases = {
            f"{index + 1:02d}  {phase.phase.value}  "
            f"{phase.start_frame_index}-{phase.end_frame_index_exclusive}": phase.phase_id
            for index, phase in enumerate(episode.phases)
        }
        phase_labels = tuple(self._phases)
        dpg.configure_item("annotation-phase-list", items=phase_labels)
        dpg.set_value("annotation-phase-list", phase_labels[0] if phase_labels else "")

    def _draw_timeline(self) -> None:
        dpg.delete_item("annotation-timeline-content", children_only=True)
        clip = self.controller.selected_clip
        detail = self.controller.detail
        if clip is None or detail is None:
            return
        width = 870

        def x(frame: int) -> float:
            return frame / max(1, clip.frame_count) * width

        proposals = self.controller.proposal_batch
        if proposals is not None:
            for proposal in proposals.proposals:
                if proposal.clip_id == clip.clip_id:
                    dpg.draw_rectangle(
                        (x(proposal.start_frame_index), 8),
                        (x(proposal.end_frame_index_exclusive), 20),
                        color=(112, 124, 124, 255),
                        fill=(60, 68, 69, 150),
                        parent="annotation-timeline-content",
                    )
        for episode in detail.draft.episodes:
            if episode.clip_id != clip.clip_id:
                continue
            selected = episode.episode_id == self.controller.selected_episode_id
            color = (93, 199, 164, 255) if selected else (73, 126, 116, 255)
            dpg.draw_rectangle(
                (x(episode.start_frame_index), 28),
                (x(episode.end_frame_index_exclusive), 62),
                color=color,
                fill=(*color[:3], 130),
                parent="annotation-timeline-content",
            )
            if selected:
                for phase in episode.phases:
                    dpg.draw_rectangle(
                        (x(phase.start_frame_index), 72),
                        (x(phase.end_frame_index_exclusive), 96),
                        color=(245, 183, 83, 255),
                        fill=(121, 91, 43, 170),
                        parent="annotation-timeline-content",
                    )
        playhead_x = x(self.controller.current_frame_index)
        dpg.draw_line(
            (playhead_x, 0),
            (playhead_x, 108),
            color=(235, 104, 92, 255),
            thickness=2,
            parent="annotation-timeline-content",
        )

    def _frame_readout(self) -> str:
        clip = self.controller.selected_clip
        if clip is None:
            return "帧 0 / 0"
        seconds = self.controller.current_frame_index / clip.fps
        return (
            f"帧 {self.controller.current_frame_index} / {clip.frame_count - 1}  ·  "
            f"{seconds:.3f} s"
        )

    def _clear_editor(self) -> None:
        self._clips = {}
        self._episodes = {}
        self._phases = {}
        dpg.configure_item("annotation-clip-select", items=())
        dpg.configure_item("annotation-episode-select", items=())
        dpg.configure_item("annotation-phase-list", items=())
        self._set_status("所选会话没有可标注的视频片段", False)
        self._draw_timeline()

    def _run(self, command):
        try:
            return command()
        except AnnotationValidationError as error:
            self._set_status("；".join(error.issues), False)
        except (AnnotationEditorError, RuntimeError, ValueError, OSError) as error:
            self._set_status(str(error), False)
        return None

    @staticmethod
    def _set_status(detail: str, succeeded: bool) -> None:
        dpg.set_value("annotation-status", detail)
        dpg.configure_item(
            "annotation-status",
            color=(93, 199, 164) if succeeded else (235, 104, 92),
        )


def _label_for(mapping: dict[str, str], value: str) -> str:
    return next(
        (label for label, stored in mapping.items() if stored == value),
        next(iter(mapping)),
    )
