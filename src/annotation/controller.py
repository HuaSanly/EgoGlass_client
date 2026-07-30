from __future__ import annotations

import uuid
from collections.abc import Callable

from .models import (
    AnnotationDraft,
    AnnotationSession,
    AnnotationWorkspace,
    EpisodeDraft,
    EpisodeLabels,
    EpisodeOutcome,
    PhaseDraft,
    PhaseKind,
    ProposalBatch,
    ProposalRequest,
    PublishedRevision,
    SaveDraftRequest,
    SegmentationStrategy,
)
from .store import AnnotationStore


class AnnotationEditorError(RuntimeError):
    """Raised when an editor command would produce an invalid draft."""


class AnnotationController:
    """Own native annotation editor state while persistence stays in ``AnnotationStore``."""

    def __init__(
        self,
        store: AnnotationStore,
        *,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
        history_limit: int = 80,
    ) -> None:
        if history_limit < 1:
            raise ValueError("history_limit must be positive")
        self.store = store
        self._id_factory = id_factory
        self._history_limit = history_limit
        self.workspace = AnnotationWorkspace(
            implemented_strategies=["manual", "clip_as_episode", "fixed_window"],
            planned_strategies=[
                "event_marker",
                "motion_change",
                "hand_object_interaction",
                "vlm_semantic",
            ],
            skipped_session_count=0,
            skipped_session_reasons={},
            sessions=[],
        )
        self.detail: AnnotationSession | None = None
        self.selected_session_id: str | None = None
        self.selected_clip_id: str | None = None
        self.selected_episode_id: str | None = None
        self.current_frame_index = 0
        self.episode_mark_in: int | None = None
        self.phase_mark_in: int | None = None
        self.proposal_batch: ProposalBatch | None = None
        self.dirty = False
        self._undo: list[tuple[AnnotationDraft, str | None]] = []
        self._redo: list[tuple[AnnotationDraft, str | None]] = []

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def selected_clip(self):
        if self.detail is None or self.selected_clip_id is None:
            return None
        return next(
            (clip for clip in self.detail.session.clips if clip.clip_id == self.selected_clip_id),
            None,
        )

    @property
    def selected_episode(self) -> EpisodeDraft | None:
        if self.detail is None or self.selected_episode_id is None:
            return None
        return next(
            (
                episode
                for episode in self.detail.draft.episodes
                if episode.episode_id == self.selected_episode_id
            ),
            None,
        )

    def refresh_workspace(self) -> AnnotationWorkspace:
        self.workspace = self.store.workspace()
        available = {session.session_id for session in self.workspace.sessions}
        if self.selected_session_id not in available:
            self.clear_selection()
        return self.workspace

    def clear_selection(self) -> None:
        self.detail = None
        self.selected_session_id = None
        self.selected_clip_id = None
        self.selected_episode_id = None
        self.proposal_batch = None
        self._undo.clear()
        self._redo.clear()
        self.dirty = False

    def select_session(self, session_id: str) -> AnnotationSession:
        self.detail = self.store.session(session_id)
        self.selected_session_id = session_id
        self.selected_clip_id = (
            self.detail.session.clips[0].clip_id if self.detail.session.clips else None
        )
        self.selected_episode_id = None
        self.current_frame_index = 0
        self.episode_mark_in = None
        self.phase_mark_in = None
        self.proposal_batch = None
        self._undo.clear()
        self._redo.clear()
        self.dirty = False
        return self.detail

    def select_clip(self, clip_id: str) -> None:
        if self.detail is None or clip_id not in {
            clip.clip_id for clip in self.detail.session.clips
        }:
            raise AnnotationEditorError("annotation clip is unavailable")
        self.selected_clip_id = clip_id
        self.selected_episode_id = None
        self.current_frame_index = 0
        self.episode_mark_in = None
        self.phase_mark_in = None
        self.proposal_batch = None

    def select_episode(self, episode_id: str | None) -> None:
        if episode_id is None:
            self.selected_episode_id = None
            return
        if self.detail is None:
            raise AnnotationEditorError("no annotation session is selected")
        episode = next(
            (item for item in self.detail.draft.episodes if item.episode_id == episode_id),
            None,
        )
        if episode is None:
            raise AnnotationEditorError("annotation episode is unavailable")
        self.selected_episode_id = episode_id
        self.selected_clip_id = episode.clip_id

    def set_current_frame(self, frame_index: int) -> int:
        clip = self._require_clip()
        self.current_frame_index = max(0, min(int(frame_index), clip.frame_count - 1))
        return self.current_frame_index

    def mark_episode_in(self) -> None:
        self.episode_mark_in = self.current_frame_index

    def mark_phase_in(self) -> None:
        self.phase_mark_in = self.current_frame_index

    def add_episode(
        self,
        start_frame_index: int | None = None,
        end_frame_index_exclusive: int | None = None,
        *,
        source_strategy: SegmentationStrategy = SegmentationStrategy.MANUAL,
    ) -> EpisodeDraft:
        detail = self._require_editable_detail()
        clip = self._require_clip()
        start = self.episode_mark_in if start_frame_index is None else start_frame_index
        end = (
            self.current_frame_index + 1
            if end_frame_index_exclusive is None
            else end_frame_index_exclusive
        )
        if start is None or start < 0 or start >= end or end > clip.frame_count:
            raise AnnotationEditorError("Episode boundaries are invalid")
        if any(
            episode.clip_id == clip.clip_id
            and start < episode.end_frame_index_exclusive
            and end > episode.start_frame_index
            for episode in detail.draft.episodes
        ):
            raise AnnotationEditorError("Episode cannot overlap an existing interval")
        self._push_history()
        episode = EpisodeDraft(
            episode_id=self._id_factory(),
            clip_id=clip.clip_id,
            start_frame_index=start,
            end_frame_index_exclusive=end,
            source_strategy=source_strategy,
            labels=detail.draft.default_labels.model_copy(deep=True),
        )
        detail.draft.episodes.append(episode)
        self._sort_episodes()
        self.selected_episode_id = episode.episode_id
        self.episode_mark_in = None
        self.dirty = True
        return episode

    def split_episode(self, frame_index: int | None = None) -> EpisodeDraft:
        episode = self._require_episode()
        split_at = self.current_frame_index if frame_index is None else frame_index
        if not episode.start_frame_index < split_at < episode.end_frame_index_exclusive:
            raise AnnotationEditorError("split frame must be inside the selected Episode")
        if any(
            phase.start_frame_index < split_at < phase.end_frame_index_exclusive
            for phase in episode.phases
        ):
            raise AnnotationEditorError("split frame crosses an existing phase")
        self._push_history()
        second = episode.model_copy(deep=True)
        second.episode_id = self._id_factory()
        second.start_frame_index = split_at
        second.labels.outcome = EpisodeOutcome.UNREVIEWED
        second.phases = [
            phase for phase in second.phases if phase.start_frame_index >= split_at
        ]
        episode.end_frame_index_exclusive = split_at
        episode.labels.outcome = EpisodeOutcome.UNREVIEWED
        episode.phases = [
            phase for phase in episode.phases if phase.end_frame_index_exclusive <= split_at
        ]
        assert self.detail is not None
        self.detail.draft.episodes.append(second)
        self._sort_episodes()
        self.selected_episode_id = second.episode_id
        self.dirty = True
        return second

    def merge_with_next(self) -> EpisodeDraft:
        episode = self._require_episode()
        assert self.detail is not None
        ordered = sorted(
            (
                item
                for item in self.detail.draft.episodes
                if item.clip_id == episode.clip_id
            ),
            key=lambda item: item.start_frame_index,
        )
        index = next(
            (
                position
                for position, item in enumerate(ordered)
                if item.episode_id == episode.episode_id
            ),
            -1,
        )
        if index < 0 or index + 1 >= len(ordered):
            raise AnnotationEditorError("selected Episode has no following interval")
        following = ordered[index + 1]
        self._push_history()
        episode.end_frame_index_exclusive = following.end_frame_index_exclusive
        episode.labels.outcome = EpisodeOutcome.UNREVIEWED
        episode.phases = sorted(
            [*episode.phases, *following.phases],
            key=lambda phase: phase.start_frame_index,
        )
        self.detail.draft.episodes = [
            item for item in self.detail.draft.episodes if item.episode_id != following.episode_id
        ]
        self.dirty = True
        return episode

    def delete_episode(self) -> None:
        episode = self._require_episode()
        self._push_history()
        assert self.detail is not None
        self.detail.draft.episodes = [
            item for item in self.detail.draft.episodes if item.episode_id != episode.episode_id
        ]
        self.selected_episode_id = None
        self.dirty = True

    def add_phase(
        self,
        phase: PhaseKind,
        *,
        start_frame_index: int | None = None,
        end_frame_index_exclusive: int | None = None,
        active_hand: str = "unspecified",
        action_verb: str | None = None,
        object_name: str | None = None,
    ) -> PhaseDraft:
        episode = self._require_episode()
        start = self.phase_mark_in if start_frame_index is None else start_frame_index
        end = (
            self.current_frame_index + 1
            if end_frame_index_exclusive is None
            else end_frame_index_exclusive
        )
        if (
            start is None
            or start < episode.start_frame_index
            or start >= end
            or end > episode.end_frame_index_exclusive
        ):
            raise AnnotationEditorError("phase boundaries must be inside the selected Episode")
        if any(
            start < current.end_frame_index_exclusive
            and end > current.start_frame_index
            for current in episode.phases
        ):
            raise AnnotationEditorError("phases cannot overlap")
        self._push_history()
        created = PhaseDraft(
            phase_id=self._id_factory(),
            start_frame_index=start,
            end_frame_index_exclusive=end,
            phase=phase,
            active_hand=active_hand,
            action_verb=action_verb,
            object=object_name,
        )
        episode.phases.append(created)
        episode.phases.sort(key=lambda item: item.start_frame_index)
        self.phase_mark_in = None
        self.dirty = True
        return created

    def delete_phase(self, phase_id: str) -> None:
        episode = self._require_episode()
        if phase_id not in {phase.phase_id for phase in episode.phases}:
            raise AnnotationEditorError("annotation phase is unavailable")
        self._push_history()
        episode.phases = [phase for phase in episode.phases if phase.phase_id != phase_id]
        self.dirty = True

    def update_labels(self, labels: EpisodeLabels) -> None:
        episode = self._require_episode()
        self._push_history()
        episode.labels = labels.model_copy(deep=True)
        self.dirty = True

    def generate_proposals(
        self,
        strategy: str,
        *,
        window_duration_ms: int = 8000,
        stride_duration_ms: int = 8000,
    ) -> ProposalBatch:
        clip = self._require_clip()
        self.proposal_batch = self.store.proposals(
            self._require_session_id(),
            ProposalRequest(
                strategy=strategy,
                clip_id=clip.clip_id,
                window_duration_ms=window_duration_ms,
                stride_duration_ms=stride_duration_ms,
            ),
        )
        return self.proposal_batch

    def accept_proposals(self) -> int:
        detail = self._require_editable_detail()
        clip = self._require_clip()
        batch = self.proposal_batch
        if batch is None:
            raise AnnotationEditorError("no proposal batch is available")
        self._push_history()
        detail.draft.episodes = [
            episode for episode in detail.draft.episodes if episode.clip_id != clip.clip_id
        ]
        for proposal in batch.proposals:
            detail.draft.episodes.append(
                EpisodeDraft(
                    episode_id=self._id_factory(),
                    clip_id=proposal.clip_id,
                    start_frame_index=proposal.start_frame_index,
                    end_frame_index_exclusive=proposal.end_frame_index_exclusive,
                    source_strategy=batch.strategy,
                    labels=detail.draft.default_labels.model_copy(deep=True),
                )
            )
        detail.draft.segmentation_strategy = SegmentationStrategy(batch.strategy)
        self._sort_episodes()
        selected = next(
            (episode for episode in detail.draft.episodes if episode.clip_id == clip.clip_id),
            None,
        )
        self.selected_episode_id = selected.episode_id if selected is not None else None
        count = len(batch.proposals)
        self.proposal_batch = None
        self.dirty = True
        return count

    def save(self) -> AnnotationDraft:
        detail = self._require_editable_detail()
        draft = self.store.save_draft(
            self._require_session_id(),
            SaveDraftRequest(
                base_revision=detail.draft.draft_revision,
                segmentation_strategy=detail.draft.segmentation_strategy,
                default_labels=detail.draft.default_labels,
                episodes=detail.draft.episodes,
            ),
        )
        detail.draft = draft
        self.dirty = False
        return draft

    def publish(self) -> PublishedRevision:
        if self.dirty:
            self.save()
        detail = self._require_editable_detail()
        revision = self.store.publish(
            self._require_session_id(),
            detail.draft.draft_revision,
        )
        detail.draft.latest_published_revision_id = revision.annotation_revision_id
        self.refresh_workspace()
        return revision

    def undo(self) -> None:
        detail = self._require_editable_detail()
        if not self._undo:
            return
        self._redo.append((detail.draft.model_copy(deep=True), self.selected_episode_id))
        detail.draft, self.selected_episode_id = self._undo.pop()
        self.dirty = True

    def redo(self) -> None:
        detail = self._require_editable_detail()
        if not self._redo:
            return
        self._undo.append((detail.draft.model_copy(deep=True), self.selected_episode_id))
        detail.draft, self.selected_episode_id = self._redo.pop()
        self.dirty = True

    def _push_history(self) -> None:
        detail = self._require_editable_detail()
        self._undo.append((detail.draft.model_copy(deep=True), self.selected_episode_id))
        if len(self._undo) > self._history_limit:
            del self._undo[0]
        self._redo.clear()

    def _sort_episodes(self) -> None:
        assert self.detail is not None
        self.detail.draft.episodes.sort(
            key=lambda episode: (episode.clip_id, episode.start_frame_index)
        )

    def _require_session_id(self) -> str:
        if self.selected_session_id is None:
            raise AnnotationEditorError("no annotation session is selected")
        return self.selected_session_id

    def _require_editable_detail(self) -> AnnotationSession:
        if self.detail is None:
            raise AnnotationEditorError("no annotation session is selected")
        if not self.detail.session.editable:
            raise AnnotationEditorError("active or finalizing sessions are read-only")
        return self.detail

    def _require_clip(self):
        clip = self.selected_clip
        if clip is None:
            raise AnnotationEditorError("no annotation clip is selected")
        return clip

    def _require_episode(self) -> EpisodeDraft:
        self._require_editable_detail()
        episode = self.selected_episode
        if episode is None:
            raise AnnotationEditorError("no annotation Episode is selected")
        return episode
