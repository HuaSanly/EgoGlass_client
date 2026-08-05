from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.conftest import CLIP_ID, SESSION_ID, complete_episode, create_capture_session
from ui.annotation.models import ProposalRequest, SaveDraftRequest
from ui.annotation.store import (
    AnnotationReadOnlyError,
    AnnotationStore,
    AnnotationValidationError,
    RevisionConflictError,
)


def test_workspace_discovers_capture_sessions_and_declares_real_capabilities(
    recordings_root: Path,
) -> None:
    workspace = AnnotationStore(recordings_root).workspace()

    assert workspace.implemented_strategies == [
        "manual",
        "clip_as_episode",
        "fixed_window",
    ]
    assert workspace.planned_strategies == [
        "event_marker",
        "motion_change",
        "hand_object_interaction",
        "vlm_semantic",
    ]
    assert len(workspace.sessions) == 1
    session = workspace.sessions[0]
    assert session.session_id == SESSION_ID
    assert session.editable is True
    assert session.annotation_status == "unannotated"
    assert session.clips[0].exact_frame_index_available is True


def test_workspace_isolates_unsupported_32_character_session_directories(
    recordings_root: Path,
) -> None:
    legacy_id = "a" * 32
    legacy_directory = recordings_root / legacy_id
    legacy_directory.mkdir()
    (legacy_directory / "session.json").write_text(
        '{"session_id":"' + legacy_id + '","clips":[]}', encoding="utf-8"
    )

    workspace = AnnotationStore(recordings_root).workspace()

    assert [session.session_id for session in workspace.sessions] == [SESSION_ID]
    assert workspace.skipped_session_count == 1
    assert workspace.skipped_session_reasons == {"unsupported_session_contract": 1}


def test_proposal_providers_return_whole_clip_and_deterministic_windows(
    recordings_root: Path,
) -> None:
    ids = iter(f"{value:032x}" for value in range(1, 20))
    store = AnnotationStore(recordings_root, clock_ns=lambda: 123, id_factory=lambda: next(ids))

    whole_clip = store.proposals(SESSION_ID, ProposalRequest(strategy="clip_as_episode"))
    windows = store.proposals(
        SESSION_ID,
        ProposalRequest(
            strategy="fixed_window",
            window_duration_ms=4000,
            stride_duration_ms=4000,
        ),
    )

    whole_clip_intervals = [
        (item.start_frame_index, item.end_frame_index_exclusive) for item in whole_clip.proposals
    ]
    window_intervals = [
        (item.start_frame_index, item.end_frame_index_exclusive) for item in windows.proposals
    ]
    assert whole_clip_intervals == [(0, 300)]
    assert window_intervals == [
        (0, 120),
        (120, 240),
        (240, 300),
    ]
    assert all(item.confidence is None for item in windows.proposals)


def test_draft_save_is_atomic_revisioned_and_rejects_overlap(recordings_root: Path) -> None:
    store = AnnotationStore(recordings_root, clock_ns=lambda: 456)
    request = SaveDraftRequest(
        base_revision=0,
        segmentation_strategy="manual",
        episodes=[complete_episode()],
    )
    draft = store.save_draft(SESSION_ID, request)

    assert draft.draft_revision == 1
    assert draft.updated_at_unix_ns == 456
    assert (recordings_root / SESSION_ID / "annotations/episode-annotation-v1/draft.json").is_file()
    with pytest.raises(RevisionConflictError):
        store.save_draft(SESSION_ID, request)

    overlapping = complete_episode(episode_id="6" * 32)
    overlapping["start_frame_index"] = 200
    overlapping["end_frame_index_exclusive"] = 260
    with pytest.raises(AnnotationValidationError, match="annotation validation failed"):
        store.save_draft(
            SESSION_ID,
            SaveDraftRequest(
                base_revision=1,
                segmentation_strategy="manual",
                episodes=[complete_episode(), overlapping],
            ),
        )


def test_publish_resolves_exact_pts_and_writes_immutable_content_revision(
    recordings_root: Path,
) -> None:
    store = AnnotationStore(recordings_root, clock_ns=lambda: 789)
    draft = store.save_draft(
        SESSION_ID,
        SaveDraftRequest(
            base_revision=0,
            segmentation_strategy="manual",
            episodes=[complete_episode()],
        ),
    )
    source_media = recordings_root / SESSION_ID / "media" / f"{CLIP_ID}.mp4"
    before = hashlib.sha256(source_media.read_bytes()).hexdigest()

    revision = store.publish(SESSION_ID, draft.draft_revision)
    repeated = store.publish(SESSION_ID, draft.draft_revision)

    assert revision.annotation_revision_id == repeated.annotation_revision_id
    assert revision.content_sha256 == repeated.content_sha256
    assert revision.episodes[0].start.mp4_pts == 90_000
    assert revision.episodes[0].start.timing_status == "exact"
    assert revision.episodes[0].end_exclusive.mp4_pts == 720_000
    assert revision.quality.status == "pass"
    revision_path = (
        recordings_root
        / SESSION_ID
        / "annotations/episode-annotation-v1/revisions"
        / f"{revision.annotation_revision_id}.json"
    )
    assert revision_path.is_file()
    assert hashlib.sha256(source_media.read_bytes()).hexdigest() == before


def test_publish_uses_exact_pts_without_inventing_pending_session_time(
    tmp_path: Path,
) -> None:
    create_capture_session(tmp_path, with_perception_alignment=False)
    store = AnnotationStore(tmp_path, clock_ns=lambda: 790)
    draft = store.save_draft(
        SESSION_ID,
        SaveDraftRequest(
            base_revision=0,
            segmentation_strategy="manual",
            episodes=[complete_episode()],
        ),
    )

    revision = store.publish(SESSION_ID, draft.draft_revision)

    assert revision.episodes[0].start.mp4_pts == 90_000
    assert revision.episodes[0].start.session_time_ns is None
    assert revision.episodes[0].start.timing_status == "unmapped"
    assert revision.episodes[0].end_exclusive.mp4_pts == 720_000
    assert revision.episodes[0].end_exclusive.session_time_ns is None
    assert revision.episodes[0].end_exclusive.timing_status == "unmapped"


def test_publish_rejects_missing_labels_and_phases(recordings_root: Path) -> None:
    episode = complete_episode()
    episode["labels"]["instruction"] = ""
    episode["phases"] = []
    store = AnnotationStore(recordings_root)
    draft = store.save_draft(
        SESSION_ID,
        SaveDraftRequest(
            base_revision=0,
            segmentation_strategy="manual",
            episodes=[episode],
        ),
    )

    with pytest.raises(AnnotationValidationError) as error:
        store.publish(SESSION_ID, draft.draft_revision)

    assert any("至少需要一个内部阶段" in issue for issue in error.value.issues)
    assert any("缺少任务描述" in issue for issue in error.value.issues)


def test_active_session_is_read_only(tmp_path: Path) -> None:
    create_capture_session(tmp_path, state="active")
    store = AnnotationStore(tmp_path)

    assert store.workspace().sessions[0].editable is False
    with pytest.raises(AnnotationReadOnlyError):
        store.save_draft(
            SESSION_ID,
            SaveDraftRequest(
                base_revision=0,
                segmentation_strategy="manual",
                episodes=[],
            ),
        )
