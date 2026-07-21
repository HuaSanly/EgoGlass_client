from __future__ import annotations

import hashlib
import json
from pathlib import Path

from egoglass_data_platform.models import ProposalRequest, SaveDraftRequest
from egoglass_data_platform.store import AnnotationStore
from tests.conftest import CLIP_ID, SESSION_ID, complete_episode


def test_eval_manual_review_publishes_traceable_non_destructive_revision(
    recordings_root: Path,
) -> None:
    store = AnnotationStore(recordings_root)
    media = recordings_root / SESSION_ID / "media" / f"{CLIP_ID}.mp4"
    telemetry = recordings_root / SESSION_ID / "telemetry" / "telemetry.sqlite"
    source_hashes = {
        "media": hashlib.sha256(media.read_bytes()).hexdigest(),
        "telemetry": hashlib.sha256(telemetry.read_bytes()).hexdigest(),
    }
    proposed = store.proposals(
        SESSION_ID,
        ProposalRequest(
            strategy="fixed_window",
            window_duration_ms=4000,
            stride_duration_ms=4000,
        ),
    )
    assert len(proposed.proposals) == 3

    reviewed_episode = complete_episode()
    reviewed_episode["source_strategy"] = "fixed_window"
    draft = store.save_draft(
        SESSION_ID,
        SaveDraftRequest(
            base_revision=0,
            segmentation_strategy="fixed_window",
            episodes=[reviewed_episode],
        ),
    )
    revision = store.publish(SESSION_ID, draft.draft_revision)

    assert revision.quality.status == "pass"
    assert revision.quality.episode_count == 1
    assert revision.quality.phase_count == 2
    assert revision.episodes[0].source_strategy == "fixed_window"
    assert revision.episodes[0].start.session_time_ns is not None
    assert hashlib.sha256(media.read_bytes()).hexdigest() == source_hashes["media"]
    assert hashlib.sha256(telemetry.read_bytes()).hexdigest() == source_hashes["telemetry"]


def test_eval_declared_future_providers_never_return_placeholder_proposals(
    recordings_root: Path,
) -> None:
    workspace = AnnotationStore(recordings_root).workspace()

    assert set(workspace.implemented_strategies).isdisjoint(workspace.planned_strategies)
    assert set(workspace.planned_strategies) == {
        "event_marker",
        "motion_change",
        "hand_object_interaction",
        "vlm_semantic",
    }


def test_eval_mixed_session_root_keeps_valid_v1_sessions_visible(
    recordings_root: Path,
) -> None:
    unsupported_id = "a" * 32
    malformed_id = "b" * 32
    unsupported = recordings_root / unsupported_id
    malformed = recordings_root / malformed_id
    unsupported.mkdir()
    malformed.mkdir()
    (unsupported / "session.json").write_text(
        json.dumps({"session_id": unsupported_id, "clips": []}), encoding="utf-8"
    )
    (malformed / "session.json").write_text("{", encoding="utf-8")

    workspace = AnnotationStore(recordings_root).workspace()

    assert [session.session_id for session in workspace.sessions] == [SESSION_ID]
    assert workspace.skipped_session_count == 2
    assert workspace.skipped_session_reasons == {
        "unreadable_session": 1,
        "unsupported_session_contract": 1,
    }
