from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from egoglass_data_platform.app import _is_loopback_client, create_app
from tests.conftest import CLIP_ID, SESSION_ID, complete_episode


def test_annotation_api_runs_complete_draft_proposal_and_publish_workflow(
    recordings_root: Path,
) -> None:
    with TestClient(create_app(recordings_root)) as client:
        health = client.get("/api/v1/health")
        workspace = client.get("/api/v1/annotations/workspace")
        session = client.get(f"/api/v1/annotations/sessions/{SESSION_ID}")
        proposals = client.post(
            f"/api/v1/annotations/sessions/{SESSION_ID}/proposals",
            json={"strategy": "clip_as_episode"},
        )
        draft = client.put(
            f"/api/v1/annotations/sessions/{SESSION_ID}/draft",
            json={
                "base_revision": 0,
                "segmentation_strategy": "manual",
                "default_labels": {},
                "episodes": [complete_episode()],
            },
        )
        published = client.post(
            f"/api/v1/annotations/sessions/{SESSION_ID}/publish",
            json={"base_revision": 1},
        )
        media = client.get(f"/api/v1/annotations/media/{SESSION_ID}/{CLIP_ID}")

    assert health.json()["service"] == "data-platform"
    assert workspace.json()["sessions"][0]["session_id"] == SESSION_ID
    assert session.json()["draft"]["draft_revision"] == 0
    assert proposals.json()["proposals"][0]["end_frame_index_exclusive"] == 300
    assert draft.status_code == 200, draft.text
    assert published.status_code == 200, published.text
    assert published.json()["contract_id"] == "episode-annotation-v1"
    assert media.content == b"synthetic-mp4-fixture"


def test_api_returns_structured_validation_and_conflict_errors(recordings_root: Path) -> None:
    with TestClient(create_app(recordings_root)) as client:
        invalid = client.put(
            f"/api/v1/annotations/sessions/{SESSION_ID}/draft",
            json={
                "base_revision": 0,
                "segmentation_strategy": "manual",
                "default_labels": {},
                "episodes": [
                    complete_episode(),
                    {
                        **complete_episode(episode_id="6" * 32),
                        "start_frame_index": 200,
                        "end_frame_index_exclusive": 260,
                    },
                ],
            },
        )
        missing = client.get(f"/api/v1/annotations/sessions/{'f' * 32}")

    assert invalid.status_code == 422
    assert "issues" in invalid.json()["detail"]
    assert missing.status_code == 404


def test_data_platform_accepts_only_loopback_client_hosts() -> None:
    assert _is_loopback_client("127.0.0.1") is True
    assert _is_loopback_client("::1") is True
    assert _is_loopback_client("testclient") is True
    assert _is_loopback_client("192.168.1.20") is False
    assert _is_loopback_client("data-platform.local") is False
