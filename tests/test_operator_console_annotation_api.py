from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from operator_console.app import create_app
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

    assert health.json()["service"] == "operator-console"
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


def test_annotation_api_uses_the_desktop_session(recordings_root: Path) -> None:
    app = create_app(recordings_root, desktop_token="desktop-test-token")
    with TestClient(app) as client:
        unauthorized = client.get("/api/v1/annotations/workspace")
        authorized = client.get(
            "/api/v1/annotations/workspace?desktop_token=desktop-test-token"
        )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert authorized.json()["sessions"][0]["session_id"] == SESSION_ID
