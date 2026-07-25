from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from operator_console.app import create_app


def test_annotation_workspace_assets_and_runtime_config_are_served() -> None:
    app = create_app(data_platform_origin="http://127.0.0.1:8789")
    with TestClient(app) as client:
        page = client.get("/annotations")
        script = client.get("/assets/annotations.js")
        api = client.get("/assets/annotations-api.js")
        runtime = client.get("/api/v1/runtime")

    assert page.status_code == 200
    assert script.status_code == 200
    assert api.status_code == 200
    assert runtime.json() == {
        "schema_version": "1.0",
        "data_platform_origin": "http://127.0.0.1:8789",
    }
    for element_id in (
        "annotation-session-list",
        "annotation-video",
        "episode-track",
        "phase-track",
        "segmentation-strategy",
        "publish-annotation-button",
        "annotation-inspector-form",
    ):
        assert f'id="{element_id}"' in page.text
    assert '<option value="manual">' in page.text
    assert '<option value="clip_as_episode">' in page.text
    assert '<option value="fixed_window">' in page.text


def test_annotation_runtime_uses_real_service_and_non_destructive_drafts() -> None:
    with TestClient(create_app()) as client:
        page = client.get("/annotations").text
        script = client.get("/assets/annotations.js").text
        api = client.get("/assets/annotations-api.js").text
        styles = client.get("/assets/styles.css").text

    shipped = "\n".join((page, script, api, styles))
    assert 'fetch("/api/v1/runtime"' in api
    assert "/api/v1/annotations/sessions/${sessionId}/draft" in api
    assert "/api/v1/annotations/sessions/${sessionId}/proposals" in api
    assert "/api/v1/annotations/sessions/${sessionId}/publish" in api
    assert "state.proposalBatch" in script
    assert "pushHistory()" in script
    assert "episode.phases" in script
    assert "window.confirm" in script
    assert "mock" not in shipped.lower()
    assert "synthetic" not in shipped.lower()
    assert ".annotation-workspace" in styles
    annotation_layout = styles.split(".annotation-workspace {", maxsplit=1)[1].split(
        "}", maxsplit=1
    )[0]
    assert "grid-template-columns" in annotation_layout
    assert "overflow: hidden" in annotation_layout


@pytest.mark.parametrize(
    "origin",
    [
        "http://0.0.0.0:8780",
        "http://192.168.1.8:8780",
        "https://127.0.0.1:8780",
        "http://127.0.0.1:8780/path",
    ],
)
def test_annotation_runtime_rejects_non_loopback_or_non_origin_urls(origin: str) -> None:
    with pytest.raises(ValueError, match="127.0.0.1"):
        create_app(data_platform_origin=origin)
