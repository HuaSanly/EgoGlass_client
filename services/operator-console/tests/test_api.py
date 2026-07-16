from html.parser import HTMLParser

import pytest
from starlette.testclient import TestClient

from egoglass_operator_console.app import create_app


class ElementIdParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags_by_id: dict[str, str] = {}

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        element_id = dict(attrs).get("id")
        if element_id is not None:
            self.tags_by_id[element_id] = tag


def make_client() -> TestClient:
    return TestClient(create_app())


def test_health_and_real_video_console_are_served() -> None:
    with make_client() as client:
        health = client.get("/api/v1/health")
        page = client.get("/")
        script = client.get("/assets/app.js")

    assert health.json() == {
        "status": "ok",
        "service": "operator-console",
        "version": "0.1.0",
    }
    assert page.status_code == 200
    assert script.status_code == 200
    parser = ElementIdParser()
    parser.feed(page.text)
    assert parser.tags_by_id["live-video-source"] == "video"
    assert 'class="topbar"' not in page.text
    assert 'class="brand"' not in page.text
    viewer_header = page.text.split('<section class="viewer-tool"', maxsplit=1)[1].split(
        '<div class="viewer-stage"', maxsplit=1
    )[0]
    assert 'id="connection-label"' in viewer_header
    assert 'id="fullscreen-button"' in viewer_header
    right_column = page.text.split('<aside class="right-column"', maxsplit=1)[1].split(
        "</aside>", maxsplit=1
    )[0]
    assert right_column.index('class="signal-tool"') < right_column.index(
        'class="event-tool"'
    )
    assert "<th>详情</th>" not in right_column
    assert "GLASS3 LIVE SOURCE" in page.text
    assert "WebRTC / DTLS-SRTP" in page.text
    assert "127.0.0.1:8770/api/v1/webrtc/viewer/sessions" in script.text
    assert "new RTCPeerConnection" in script.text
    assert "requestVideoFrameCallback" in script.text
    assert 'event.className = "event-message"' in script.text
    assert "event.append(eventTitle, eventDetail)" in script.text
    assert "frame.jpg" not in script.text
    assert "EgoGlass Operator Console" in page.text


def test_runtime_contains_no_simulated_data_controls_or_transport() -> None:
    with make_client() as client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text
        styles = client.get("/assets/styles.css").text

    shipped_runtime = "\n".join((page, script, styles)).lower()
    for forbidden in (
        "simulation",
        "synthetic",
        "mock",
        "trajectory",
        "telemetry",
        "calibration",
        "recording",
        "模拟",
    ):
        assert forbidden not in shipped_runtime
    assert "websocket" not in script.lower()
    assert "scene-canvas" not in page


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/v1/state"),
        ("put", "/api/v1/settings"),
        ("post", "/api/v1/session/start"),
        ("post", "/api/v1/session/stop"),
        ("post", "/api/v1/recording/start"),
        ("post", "/api/v1/recording/stop"),
    ],
)
def test_removed_simulation_routes_are_not_exposed(method: str, path: str) -> None:
    with make_client() as client:
        response = client.request(method, path)

    assert response.status_code == 404


def test_sidebar_contains_only_the_current_home_page_link() -> None:
    with make_client() as client:
        page = client.get("/")

    navigation = page.text.split('<nav class="nav-rail"', maxsplit=1)[1].split(
        "</nav>", maxsplit=1
    )[0]
    assert navigation.count("<a ") == 1
    assert '<a class="nav-item is-active" href="/" aria-current="page"' in navigation
    assert "<span>主页</span>" in navigation
    assert "data-view=" not in navigation


def test_desktop_token_becomes_session_cookie_for_ui_assets() -> None:
    app = create_app(desktop_token="test-desktop-token")
    with TestClient(app) as client:
        unauthorized = client.get("/")
        health = client.get("/api/v1/health")
        wrong_token = client.get("/?desktop_token=wrong")
        authorized_root = client.get("/?desktop_token=test-desktop-token")
        authorized_asset = client.get("/assets/app.js")

    assert unauthorized.status_code == 401
    assert health.status_code == 200
    assert wrong_token.status_code == 401
    assert authorized_root.status_code == 200
    set_cookie = authorized_root.headers["set-cookie"].lower()
    assert "egoglass_desktop_session=" in set_cookie
    assert "httponly" in set_cookie
    assert "path=/" in set_cookie
    assert "samesite=strict" in set_cookie
    assert authorized_asset.status_code == 200
