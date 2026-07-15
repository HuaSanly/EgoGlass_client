import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from egoglass_operator_console.app import create_app
from egoglass_operator_console.runtime import ConsoleRuntime


def make_client() -> TestClient:
    return TestClient(create_app(ConsoleRuntime()))


def test_health_and_static_console_are_served() -> None:
    with make_client() as client:
        health = client.get("/api/v1/health")
        page = client.get("/")

    assert health.json() == {
        "status": "ok",
        "service": "operator-console",
        "version": "0.1.0",
    }
    assert page.status_code == 200
    assert 'id="scene-canvas"' in page.text
    assert 'id="live-video-source"' in page.text
    assert "GLASS3 LIVE SOURCE" in page.text
    assert 'id="left-trajectory-toggle" type="checkbox" aria-label="左手模拟轨迹">' in page.text
    assert 'id="right-trajectory-toggle" type="checkbox" aria-label="右手模拟轨迹">' in page.text
    assert "SyntheticFrameSource" not in page.text
    assert "WebSocket v1" not in page.text
    assert "WebRTC / DTLS-SRTP" in page.text
    assert "模拟 H.264" not in page.text
    assert "EgoGlass Operator Console" in page.text


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
    assert "<span>实时</span>" not in navigation
    assert "<span>会话</span>" not in navigation
    assert "<span>数据</span>" not in navigation
    assert "<span>设置</span>" not in navigation


def test_settings_round_trip_and_revision() -> None:
    with make_client() as client:
        initial = client.get("/api/v1/state").json()
        settings = initial["settings"]
        settings["capture_fps"] = 15
        settings["inference_fps"] = 5
        updated = client.put("/api/v1/settings", json=settings)

    assert updated.status_code == 200
    payload = updated.json()
    assert payload["settings_revision"] == 2
    assert payload["settings"]["capture_fps"] == 15
    assert payload["settings"]["inference_fps"] == 5


def test_invalid_rate_returns_validation_error() -> None:
    with make_client() as client:
        settings = client.get("/api/v1/state").json()["settings"]
        settings["capture_fps"] = 10
        settings["inference_fps"] = 20
        response = client.put("/api/v1/settings", json=settings)

    assert response.status_code == 422


def test_recording_and_session_conflict_are_explicit() -> None:
    with make_client() as client:
        started = client.post("/api/v1/recording/start")
        stopped_session = client.post("/api/v1/session/stop")
        conflict = client.post("/api/v1/recording/start")

    assert started.json()["recording"] is True
    assert stopped_session.json()["recording"] is False
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "recording requires a live session"


def test_websocket_emits_versioned_trajectory_telemetry() -> None:
    with make_client() as client, client.websocket_connect("/api/v1/telemetry") as websocket:
        payload = websocket.receive_json()

    assert payload["schema_version"] == "1.0"
    assert payload["source"] == "synthetic"
    assert payload["calibration"]["state"] == "simulated"
    assert [hand["side"] for hand in payload["hands"]] == ["left", "right"]
    assert len(payload["hands"][0]["waypoints"]) == 10


def test_desktop_token_becomes_session_cookie_for_http_and_websocket() -> None:
    app = create_app(ConsoleRuntime(), desktop_token="test-desktop-token")
    with TestClient(app) as client:
        unauthorized = client.get("/api/v1/state")
        wrong_token = client.get("/?desktop_token=wrong")
        authorized_root = client.get("/?desktop_token=test-desktop-token")
        authorized_state = client.get("/api/v1/state")
        with client.websocket_connect("/api/v1/telemetry") as websocket:
            telemetry = websocket.receive_json()

    assert unauthorized.status_code == 401
    assert wrong_token.status_code == 401
    assert authorized_root.status_code == 200
    set_cookie = authorized_root.headers["set-cookie"].lower()
    assert "egoglass_desktop_session=" in set_cookie
    assert "httponly" in set_cookie
    assert "path=/" in set_cookie
    assert "samesite=strict" in set_cookie
    assert authorized_state.status_code == 200
    assert telemetry["source"] == "synthetic"


def test_desktop_websocket_rejects_missing_session_cookie() -> None:
    app = create_app(ConsoleRuntime(), desktop_token="test-desktop-token")

    with (
        TestClient(app) as client,
        pytest.raises(WebSocketDisconnect) as disconnect,
        client.websocket_connect("/api/v1/telemetry"),
    ):
        pass

    assert disconnect.value.code == 4401
