from html.parser import HTMLParser

import pytest
from starlette.testclient import TestClient

from operator_console.app import create_app


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
        storage_page = client.get("/storage")
        script = client.get("/assets/app.js")
        storage_script = client.get("/assets/storage.js")
        recordings_api = client.get("/assets/recordings-api.js")
        imu_scene = client.get("/assets/imu-scene.js")
        styles = client.get("/assets/styles.css")
        three = client.get("/assets/vendor/three.module-0.185.1.min.js")
        ahrs = client.get("/assets/vendor/ahrs-1.3.3.js")

    assert health.json() == {
        "status": "ok",
        "service": "operator-console",
        "version": "0.1.0",
    }
    assert page.status_code == 200
    assert storage_page.status_code == 200
    assert script.status_code == 200
    assert storage_script.status_code == 200
    assert recordings_api.status_code == 200
    assert imu_scene.status_code == 200
    assert three.status_code == 200
    assert ahrs.status_code == 200
    parser = ElementIdParser()
    parser.feed(page.text)
    assert parser.tags_by_id["decoded-preview-source"] == "img"
    assert parser.tags_by_id["hand-tracking-overlay"] == "canvas"
    assert parser.tags_by_id["hand-replay-video"] == "video"
    assert parser.tags_by_id["viewer-mode-live"] == "button"
    assert parser.tags_by_id["viewer-mode-replay"] == "button"
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
    left_column = page.text.split('<div class="left-column"', maxsplit=1)[1].split(
        '<aside class="right-column"', maxsplit=1
    )[0]
    assert left_column.index('class="viewer-tool"') < left_column.index(
        'class="algorithm-tool"'
    )
    assert right_column.index('class="signal-tool"') < right_column.index('class="imu-tool"')
    assert 'class="algorithm-tool"' not in right_column
    assert "事件记录" not in right_column
    assert parser.tags_by_id["start-replay-button"] == "button"
    assert parser.tags_by_id["replay-session"] == "select"
    assert parser.tags_by_id["imu-scene-canvas"] == "canvas"
    assert parser.tags_by_id["reset-imu-button"] == "button"
    assert "GLASS3 LIVE SOURCE" in page.text
    assert "WebRTC / DTLS-SRTP" in page.text
    assert "127.0.0.1:8770/api/v1/webrtc/decoded-preview.mjpg" in script.text
    assert "127.0.0.1:8770/api/v1/webrtc/decoded-preview/status" in script.text
    assert "RTCPeerConnection" not in script.text
    assert "/api/v1/webrtc/viewer/sessions" not in script.text
    assert 'id="link-state"' not in page.text
    assert "LAN DIRECT" not in page.text
    assert parser.tags_by_id["stream-toggle-button"] == "button"
    assert parser.tags_by_id["recording-toggle-button"] == "button"
    assert parser.tags_by_id["recording-countdown"] == "div"
    assert 'id="stream-control-status"' in right_column
    assert "127.0.0.1:8770/api/v1/webrtc/control" in script.text
    assert 'body: JSON.stringify({ action })' in script.text
    assert "controllableStreamStates" in script.text
    assert 'state.controlState === "streaming"' in script.text
    assert 'state.controlState === "stopped"' in script.text
    assert "127.0.0.1:8770/api/v1/webrtc/imu/status" in script.text
    assert "pollImuStatus" in script.text
    assert 'import { ImuSceneController } from "./imu-scene.js"' in script.text
    assert "function pollHandTrackingStatus()" in script.text
    assert "/api/v1/perception/hand-tracking" in script.text
    assert "source_keypoints_2d_px" in script.text
    assert "source_bbox_xyxy_px" in script.text
    assert 'addEvent("OK", "Glass3 视频已连接"' in script.text
    assert 'addEvent("OK", "Glass3 IMU 已连接"' in script.text
    assert "new THREE.WebGLRenderer" in imu_scene.text
    assert "THREE.PCFShadowMap" in imu_scene.text
    assert "THREE.PCFSoftShadowMap" not in imu_scene.text
    assert 'algorithm: "Madgwick"' in imu_scene.text
    assert 'browserRequire("ahrs")' in imu_scene.text
    assert "aspect-ratio: 16 / 9" in styles.text
    live_video_styles = styles.text.split(
        ".decoded-preview-source,\n.hand-replay-video {",
        maxsplit=1,
    )[1].split("}", maxsplit=1)[0]
    assert "object-fit: cover" in live_video_styles
    assert "object-fit: contain" not in live_video_styles
    assert "frame.jpg" not in script.text
    assert "EgoGlass Operator Console" in page.text
    assert "api/v1/recordings/status" in recordings_api.text
    assert "api/v1/recordings/library" in recordings_api.text
    assert 'video.controls = true' in storage_script.text


def test_home_uses_one_stage_for_live_tracking_and_replay() -> None:
    with make_client() as client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text
        styles = client.get("/assets/styles.css").text

    viewer_stage = page.split(
        '<div class="viewer-stage" id="viewer-stage">', maxsplit=1
    )[1].split('<div class="viewer-footer">', maxsplit=1)[0]
    algorithm_panel = page.split(
        '<section class="algorithm-tool"', maxsplit=1
    )[1].split('</section>\n          </div>', maxsplit=1)[0]

    assert page.count('class="viewer-stage"') == 1
    assert 'id="decoded-preview-source"' in viewer_stage
    assert 'id="hand-tracking-overlay"' in viewer_stage
    assert 'id="hand-replay-video"' in viewer_stage
    assert 'id="hand-tracking-overlay"' not in algorithm_panel
    assert 'id="hand-replay-video"' not in algorithm_panel
    assert 'class="algorithm-preview-stage"' not in page
    assert ".algorithm-preview-stage" not in styles
    assert 'id="viewer-mode-live"' in page
    assert 'id="viewer-mode-replay"' in page
    assert 'viewerMode: "live"' in script
    assert 'function setViewerMode(mode)' in script
    assert 'elements.decodedPreview.hidden = !liveMode' in script
    assert 'elements.handTrackingOverlay.hidden = !liveMode' in script
    assert "drawHandTrackingOverlay(result)" in script
    assert 'elements.handReplayVideo.hidden = !showReplay' in script
    assert 'const showLiveVideo = liveMode && state.liveVideoReady' in script
    assert 'sessions.length === 0 || state.replayRequestInFlight' in script
    assert 'elements.viewerEmpty.hidden = showLiveVideo || showReplay' in script


def test_runtime_contains_no_simulated_data_controls_or_transport() -> None:
    with make_client() as client:
        page = client.get("/").text
        script = client.get("/assets/app.js").text
        imu_scene = client.get("/assets/imu-scene.js").text
        styles = client.get("/assets/styles.css").text

    shipped_runtime = "\n".join((page, script, imu_scene, styles)).lower()
    for forbidden in (
        "simulation",
        "synthetic",
        "mock",
        "trajectory",
        "calibration",
        "模拟",
    ):
        assert forbidden not in shipped_runtime
    assert "websocket" not in script.lower()
    assert 'id="imu-scene-canvas"' in page
    assert 'class="algorithm-tool"' in page
    assert 'id="event-rows"' not in page
    assert "state.events" not in script


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


def test_sidebar_contains_home_storage_and_annotation_page_links() -> None:
    with make_client() as client:
        page = client.get("/")

    navigation = page.text.split('<nav class="nav-rail"', maxsplit=1)[1].split(
        "</nav>", maxsplit=1
    )[0]
    assert navigation.count("<a ") == 3
    assert '<a class="nav-item is-active" href="/" aria-current="page"' in navigation
    assert "<span>主页</span>" in navigation
    assert '<a class="nav-item" href="/storage"' in navigation
    assert "<span>存储</span>" in navigation
    assert '<a class="nav-item" href="/annotations"' in navigation
    assert "<span>标注</span>" in navigation
    assert "data-view=" not in navigation


def test_desktop_token_becomes_session_cookie_for_ui_assets() -> None:
    app = create_app(desktop_token="test-desktop-token")
    with TestClient(app) as client:
        unauthorized = client.get("/")
        unauthorized_storage = client.get("/storage")
        unauthorized_annotations = client.get("/annotations")
        health = client.get("/api/v1/health")
        wrong_token = client.get("/?desktop_token=wrong")
        authorized_root = client.get("/?desktop_token=test-desktop-token")
        authorized_storage = client.get("/storage")
        authorized_annotations = client.get("/annotations")
        authorized_asset = client.get("/assets/app.js")

    assert unauthorized.status_code == 401
    assert unauthorized_storage.status_code == 401
    assert unauthorized_annotations.status_code == 401
    assert health.status_code == 200
    assert wrong_token.status_code == 401
    assert authorized_root.status_code == 200
    set_cookie = authorized_root.headers["set-cookie"].lower()
    assert "egoglass_desktop_session=" in set_cookie
    assert "httponly" in set_cookie
    assert "path=/" in set_cookie
    assert "samesite=strict" in set_cookie
    assert authorized_storage.status_code == 200
    assert authorized_annotations.status_code == 200
    assert authorized_asset.status_code == 200
