from egoglass_operator_console.app import STATIC_DIR


def test_live_glass3_preview_is_the_only_viewer_source() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'id="live-video-source"' in html
    assert "127.0.0.1:8770/api/v1/webrtc/viewer/sessions" in script
    assert "state.liveVideoReady" in script
    assert "new RTCPeerConnection" in script
    assert 'addTransceiver("video", { direction: "recvonly" })' in script
    assert "requestVideoFrameCallback" in script
    assert "frame.jpg" not in script
    assert 'id="live-badge-label">WAITING</span>' in html
    assert "renderVideoState" in script
    assert "Glass3 视频在线" in script
    assert 'id="preview-fps"' in html
    assert 'id="viewer-empty"' in html


def test_shipped_operator_runtime_has_no_simulated_data_path() -> None:
    service_package = STATIC_DIR.parent
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    backend = (service_package / "app.py").read_text(encoding="utf-8")
    shipped_runtime = "\n".join((html, script, styles, backend)).lower()

    assert not (service_package / "models.py").exists()
    assert not (service_package / "runtime.py").exists()
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
    assert "/api/v1/state" not in backend
    assert "@app.websocket" not in backend
