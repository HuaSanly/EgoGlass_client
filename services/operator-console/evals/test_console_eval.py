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


def test_live_video_contract_is_independent_of_checkout_line_endings() -> None:
    html = (STATIC_DIR / "index.html").read_bytes().replace(b"\r\n", b"\n")

    for content in (html, html.replace(b"\n", b"\r\n")):
        assert b'id="live-video-source"' in content


def test_video_status_replaces_the_removed_global_topbar() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    viewer_header = html.split('<section class="viewer-tool"', maxsplit=1)[1].split(
        '<div class="viewer-stage"', maxsplit=1
    )[0]

    assert 'class="topbar"' not in html
    assert 'class="brand"' not in html
    assert 'id="connection-label"' in viewer_header
    assert 'id="fullscreen-button"' in viewer_header
    assert "--topbar-height" not in styles
    assert ".topbar" not in styles


def test_main_window_uses_fixed_viewport_with_events_in_the_right_column() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    right_column = html.split('<aside class="right-column"', maxsplit=1)[1].split(
        "</aside>", maxsplit=1
    )[0]

    assert right_column.index('class="signal-tool"') < right_column.index(
        'class="event-tool"'
    )
    assert "height: 100vh" in styles
    assert ".event-table-wrap" in styles
    assert "overflow-y: auto" in styles
    assert 'event.className = "event-message"' in script
    assert '<col class="event-time-column">' in right_column
    assert '<col class="event-level-column">' in right_column
    assert "vertical-align: middle" in styles


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
