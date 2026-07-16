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


def test_main_window_uses_fixed_viewport_with_imu_in_the_right_column() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    right_column = html.split('<aside class="right-column"', maxsplit=1)[1].split(
        "</aside>", maxsplit=1
    )[0]

    assert right_column.index('class="signal-tool"') < right_column.index('class="imu-tool"')
    assert 'class="event-tool"' not in right_column
    assert "事件记录" not in right_column
    assert "height: 100vh" in styles
    assert "overflow: hidden" in styles
    assert 'id="imu-scene-canvas"' in right_column
    assert 'id="reset-imu-button"' in right_column
    assert "pollImuStatus" in script
    assert ".imu-stage" in styles
    assert "grid-template-rows: auto minmax(0, 1fr) auto" in styles


def test_video_stage_preserves_source_ratio_and_reserves_lower_workspace() -> None:
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert "aspect-ratio: 16 / 9" in styles
    assert "object-fit: cover" in styles
    assert "align-self: start" in styles
    assert "grid-template-rows: auto auto auto" in styles


def test_stream_controls_replace_lan_status_and_follow_gateway_state() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert 'class="link-state"' not in html
    assert "LAN DIRECT" not in html
    assert 'id="start-stream-button"' in html
    assert 'id="stop-stream-button"' in html
    assert 'id="stream-control-status"' in html
    assert "streamControlEndpoint" in script
    assert "pollStreamControlStatus" in script
    assert 'method: "POST"' in script
    assert 'body: JSON.stringify({ action })' in script
    assert 'new Set(["ready", "streaming", "stopped"])' in script
    assert 'state.controlState === "streaming"' in script
    assert 'state.controlState === "stopped"' in script
    assert 'state.controlCommandInFlight || ["starting", "stopping"]' in script
    assert ".stream-control-actions" in styles
    assert ".button:disabled" in styles


def test_shipped_operator_runtime_has_no_simulated_data_path() -> None:
    service_package = STATIC_DIR.parent
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    imu_scene = (STATIC_DIR / "imu-scene.js").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    backend = (service_package / "app.py").read_text(encoding="utf-8")
    shipped_runtime = "\n".join((html, script, imu_scene, styles, backend)).lower()

    assert not (service_package / "models.py").exists()
    assert not (service_package / "runtime.py").exists()
    for forbidden in (
        "simulation",
        "synthetic",
        "mock",
        "trajectory",
        "calibration",
        "recording",
        "模拟",
    ):
        assert forbidden not in shipped_runtime
    assert "/api/v1/state" not in backend
    assert "@app.websocket" not in backend


def test_imu_scene_uses_real_samples_and_vendored_sensor_fusion() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    imu_scene = (STATIC_DIR / "imu-scene.js").read_text(encoding="utf-8")
    three = STATIC_DIR / "vendor" / "three.module-0.185.1.min.js"
    three_core = STATIC_DIR / "vendor" / "three.core.min.js"
    ahrs = STATIC_DIR / "vendor" / "ahrs-1.3.3.js"

    assert 'id="imu-scene-canvas"' in html
    assert "127.0.0.1:8770/api/v1/webrtc/imu/status" in script
    assert "setTimeout(pollImuStatus, delayMs)" in script
    assert "readImuSample" in script
    assert "sensor_event_monotonic_ns" in script
    assert "new THREE.WebGLRenderer" in imu_scene
    assert 'algorithm: "Madgwick"' in imu_scene
    assert "this.filter.update" in imu_scene
    assert "referenceInverse" in imu_scene
    assert three.stat().st_size > 300_000
    assert three_core.stat().st_size > 100_000
    assert ahrs.stat().st_size > 20_000


def test_imu_display_mapping_reverses_pitch_after_fusion_only() -> None:
    imu_scene = (STATIC_DIR / "imu-scene.js").read_text(encoding="utf-8")

    assert "mapRelativeOrientationForDisplay(relative)" in imu_scene
    assert "displayEuler.y = -sensorEuler.y" in imu_scene
    assert "this.targetQuaternion.copy(displayOrientation.quaternion)" in imu_scene
    assert "pitch: THREE.MathUtils.radToDeg(displayOrientation.euler.y)" in imu_scene
    assert "this.filter.update(\n      gx,\n      gy,\n      gz," in imu_scene
