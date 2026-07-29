from operator_console.app import STATIC_DIR


def test_gateway_decoded_preview_is_the_only_live_viewer_source() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'id="decoded-preview-source"' in html
    assert 'id="hand-tracking-overlay"' in html
    assert "127.0.0.1:8770/api/v1/webrtc/decoded-preview.mjpg" in script
    assert "127.0.0.1:8770/api/v1/webrtc/decoded-preview/status" in script
    assert "state.liveVideoReady" in script
    assert "RTCPeerConnection" not in script
    assert "/api/v1/webrtc/viewer/sessions" not in script
    assert "frame.jpg" not in script
    assert 'id="live-badge-label">WAITING</span>' in html
    assert "renderDecodedPreviewState" in script
    assert "Glass3 解码视频在线" in script
    assert 'id="preview-fps"' in html
    assert 'id="viewer-empty"' in html


def test_live_video_contract_is_independent_of_checkout_line_endings() -> None:
    html = (STATIC_DIR / "index.html").read_bytes().replace(b"\r\n", b"\n")

    for content in (html, html.replace(b"\n", b"\r\n")):
        assert b'id="decoded-preview-source"' in content


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


def test_main_window_places_perception_below_video_and_imu_in_right_column() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    right_column = html.split('<aside class="right-column"', maxsplit=1)[1].split(
        "</aside>", maxsplit=1
    )[0]
    left_column = html.split('<div class="left-column"', maxsplit=1)[1].split(
        '<aside class="right-column"', maxsplit=1
    )[0]

    assert left_column.index('class="viewer-tool"') < left_column.index(
        'class="algorithm-tool"'
    )
    assert right_column.index('class="signal-tool"') < right_column.index('class="imu-tool"')
    assert 'class="algorithm-tool"' not in right_column
    assert "height: 100vh" in styles
    assert "overflow: hidden" in styles
    assert "grid-template-rows: auto minmax(96px, 1fr)" in styles
    assert ".algorithm-preview-stage" not in styles
    assert "overflow-y: auto" in styles
    assert 'id="imu-scene-canvas"' in right_column
    assert 'id="reset-imu-button"' in right_column
    assert "pollImuStatus" in script
    assert ".imu-stage" in styles
    assert "grid-template-rows: auto minmax(0, 1fr) auto" in styles


def test_hand_tracking_panel_uses_live_status_and_offline_replay() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    viewer_stage = html.split(
        '<div class="viewer-stage" id="viewer-stage">', maxsplit=1
    )[1].split('<div class="viewer-footer">', maxsplit=1)[0]

    assert 'id="event-rows"' not in html
    assert html.count('class="viewer-stage"') == 1
    assert 'id="decoded-preview-source"' in viewer_stage
    assert 'id="hand-tracking-overlay"' in viewer_stage
    assert 'id="hand-replay-video"' in viewer_stage
    assert 'class="algorithm-preview-stage"' not in html
    assert ".algorithm-preview-stage" not in styles
    assert 'id="viewer-mode-live"' in html
    assert 'id="viewer-mode-replay"' in html
    assert 'id="replay-session"' in html
    assert 'id="start-replay-button"' in html
    assert "function connectHandTrackingEvents()" in script
    assert 'new EventSource(`${handTrackingEndpoint}/events`)' in script
    assert 'eventSource.addEventListener("status"' in script
    assert "function pollHandTrackingStatus()" not in script
    assert "function startHandTrackingReplay()" in script
    assert 'viewerMode: "live"' in script
    assert 'setViewerMode("replay")' in script
    assert "drawHandTrackingOverlay(result)" in script
    assert "source_keypoints_2d_px" in script
    assert "source_bbox_xyxy_px" in script
    assert "const showLiveVideo = liveMode && state.liveVideoReady" in script
    assert "replayMatchesSelectedSession()" in script
    assert "/api/v1/perception/hand-tracking" in script
    assert "state.events" not in script
    assert "innerHTML" not in script


def test_video_stage_preserves_source_ratio_and_reserves_lower_workspace() -> None:
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert "aspect-ratio: 16 / 9" in styles
    assert "object-fit: cover" in styles
    assert "align-self: start" in styles
    assert "grid-template-rows: auto auto auto" in styles


def test_stream_toggle_and_recording_control_follow_gateway_state() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert 'class="link-state"' not in html
    assert "LAN DIRECT" not in html
    assert 'id="stream-toggle-button"' in html
    assert 'id="recording-toggle-button"' in html
    assert 'id="start-stream-button"' not in html
    assert 'id="stop-stream-button"' not in html
    assert 'id="recording-countdown"' in html
    assert 'id="stream-control-status"' in html
    assert "streamControlEndpoint" in script
    assert "pollStreamControlStatus" in script
    assert 'method: "POST"' in script
    assert 'body: JSON.stringify({ action })' in script
    assert 'new Set(["ready", "streaming", "stopped"])' in script
    assert 'state.controlState === "streaming"' in script
    assert 'elements.streamToggleButton.dataset.action = shouldStop ? "stop" : "start"' in script
    assert 'Math.min(3, Math.max(1, Math.ceil(remainingMs / 1000)))' in script
    assert 'state.controlCommandInFlight || ["starting", "stopping"]' in script
    assert ".stream-control-actions" in styles
    assert 'class="stream-control-icon"' in html
    assert 'class="stream-control-label"' in html
    assert ".stream-control-icon {" in styles
    assert ".stream-control-label {" in styles
    assert "white-space: nowrap" in styles
    assert ".stream-control-button span {" not in styles
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
    assert "beta: 0.05" in imu_scene
    assert "this.filter.update" in imu_scene
    assert "export class ImuFusionTracker" in imu_scene
    assert "INITIALIZATION_SAMPLE_COUNT = 6" in imu_scene
    assert "REFERENCE_SAMPLE_COUNT = 12" in imu_scene
    assert "isStableReferenceSample" in imu_scene
    assert "this.filter.init(average.x, average.y, average.z, 1, 0, 0)" in imu_scene
    assert "referenceCandidate.angleTo(this.rawQuaternion) <= MAX_REFERENCE_STEP_RAD" in imu_scene
    assert "referenceInverse" in imu_scene
    assert three.stat().st_size > 300_000
    assert three_core.stat().st_size > 100_000
    assert ahrs.stat().st_size > 20_000


def test_imu_display_mapping_reverses_pitch_after_fusion_only() -> None:
    imu_scene = (STATIC_DIR / "imu-scene.js").read_text(encoding="utf-8")

    assert "mapRelativeOrientationForDisplay(relative)" in imu_scene
    assert "displayEuler.x = -sensorEuler.x" in imu_scene
    assert "displayEuler.y = -sensorEuler.y" not in imu_scene
    assert "this.targetQuaternion.copy(displayOrientation.quaternion)" in imu_scene
    assert "pitch: THREE.MathUtils.radToDeg(displayOrientation.angles.pitch)" in imu_scene
    assert "this.filter.update(\n      gx,\n      gy,\n      gz," in imu_scene
