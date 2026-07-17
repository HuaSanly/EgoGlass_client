from egoglass_operator_console.app import STATIC_DIR


def test_recording_workflow_is_gateway_backed_and_countdown_is_authoritative() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    api_script = (STATIC_DIR / "recordings-api.js").read_text(encoding="utf-8")

    assert 'id="recording-toggle-button"' in html
    assert 'id="recording-countdown"' in html
    assert "recordingStatusEndpoint" in app_script
    assert "recordingCommandEndpoint" in app_script
    assert 'method: "POST"' in app_script
    assert 'body: JSON.stringify({ action })' in app_script
    assert "status.recording_starts_at_unix_ms - Date.now()" in app_script
    assert 'status?.state !== "countdown"' in app_script
    assert "setTimeout(updateRecordingCountdown, 50)" in app_script
    assert '"countdown", "recording", "finalizing"' in app_script
    assert "停止录制后才能控制视频流" in app_script
    assert "127.0.0.1:8770/api/v1/recordings/status" in api_script
    assert "127.0.0.1:8770/api/v1/recordings/commands" in api_script


def test_storage_page_groups_real_playable_clips_by_session() -> None:
    html = (STATIC_DIR / "storage.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "storage.js").read_text(encoding="utf-8")
    api_script = (STATIC_DIR / "recordings-api.js").read_text(encoding="utf-8")

    assert '<a class="nav-item is-active" href="/storage" aria-current="page"' in html
    assert 'id="session-list"' in html
    assert "library.sessions.forEach" in script
    assert "session.clips.forEach" in script
    assert 'document.createElement("video")' in script
    assert "video.controls = true" in script
    assert "video.src = clip.media_url" in script
    assert "innerHTML" not in script
    assert "127.0.0.1:8770/api/v1/recordings/library" in api_script
    assert "mediaUrl.origin !== gatewayOrigin" in api_script


def test_connected_gateway_without_video_is_presented_as_waiting() -> None:
    home_script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    storage_script = (STATIC_DIR / "storage.js").read_text(encoding="utf-8")

    assert 'unavailable: "等待 Glass3 视频"' in home_script
    assert 'unavailable: "等待 Glass3 视频"' in storage_script
    assert 'state.recordingPollError = error.message' in home_script
    assert 'elements.recordingLabel.textContent = "录制服务未连接"' in storage_script
