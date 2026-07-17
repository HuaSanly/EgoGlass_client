from html.parser import HTMLParser

from starlette.testclient import TestClient

from egoglass_operator_console.app import STATIC_DIR, create_app


class MediaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.video_ids: list[str] = []
        self.current_links: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        if tag == "video" and attributes.get("id"):
            self.video_ids.append(attributes["id"] or "")
        if tag == "a" and attributes.get("aria-current") == "page":
            self.current_links.append(attributes.get("href") or "")


def test_storage_route_is_a_real_second_page() -> None:
    with TestClient(create_app()) as client:
        home = client.get("/")
        storage = client.get("/storage")
        storage_script = client.get("/assets/storage.js")
        recordings_api = client.get("/assets/recordings-api.js")

    assert home.status_code == 200
    assert storage.status_code == 200
    assert storage_script.status_code == 200
    assert recordings_api.status_code == 200
    parser = MediaParser()
    parser.feed(storage.text)
    assert parser.current_links == ["/storage"]
    assert 'id="session-list"' in storage.text
    assert 'id="library-loading"' in storage.text
    assert 'id="library-empty"' in storage.text
    assert 'id="library-error"' in storage.text
    assert 'id="session-back-button"' in storage.text
    assert 'id="rename-session-dialog"' in storage.text
    assert 'src="/assets/storage.js"' in storage.text


def test_home_uses_two_state_stream_and_recording_controls() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert 'id="stream-toggle-button"' in html
    assert 'id="recording-toggle-button"' in html
    assert 'id="recording-countdown"' in html
    assert 'id="recording-countdown-value">3<' in html
    assert 'id="start-stream-button"' not in html
    assert 'id="stop-stream-button"' not in html
    assert 'dataset.action = shouldStop ? "stop" : "start"' in script
    assert 'sendRecordingCommand(elements.recordingToggleButton.dataset.action)' in script
    assert 'body: JSON.stringify({ action })' in script
    assert 'Math.ceil(remainingMs / 1000)' in script
    assert "Date.now()" in script
    assert (
        "elements.streamToggleButton.disabled = !controlReady || busy || recordingActive"
        in script
    )
    assert html.count('class="stream-control-icon"') == 2
    assert html.count('class="stream-control-label"') == 2
    assert ".stream-control-icon {" in styles
    assert "flex: 0 0 14px" in styles
    assert ".stream-control-label {" in styles
    assert "white-space: nowrap" in styles
    assert ".stream-control-button span {" not in styles


def test_recording_contract_is_strict_and_contains_no_simulation() -> None:
    api = (STATIC_DIR / "recordings-api.js").read_text(encoding="utf-8")

    for required in (
        'payload.schema_version !== "1.0"',
        "recordingStates.has(payload.state)",
        "payload.recording_starts_at_unix_ms",
        "payload.recording_started_at_unix_ms",
        "payload.output",
        "output.width !== 1920",
        "output.height !== 1080",
        "output.fps !== 30",
        'output.container !== "mp4"',
        'output.video_codec !== "h264"',
        "Array.isArray(payload.sessions)",
        "session.clips.map(readClip)",
        "readMediaUrl(clip.media_url)",
        "recordingDeleteEndpoint(sessionId, clipId)",
        "recordingSessionEndpoint(sessionId)",
        "recordingIdPattern.test(session.session_id)",
        "recordingIdPattern.test(clip.clip_id)",
        "session.display_name ?? null",
        "displayName !== displayName.trim()",
        "mediaUrl.origin !== gatewayOrigin",
    ):
        assert required in api
    assert "simulation" not in api.lower()
    assert "synthetic" not in api.lower()
    assert "mock" not in api.lower()


def test_storage_library_builds_safe_playable_media_without_document_scroll() -> None:
    script = (STATIC_DIR / "storage.js").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert "video.controls = true" in script
    assert 'video.preload = "metadata"' in script
    assert "video.src = clip.media_url" in script
    assert "readRecordingLibrary(payload)" in script
    assert "renderSessionFolder(session)" in script
    assert "renderSessionDetail(selectedSession)" in script
    assert "selectedSessionId = sessionId" in script
    assert 'elements.sessionList.dataset.view = "folders"' in script
    assert 'elements.sessionList.dataset.view = "detail"' in script
    assert "innerHTML" not in script
    assert "height: 100vh" in styles
    assert "overflow: hidden" in styles
    assert ".session-list" in styles
    assert "overflow-y: auto" in styles


def test_storage_delete_requires_confirmation_and_gateway_success() -> None:
    html = (STATIC_DIR / "storage.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "storage.js").read_text(encoding="utf-8")
    api = (STATIC_DIR / "recordings-api.js").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert 'id="delete-recording-dialog"' in html
    assert 'id="confirm-delete-button"' in html
    assert 'id="cancel-delete-button"' in html
    assert "elements.deleteDialog.showModal()" in script
    assert 'deleteButton.addEventListener("click"' in script
    assert 'method: "DELETE"' in script
    assert "readRecordingLibrary(payload)" in script
    assert "elements.deleteDialog.close(\"deleted\")" in script
    assert "recordingDeleteEndpoint(target.session_id, target.clip_id)" in script
    assert "/api/v1/recordings/clips/${sessionId}/${clipId}" in api
    assert ".clip-delete-button" in styles
    assert ".delete-dialog::backdrop" in styles


def test_storage_session_folders_use_time_names_and_persist_renames() -> None:
    html = (STATIC_DIR / "storage.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "storage.js").read_text(encoding="utf-8")
    api = (STATIC_DIR / "recordings-api.js").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert 'id="rename-session-dialog"' in html
    assert 'id="rename-session-form"' in html
    assert 'id="session-name-input"' in html
    assert 'maxlength="64"' in html
    assert "formatSessionFolderName(session.started_at_unix_ms)" in script
    assert 'item.className = "session-folder"' in script
    assert 'openButton.addEventListener("click"' in script
    assert "openRenameDialog(session)" in script
    assert "recordingSessionEndpoint(pendingRenameSessionId)" in script
    assert 'method: "PATCH"' in script
    assert 'body: JSON.stringify({ display_name: displayName })' in script
    assert 'elements.renameDialog.close("renamed")' in script
    assert "/api/v1/recordings/sessions/${sessionId}" in api
    assert ".session-folder-open" in styles
    assert ".session-detail-header" in styles
    assert ".rename-field input" in styles


def test_missing_video_is_distinct_from_gateway_disconnect() -> None:
    home_script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    storage_script = (STATIC_DIR / "storage.js").read_text(encoding="utf-8")

    assert 'unavailable: "等待 Glass3 视频"' in home_script
    assert 'state.recordingPollError = error.message' in home_script
    assert 'unavailable: "等待 Glass3 视频"' in storage_script
    assert 'elements.recordingLabel.textContent = "录制服务未连接"' in storage_script
