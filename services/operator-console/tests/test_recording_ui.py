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
    assert "sendRecordingCommand(elements.recordingToggleButton.dataset.action)" in script
    assert "body: JSON.stringify({ action })" in script
    assert "Math.ceil(remainingMs / 1000)" in script
    assert "Date.now()" in script
    assert (
        "elements.streamToggleButton.disabled = !controlReady || busy || recordingActive" in script
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
        "collectionSessionStates.has(payload.session_state)",
        "payload.recording_starts_at_unix_ms",
        "payload.recording_started_at_unix_ms",
        "payload.output",
        "output.width !== 1280",
        "output.height !== 720",
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
        "readSessionQuality(session.quality)",
        '"telemetry/telemetry.sqlite"',
        'quality.timestamp_alignment_state !== "unverified"',
        "quality.metadata_match_coverage",
        "quality.recorded_video_frame_metadata_match_count",
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
    assert 'elements.deleteDialog.close("deleted")' in script
    assert "recordingDeleteEndpoint(target.session_id, target.clip_id)" in script
    assert 'target.kind === "session"' in script
    assert "recordingSessionEndpoint(target.session_id)" in script
    assert "openDeleteSessionDialog(session)" in script
    assert '["active", "finalizing"].includes(session.state)' in script
    assert "视频、IMU、帧元数据和质量记录" in script
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
    assert 'maxlength="128"' in html
    assert "getSessionDisplayName(session)" in script
    assert "formatSessionFolderName(session.started_at_unix_ms)" in api
    assert 'item.className = "session-folder"' in script
    assert 'openButton.addEventListener("click"' in script
    assert "openRenameDialog(session)" in script
    assert "recordingSessionEndpoint(pendingRenameSessionId)" in script
    assert 'method: "PATCH"' in script
    assert "body: JSON.stringify({ display_name: displayName })" in script
    assert 'elements.renameDialog.close("renamed")' in script
    assert "/api/v1/recordings/sessions/${sessionId}" in api
    assert ".session-folder-open" in styles
    assert ".session-detail-header" in styles
    assert ".rename-field input" in styles


def test_missing_video_is_distinct_from_gateway_disconnect() -> None:
    home_script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    storage_script = (STATIC_DIR / "storage.js").read_text(encoding="utf-8")

    assert 'unavailable: "等待 Glass3 视频"' in home_script
    assert "state.recordingPollError = error.message" in home_script
    assert 'unavailable: "等待 Glass3 视频"' in storage_script
    assert 'elements.recordingLabel.textContent = "录制服务未连接"' in storage_script


def test_home_collection_session_is_gateway_backed_and_never_invented() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    api = (STATIC_DIR / "recordings-api.js").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    for element_id in (
        "new-session-button",
        "current-session-name",
        "current-session-state",
        "current-session-imu",
        "current-session-metadata",
        "current-session-sync",
    ):
        assert f'id="{element_id}"' in html
    assert "recordingLibraryEndpoint" in script
    assert "readRecordingLibrary(payload)" in script
    assert "findCurrentSession()" in script
    assert "getSessionDisplayName(session)" in script
    assert "session.quality" in script
    assert 'body: JSON.stringify({ action: "new" })' in script
    assert "recordingSessionCommandEndpoint" in script
    assert 'sessionState !== "active"' in script
    assert '["countdown", "recording", "finalizing"]' in script
    assert "下一次录制会自动开始新会话并保存 IMU" in script
    assert "api/v1/recordings/session-commands" in api
    assert ".session-overview" in styles
    assert ".session-quality-grid" in styles


def test_storage_keeps_zero_clip_telemetry_sessions_and_exposes_quality() -> None:
    html = (STATIC_DIR / "storage.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "storage.js").read_text(encoding="utf-8")
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert "采集数据" in html
    assert "session.quality.imu_sample_count" in script
    assert "formatMetadataCoverage(session.quality)" in script
    assert "感知阶段待处理" in script
    assert "源时间与 MP4 PTS 已保留，采集阶段不生成映射" in script
    assert "session.quality.telemetry_queue_overflow_count" in script
    assert "session.clips.length === 0" in script
    assert "IMU 仍在持续保存" in script
    assert "历史仅视频" in script
    assert "session.telemetry_database === null" in script
    assert 'session.state === "incomplete" && session.recoverable' in script
    assert ".session-quality-summary" in styles
    assert ".session-clips-empty" in styles
