from operator_console.app import STATIC_DIR


def test_recording_workflow_is_gateway_backed_and_countdown_is_authoritative() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    api_script = (STATIC_DIR / "recordings-api.js").read_text(encoding="utf-8")

    assert 'id="recording-toggle-button"' in html
    assert 'id="recording-countdown"' in html
    assert "recordingStatusEndpoint" in app_script
    assert "recordingCommandEndpoint" in app_script
    assert 'method: "POST"' in app_script
    assert "body: JSON.stringify({ action })" in app_script
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
    assert "renderFolderList(library)" in script
    assert "renderSessionFolder(session)" in script
    assert "renderSessionDetail(selectedSession)" in script
    assert "selectedSessionId = sessionId" in script
    assert "session.clips.forEach" in script
    assert 'document.createElement("video")' in script
    assert "video.controls = true" in script
    assert "video.src = clip.media_url" in script
    assert "renderedLibraryRevision = JSON.stringify(library)" in script
    assert "renderLibraryIfChanged(readRecordingLibrary(payload))" in script
    assert '["画面", `${clip.width} × ${clip.height}`]' in script
    assert "innerHTML" not in script
    assert "127.0.0.1:8770/api/v1/recordings/library" in api_script
    assert "mediaUrl.origin !== gatewayOrigin" in api_script


def test_clip_deletion_is_confirmed_and_manifest_backed() -> None:
    html = (STATIC_DIR / "storage.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "storage.js").read_text(encoding="utf-8")
    api_script = (STATIC_DIR / "recordings-api.js").read_text(encoding="utf-8")

    assert 'id="delete-recording-dialog"' in html
    assert "删除后无法恢复" in html
    assert "openDeleteDialog(session, clip, clipIndex)" in script
    assert "elements.deleteDialog.showModal()" in script
    assert "confirmDeleteTarget" in script
    assert 'method: "DELETE"' in script
    assert "renderLibrary(library)" in script
    assert "recordingIdPattern.test(sessionId)" in api_script
    assert "recordingIdPattern.test(clipId)" in api_script
    assert "api/v1/recordings/clips" in api_script


def test_session_folders_are_time_named_navigable_and_renameable() -> None:
    html = (STATIC_DIR / "storage.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "storage.js").read_text(encoding="utf-8")
    api_script = (STATIC_DIR / "recordings-api.js").read_text(encoding="utf-8")

    assert 'id="session-back-button"' in html
    assert 'id="rename-session-dialog"' in html
    assert 'id="session-name-input"' in html
    assert "getSessionDisplayName(session)" in script
    assert "formatSessionFolderName(session.started_at_unix_ms)" in api_script
    assert "getSessionDisplayName(session)" in script
    assert 'elements.sessionList.dataset.view = "folders"' in script
    assert 'elements.sessionList.dataset.view = "detail"' in script
    assert "recordingSessionEndpoint(pendingRenameSessionId)" in script
    assert 'method: "PATCH"' in script
    assert "readRecordingLibrary(payload)" in script
    assert "session.display_name ?? null" in api_script
    assert "api/v1/recordings/sessions" in api_script


def test_connected_gateway_without_video_is_presented_as_waiting() -> None:
    home_script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    storage_script = (STATIC_DIR / "storage.js").read_text(encoding="utf-8")

    assert 'unavailable: "等待 Glass3 视频"' in home_script
    assert 'unavailable: "等待 Glass3 视频"' in storage_script
    assert "state.recordingPollError = error.message" in home_script
    assert 'elements.recordingLabel.textContent = "录制服务未连接"' in storage_script


def test_collection_session_switch_is_authoritative_and_defers_creation() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    app_script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    api_script = (STATIC_DIR / "recordings-api.js").read_text(encoding="utf-8")

    assert 'id="new-session-button"' in html
    assert 'id="current-session-imu"' in html
    assert "recordingSessionCommandEndpoint" in app_script
    assert 'body: JSON.stringify({ action: "new" })' in app_script
    assert 'sessionState !== "active"' in app_script
    assert '["countdown", "recording", "finalizing"]' in app_script
    assert "下一次录制会自动开始新会话并保存 IMU" in app_script
    assert "api/v1/recordings/session-commands" in api_script
    assert "window.setTimeout(pollCollectionLibrary, delayMs)" in app_script


def test_unified_storage_preserves_telemetry_only_and_legacy_sessions() -> None:
    script = (STATIC_DIR / "storage.js").read_text(encoding="utf-8")
    api_script = (STATIC_DIR / "recordings-api.js").read_text(encoding="utf-8")

    assert "library.sessions.forEach" in script
    assert "session.clips.length === 0" in script
    assert "IMU 仍在持续保存" in script
    assert "历史仅视频" in script
    assert "session.quality.imu_sample_count" in script
    assert "session.quality.recorded_video_frame_metadata_match_count" in script
    assert "源时间与 MP4 PTS 已保留，采集阶段不生成映射" in script
    assert "quality.timestamp_mapping_segment_count" in api_script
    assert 'quality.timestamp_alignment_state !== "unverified"' in api_script
    assert '"telemetry/telemetry.sqlite"' in api_script
    assert "readSessionQuality(session.quality)" in api_script


def test_collection_session_delete_is_confirmed_and_active_safe() -> None:
    html = (STATIC_DIR / "storage.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "storage.js").read_text(encoding="utf-8")

    assert 'id="delete-dialog-warning"' in html
    assert "openDeleteSessionDialog(session)" in script
    assert '["active", "finalizing"].includes(session.state)' in script
    assert 'target.kind === "session"' in script
    assert "recordingSessionEndpoint(target.session_id)" in script
    assert 'elements.deleteDialog.close("deleted")' in script
    assert "视频、IMU、帧元数据和质量记录" in script
