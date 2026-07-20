import {
  getSessionDisplayName,
  readJsonResponse,
  readRecordingLibrary,
  readRecordingStatus,
  recordingDeleteEndpoint,
  recordingLibraryEndpoint,
  recordingSessionEndpoint,
  recordingStatusEndpoint,
} from "./recordings-api.js";

const elements = {
  recordingDot: document.querySelector("#storage-recording-dot"),
  recordingLabel: document.querySelector("#storage-recording-label"),
  recordingDetail: document.querySelector("#storage-recording-detail"),
  refreshButton: document.querySelector("#refresh-library-button"),
  title: document.querySelector("#library-title"),
  summary: document.querySelector("#library-summary"),
  statusPill: document.querySelector("#library-status-pill"),
  backButton: document.querySelector("#session-back-button"),
  loading: document.querySelector("#library-loading"),
  empty: document.querySelector("#library-empty"),
  error: document.querySelector("#library-error"),
  errorDetail: document.querySelector("#library-error-detail"),
  sessionList: document.querySelector("#session-list"),
  deleteDialog: document.querySelector("#delete-recording-dialog"),
  deleteDialogTitle: document.querySelector("#delete-dialog-title"),
  deleteWarning: document.querySelector("#delete-dialog-warning"),
  deleteTarget: document.querySelector("#delete-dialog-target"),
  deleteError: document.querySelector("#delete-dialog-error"),
  confirmDeleteButton: document.querySelector("#confirm-delete-button"),
  cancelDeleteButton: document.querySelector("#cancel-delete-button"),
  closeDeleteDialogButton: document.querySelector("#close-delete-dialog-button"),
  renameDialog: document.querySelector("#rename-session-dialog"),
  renameForm: document.querySelector("#rename-session-form"),
  renameInput: document.querySelector("#session-name-input"),
  renameError: document.querySelector("#rename-dialog-error"),
  confirmRenameButton: document.querySelector("#confirm-rename-button"),
  cancelRenameButton: document.querySelector("#cancel-rename-button"),
  closeRenameDialogButton: document.querySelector("#close-rename-dialog-button"),
};

const recordingLabels = {
  unavailable: "等待 Glass3 视频",
  ready: "录制服务已就绪",
  countdown: "录制倒计时中",
  recording: "正在录制视频",
  finalizing: "正在保存视频",
  error: "录制服务异常",
};
const sessionStateLabels = {
  active: "采集中",
  finalizing: "正在完成",
  complete: "已完成",
  incomplete: "异常中断",
};

let statusTimer = null;
let libraryTimer = null;
let statusInFlight = false;
let libraryInFlight = false;
let deleteInFlight = false;
let renameInFlight = false;
let pendingDelete = null;
let pendingRenameSessionId = null;
let selectedSessionId = null;
let librarySnapshot = null;

if (window.location.search) {
  window.history.replaceState({}, "", window.location.pathname);
}

function formatDateTime(unixMs) {
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(new Date(unixMs));
}

function formatDuration(durationMs) {
  const totalSeconds = Math.max(0, Math.round(durationMs / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function formatFileSize(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function getSessionDuration(session) {
  return session.clips.reduce((total, clip) => total + clip.duration_ms, 0);
}

function isLegacyVideoOnlySession(session) {
  return session.telemetry_database === null;
}

function formatMetadataCoverage(quality) {
  if (quality.metadata_match_coverage === null) return "暂无帧元数据";
  return `${(quality.metadata_match_coverage * 100).toFixed(1)}%`;
}

function createSessionStateBadge(session) {
  const badge = createTextElement(
    "span",
    `session-state-badge session-state-${session.state}`,
    sessionStateLabels[session.state],
  );
  if (session.state === "incomplete" && session.recoverable) {
    badge.title = "异常中断，会话数据可恢复检查";
  }
  return badge;
}

function createQualityMetric(label, value, detail = "") {
  const item = document.createElement("div");
  item.className = "session-quality-metric";
  item.append(
    createTextElement("dt", "", label),
    createTextElement("dd", "", value),
    createTextElement("small", "", detail),
  );
  return item;
}

function createTextElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.textContent = text;
  return element;
}

function createRenameButton(session, { withLabel = false } = {}) {
  const button = document.createElement("button");
  button.className = withLabel
    ? "button button-small session-detail-rename"
    : "button button-icon session-rename-button";
  button.type = "button";
  button.title = "重命名会话夹";
  button.setAttribute("aria-label", `重命名会话夹 ${getSessionDisplayName(session)}`);
  button.append(createTextElement("span", "", "✎"));
  if (withLabel) button.append(document.createTextNode("重命名"));
  button.addEventListener("click", () => openRenameDialog(session));
  return button;
}

function createDeleteSessionButton(session, { withLabel = false } = {}) {
  const button = document.createElement("button");
  button.className = withLabel
    ? "button button-small session-detail-delete"
    : "button button-icon session-delete-button";
  button.type = "button";
  const deletionBlocked = ["active", "finalizing"].includes(session.state);
  button.title = deletionBlocked ? "活动或正在完成的会话不能删除" : "删除整个会话夹";
  button.setAttribute("aria-label", `删除会话夹 ${getSessionDisplayName(session)}`);
  button.disabled = deletionBlocked;
  button.append(createTextElement("span", "", "×"));
  if (withLabel) button.append(document.createTextNode("删除会话"));
  button.addEventListener("click", () => openDeleteSessionDialog(session));
  return button;
}

function openSession(sessionId) {
  selectedSessionId = sessionId;
  if (librarySnapshot !== null) renderLibrary(librarySnapshot);
}

function closeSession() {
  selectedSessionId = null;
  if (librarySnapshot !== null) renderLibrary(librarySnapshot);
}

function renderSessionFolder(session) {
  const item = document.createElement("article");
  item.className = "session-folder";

  const openButton = document.createElement("button");
  openButton.className = "session-folder-open";
  openButton.type = "button";
  openButton.setAttribute("aria-label", `打开会话夹 ${getSessionDisplayName(session)}`);

  const folderMark = createTextElement("span", "session-folder-mark", "DIR");
  folderMark.setAttribute("aria-hidden", "true");
  const identity = document.createElement("span");
  identity.className = "session-folder-identity";
  identity.append(
    createTextElement("strong", "", getSessionDisplayName(session)),
    createTextElement("span", "", formatDateTime(session.started_at_unix_ms)),
  );
  const stats = document.createElement("span");
  stats.className = "session-folder-stats";
  const imuSummary = isLegacyVideoOnlySession(session)
    ? "历史仅视频"
    : `${session.quality.imu_sample_count.toLocaleString("zh-CN")} IMU`;
  stats.append(
    createSessionStateBadge(session),
    createTextElement("span", "", `${session.clips.length} 段视频`),
    createTextElement("span", "", imuSummary),
    createTextElement("span", "", formatDuration(getSessionDuration(session))),
    createTextElement("span", "session-folder-chevron", "›"),
  );
  openButton.append(folderMark, identity, stats);
  openButton.addEventListener("click", () => openSession(session.session_id));

  const actions = document.createElement("div");
  actions.className = "session-folder-actions";
  actions.append(createRenameButton(session), createDeleteSessionButton(session));
  item.append(openButton, actions);
  return item;
}

function openDeleteDialog(session, clip, clipIndex) {
  if (deleteInFlight) return;
  pendingDelete = {
    kind: "clip",
    session_id: session.session_id,
    clip_id: clip.clip_id,
  };
  elements.deleteDialogTitle.textContent = "删除视频片段";
  elements.deleteWarning.textContent = "删除后无法恢复，本地 MP4 文件也会同时移除。";
  elements.confirmDeleteButton.textContent = "删除片段";
  elements.deleteTarget.textContent = [
    getSessionDisplayName(session),
    `片段 ${String(clipIndex + 1).padStart(2, "0")}`,
    formatDateTime(clip.recorded_at_unix_ms),
  ].join(" · ");
  elements.deleteError.textContent = "";
  elements.deleteError.hidden = true;
  elements.deleteDialog.showModal();
}

function openDeleteSessionDialog(session) {
  if (deleteInFlight || ["active", "finalizing"].includes(session.state)) return;
  pendingDelete = {
    kind: "session",
    session_id: session.session_id,
  };
  elements.deleteDialogTitle.textContent = "删除整个会话夹";
  elements.deleteWarning.textContent =
    "删除后无法恢复，会话内的视频、IMU、帧元数据和质量记录都会同时移除。";
  elements.deleteTarget.textContent = [
    getSessionDisplayName(session),
    `${session.clips.length} 段视频`,
    `${session.quality.imu_sample_count.toLocaleString("zh-CN")} IMU 样本`,
  ].join(" · ");
  elements.deleteError.textContent = "";
  elements.deleteError.hidden = true;
  elements.confirmDeleteButton.textContent = "删除整个会话";
  elements.deleteDialog.showModal();
}

function openRenameDialog(session) {
  if (renameInFlight) return;
  pendingRenameSessionId = session.session_id;
  elements.renameInput.value = getSessionDisplayName(session);
  elements.renameError.textContent = "";
  elements.renameError.hidden = true;
  elements.renameDialog.showModal();
  elements.renameInput.focus();
  elements.renameInput.select();
}

function renderClip(session, clip, clipIndex) {
  const item = document.createElement("article");
  item.className = "clip-row";

  const video = document.createElement("video");
  video.className = "clip-player";
  video.controls = true;
  video.preload = "metadata";
  video.playsInline = true;
  video.src = clip.media_url;
  video.setAttribute("aria-label", `录制片段 ${clipIndex + 1}`);

  const details = document.createElement("div");
  details.className = "clip-details";
  const heading = document.createElement("div");
  heading.className = "clip-heading";
  const actions = document.createElement("div");
  actions.className = "clip-actions";
  const duration = createTextElement(
    "span",
    "clip-duration",
    formatDuration(clip.duration_ms),
  );
  const deleteButton = createTextElement("button", "clip-delete-button", "删除");
  deleteButton.type = "button";
  deleteButton.title = "删除本地视频片段";
  deleteButton.setAttribute("aria-label", `删除片段 ${clipIndex + 1}`);
  deleteButton.addEventListener("click", () => {
    openDeleteDialog(session, clip, clipIndex);
  });
  actions.append(duration, deleteButton);
  heading.append(
    createTextElement(
      "strong",
      "",
      `片段 ${String(clipIndex + 1).padStart(2, "0")}`,
    ),
    actions,
  );
  const timestamp = createTextElement(
    "time",
    "clip-timestamp",
    formatDateTime(clip.recorded_at_unix_ms),
  );
  timestamp.dateTime = new Date(clip.recorded_at_unix_ms).toISOString();
  const metadata = document.createElement("dl");
  metadata.className = "clip-metadata";
  for (const [label, value] of [
    ["画面", `${clip.width} × ${clip.height}`],
    ["帧率", `${clip.fps.toFixed(2).replace(/\.00$/, "")} FPS`],
    ["大小", formatFileSize(clip.file_size_bytes)],
  ]) {
    const row = document.createElement("div");
    row.append(createTextElement("dt", "", label), createTextElement("dd", "", value));
    metadata.append(row);
  }
  details.append(heading, timestamp, metadata);
  item.append(video, details);
  return item;
}

function renderSessionDetail(session) {
  elements.sessionList.dataset.view = "detail";
  elements.title.textContent = getSessionDisplayName(session);
  elements.summary.textContent = [
    `${session.clips.length} 段视频`,
    isLegacyVideoOnlySession(session)
      ? "历史仅视频"
      : `${session.quality.imu_sample_count.toLocaleString("zh-CN")} IMU 样本`,
    formatDateTime(session.started_at_unix_ms),
  ].join(" · ");
  elements.backButton.hidden = false;

  const header = document.createElement("header");
  header.className = "session-detail-header";
  const identity = document.createElement("div");
  const detailIdentity = document.createElement("span");
  detailIdentity.className = "session-detail-identity";
  detailIdentity.append(
    createTextElement("strong", "", getSessionDisplayName(session)),
    createSessionStateBadge(session),
  );
  identity.append(
    createTextElement("span", "eyebrow", "SESSION CONTENT"),
    detailIdentity,
  );
  const detailActions = document.createElement("div");
  detailActions.className = "session-detail-actions";
  detailActions.append(
    createRenameButton(session, { withLabel: true }),
    createDeleteSessionButton(session, { withLabel: true }),
  );
  header.append(identity, detailActions);

  const quality = document.createElement("dl");
  quality.className = "session-quality-summary";
  if (isLegacyVideoOnlySession(session)) {
    quality.append(
      createQualityMetric("IMU 数据", "历史仅视频", "未采集遥测数据库"),
      createQualityMetric("帧元数据", "未记录", "历史清单无逐帧索引"),
      createQualityMetric("时间同步", "不可评估", "无原始时钟映射"),
      createQualityMetric("连接分段", "未记录", "历史清单无连接事件"),
    );
  } else {
    const imuDetail = [
      `ACC ${session.quality.accelerometer_sample_count.toLocaleString("zh-CN")}`,
      `GYRO ${session.quality.gyroscope_sample_count.toLocaleString("zh-CN")}`,
    ].join(" · ");
    const anomalyCount = session.quality.imu_sequence_gap_count +
      session.quality.imu_out_of_order_sample_count +
      session.quality.telemetry_queue_overflow_count;
    quality.append(
      createQualityMetric(
        "IMU 样本",
        session.quality.imu_sample_count.toLocaleString("zh-CN"),
        session.state === "active" ? `持续保存中 · ${imuDetail}` : imuDetail,
      ),
      createQualityMetric(
        "IMU 完整性",
        anomalyCount === 0 ? "未见异常" : `${anomalyCount} 项异常`,
        `缺口 ${session.quality.imu_sequence_gap_count} · 乱序 ${session.quality.imu_out_of_order_sample_count} · 队列溢出 ${session.quality.telemetry_queue_overflow_count}`,
      ),
      createQualityMetric(
        "帧元数据",
        formatMetadataCoverage(session.quality),
        `${session.quality.recorded_video_frame_metadata_match_count} / ${session.quality.recorded_video_frame_count} 个录制帧匹配`,
      ),
      createQualityMetric(
        "时间同步",
        "原始时间已保存",
        `${session.quality.timestamp_mapping_segment_count} 个映射分段 · 尚未对齐`,
      ),
      createQualityMetric(
        "连接分段",
        session.quality.connection_segment_count.toLocaleString("zh-CN"),
        session.state === "incomplete" && session.recoverable ? "可进行恢复检查" : "WebRTC 连接记录",
      ),
    );
  }

  const clips = document.createElement("div");
  clips.className = "clip-list";
  if (session.clips.length === 0) {
    const empty = document.createElement("div");
    empty.className = "session-clips-empty";
    empty.append(
      createTextElement("strong", "", "当前会话还没有视频片段"),
      createTextElement(
        "span",
        "",
        session.state === "active" && !isLegacyVideoOnlySession(session)
          ? "IMU 仍在持续保存，开始录制后视频会出现在这里"
          : "该会话保留已有遥测和质量数据",
      ),
    );
    clips.append(empty);
  } else {
    session.clips.forEach((clip, clipIndex) => {
      clips.append(renderClip(session, clip, clipIndex));
    });
  }
  elements.sessionList.append(header, quality, clips);
}

function renderFolderList(library) {
  const clipCount = library.sessions.reduce(
    (total, session) => total + session.clips.length,
    0,
  );
  elements.sessionList.dataset.view = "folders";
  elements.title.textContent = "本地会话夹";
  const imuCount = library.sessions.reduce(
    (total, session) => total + session.quality.imu_sample_count,
    0,
  );
  elements.summary.textContent =
    `${library.sessions.length} 个会话夹 · ${clipCount} 段视频 · ${imuCount.toLocaleString("zh-CN")} IMU 样本`;
  elements.backButton.hidden = true;
  library.sessions.forEach((session) => {
    elements.sessionList.append(renderSessionFolder(session));
  });
}

function setLibraryView(view) {
  elements.loading.hidden = view !== "loading";
  elements.empty.hidden = view !== "empty";
  elements.error.hidden = view !== "error";
  elements.sessionList.hidden = view !== "content";
  elements.statusPill.textContent = view === "loading" ? "LOADING" : view.toUpperCase();
  elements.statusPill.classList.toggle("pill-success", view === "content");
}

function renderLibrary(library) {
  librarySnapshot = library;
  elements.sessionList.replaceChildren();
  if (library.sessions.length === 0) {
    selectedSessionId = null;
    elements.title.textContent = "本地会话夹";
    elements.summary.textContent = "0 个会话夹 · 0 段视频 · 0 IMU 样本";
    elements.backButton.hidden = true;
    setLibraryView("empty");
    return;
  }

  const selectedSession = library.sessions.find(
    (session) => session.session_id === selectedSessionId,
  );
  if (selectedSession) {
    renderSessionDetail(selectedSession);
  } else {
    selectedSessionId = null;
    renderFolderList(library);
  }
  setLibraryView("content");
}

function renderRecordingStatus(status) {
  elements.recordingDot.dataset.state = status.state;
  elements.recordingLabel.textContent = recordingLabels[status.state];
  const output = `${status.output.width} × ${status.output.height} · ${status.output.fps} FPS · ${status.output.video_codec.toUpperCase()} ${status.output.container.toUpperCase()}`;
  elements.recordingDetail.textContent = status.session_state === "active"
    ? `采集会话进行中 · ${status.detail || output}`
    : status.detail || output;
}

function scheduleStatusPoll(delayMs = 500) {
  window.clearTimeout(statusTimer);
  if (!document.hidden) statusTimer = window.setTimeout(pollRecordingStatus, delayMs);
}

function scheduleLibraryPoll(delayMs = 5000) {
  window.clearTimeout(libraryTimer);
  if (!document.hidden) libraryTimer = window.setTimeout(pollLibrary, delayMs);
}

async function pollRecordingStatus() {
  if (statusInFlight || document.hidden) return scheduleStatusPoll();
  statusInFlight = true;
  try {
    const response = await fetch(recordingStatusEndpoint, { cache: "no-store" });
    const payload = await readJsonResponse(response, `录制状态 HTTP ${response.status}`);
    renderRecordingStatus(readRecordingStatus(payload));
  } catch (error) {
    elements.recordingDot.dataset.state = "unavailable";
    elements.recordingLabel.textContent = "录制服务未连接";
    elements.recordingDetail.textContent = error.message;
  } finally {
    statusInFlight = false;
    scheduleStatusPoll();
  }
}

async function pollLibrary({ showLoading = false } = {}) {
  if (libraryInFlight || document.hidden) return;
  libraryInFlight = true;
  elements.refreshButton.disabled = true;
  if (showLoading) setLibraryView("loading");
  try {
    const response = await fetch(recordingLibraryEndpoint, { cache: "no-store" });
    const payload = await readJsonResponse(response, `媒体库 HTTP ${response.status}`);
    renderLibrary(readRecordingLibrary(payload));
  } catch (error) {
    elements.summary.textContent = "媒体库不可用";
    elements.errorDetail.textContent = error.message;
    setLibraryView("error");
  } finally {
    libraryInFlight = false;
    elements.refreshButton.disabled = false;
    scheduleLibraryPoll();
  }
}

async function confirmDeleteTarget() {
  if (deleteInFlight || pendingDelete === null) return;
  const target = pendingDelete;
  deleteInFlight = true;
  elements.confirmDeleteButton.disabled = true;
  elements.cancelDeleteButton.disabled = true;
  elements.closeDeleteDialogButton.disabled = true;
  elements.confirmDeleteButton.textContent = "正在删除";
  elements.deleteError.hidden = true;
  try {
    const endpoint = target.kind === "session"
      ? recordingSessionEndpoint(target.session_id)
      : recordingDeleteEndpoint(target.session_id, target.clip_id);
    const response = await fetch(
      endpoint,
      { method: "DELETE", cache: "no-store" },
    );
    const operation = target.kind === "session" ? "删除会话" : "删除视频";
    const payload = await readJsonResponse(response, `${operation} HTTP ${response.status}`);
    const library = readRecordingLibrary(payload);
    pendingDelete = null;
    elements.deleteDialog.close("deleted");
    renderLibrary(library);
    scheduleLibraryPoll();
  } catch (error) {
    elements.deleteError.textContent = error.message;
    elements.deleteError.hidden = false;
  } finally {
    deleteInFlight = false;
    elements.confirmDeleteButton.disabled = false;
    elements.cancelDeleteButton.disabled = false;
    elements.closeDeleteDialogButton.disabled = false;
    elements.confirmDeleteButton.textContent = "删除片段";
  }
}

async function confirmRenameSession(event) {
  event.preventDefault();
  if (renameInFlight || pendingRenameSessionId === null) return;
  const displayName = elements.renameInput.value.trim();
  if (!displayName || /[\u0000-\u001f\u007f]/u.test(displayName)) {
    elements.renameError.textContent = "请输入有效的会话夹名称";
    elements.renameError.hidden = false;
    return;
  }

  renameInFlight = true;
  elements.renameInput.disabled = true;
  elements.confirmRenameButton.disabled = true;
  elements.cancelRenameButton.disabled = true;
  elements.closeRenameDialogButton.disabled = true;
  elements.confirmRenameButton.textContent = "正在保存";
  elements.renameError.hidden = true;
  try {
    const response = await fetch(
      recordingSessionEndpoint(pendingRenameSessionId),
      {
        method: "PATCH",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ display_name: displayName }),
      },
    );
    const payload = await readJsonResponse(response, `重命名会话 HTTP ${response.status}`);
    const library = readRecordingLibrary(payload);
    pendingRenameSessionId = null;
    elements.renameDialog.close("renamed");
    renderLibrary(library);
    scheduleLibraryPoll();
  } catch (error) {
    elements.renameError.textContent = error.message;
    elements.renameError.hidden = false;
  } finally {
    renameInFlight = false;
    elements.renameInput.disabled = false;
    elements.confirmRenameButton.disabled = false;
    elements.cancelRenameButton.disabled = false;
    elements.closeRenameDialogButton.disabled = false;
    elements.confirmRenameButton.textContent = "保存名称";
  }
}

function closeRenameDialog() {
  if (!renameInFlight) elements.renameDialog.close("cancel");
}

elements.refreshButton.addEventListener("click", () => pollLibrary({ showLoading: true }));
elements.backButton.addEventListener("click", closeSession);
elements.confirmDeleteButton.addEventListener("click", confirmDeleteTarget);
elements.renameForm.addEventListener("submit", confirmRenameSession);
elements.cancelRenameButton.addEventListener("click", closeRenameDialog);
elements.closeRenameDialogButton.addEventListener("click", closeRenameDialog);
elements.deleteDialog.addEventListener("cancel", (event) => {
  if (deleteInFlight) event.preventDefault();
});
elements.deleteDialog.addEventListener("close", () => {
  if (!deleteInFlight) pendingDelete = null;
});
elements.renameDialog.addEventListener("cancel", (event) => {
  if (renameInFlight) event.preventDefault();
});
elements.renameDialog.addEventListener("close", () => {
  if (!renameInFlight) pendingRenameSessionId = null;
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    scheduleStatusPoll(0);
    pollLibrary();
  }
});
window.addEventListener("beforeunload", () => {
  window.clearTimeout(statusTimer);
  window.clearTimeout(libraryTimer);
});

pollRecordingStatus();
pollLibrary({ showLoading: true });
