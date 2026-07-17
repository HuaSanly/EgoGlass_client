import {
  readJsonResponse,
  readRecordingLibrary,
  readRecordingStatus,
  recordingLibraryEndpoint,
  recordingStatusEndpoint,
} from "./recordings-api.js";

const elements = {
  recordingDot: document.querySelector("#storage-recording-dot"),
  recordingLabel: document.querySelector("#storage-recording-label"),
  recordingDetail: document.querySelector("#storage-recording-detail"),
  refreshButton: document.querySelector("#refresh-library-button"),
  summary: document.querySelector("#library-summary"),
  statusPill: document.querySelector("#library-status-pill"),
  loading: document.querySelector("#library-loading"),
  empty: document.querySelector("#library-empty"),
  error: document.querySelector("#library-error"),
  errorDetail: document.querySelector("#library-error-detail"),
  sessionList: document.querySelector("#session-list"),
};

const recordingLabels = {
  unavailable: "等待 Glass3 视频",
  ready: "录制服务已就绪",
  countdown: "录制倒计时中",
  recording: "正在录制视频",
  finalizing: "正在保存视频",
  error: "录制服务异常",
};

let statusTimer = null;
let libraryTimer = null;
let statusInFlight = false;
let libraryInFlight = false;

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

function createTextElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.textContent = text;
  return element;
}

function renderClip(clip, index) {
  const item = document.createElement("article");
  item.className = "clip-row";

  const video = document.createElement("video");
  video.className = "clip-player";
  video.controls = true;
  video.preload = "metadata";
  video.playsInline = true;
  video.src = clip.media_url;
  video.setAttribute("aria-label", `录制片段 ${index + 1}`);

  const details = document.createElement("div");
  details.className = "clip-details";
  const heading = document.createElement("div");
  heading.className = "clip-heading";
  heading.append(
    createTextElement("strong", "", `片段 ${String(index + 1).padStart(2, "0")}`),
    createTextElement("span", "", formatDuration(clip.duration_ms)),
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

function renderSession(session, index) {
  const section = document.createElement("section");
  section.className = "session-group";
  section.setAttribute("aria-labelledby", `session-title-${index}`);

  const header = document.createElement("header");
  header.className = "session-header";
  const titleGroup = document.createElement("div");
  const title = createTextElement("h3", "", `会话 ${String(index + 1).padStart(2, "0")}`);
  title.id = `session-title-${index}`;
  const time = createTextElement("time", "", formatDateTime(session.started_at_unix_ms));
  time.dateTime = new Date(session.started_at_unix_ms).toISOString();
  titleGroup.append(title, time);
  header.append(
    titleGroup,
    createTextElement("span", "session-clip-count", `${session.clips.length} 段视频`),
  );

  const clips = document.createElement("div");
  clips.className = "clip-list";
  if (session.clips.length === 0) {
    clips.append(createTextElement("p", "session-empty", "该会话没有可播放的视频片段"));
  } else {
    session.clips.forEach((clip, clipIndex) => clips.append(renderClip(clip, clipIndex)));
  }
  section.append(header, clips);
  return section;
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
  elements.sessionList.replaceChildren();
  const clipCount = library.sessions.reduce((total, session) => total + session.clips.length, 0);
  elements.summary.textContent = `${library.sessions.length} 次会话 · ${clipCount} 段视频`;
  if (library.sessions.length === 0) {
    setLibraryView("empty");
    return;
  }
  library.sessions.forEach((session, index) => {
    elements.sessionList.append(renderSession(session, index));
  });
  setLibraryView("content");
}

function renderRecordingStatus(status) {
  elements.recordingDot.dataset.state = status.state;
  elements.recordingLabel.textContent = recordingLabels[status.state];
  const output = `${status.output.width} × ${status.output.height} · ${status.output.fps} FPS · ${status.output.video_codec.toUpperCase()} ${status.output.container.toUpperCase()}`;
  elements.recordingDetail.textContent = status.detail || output;
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

elements.refreshButton.addEventListener("click", () => pollLibrary({ showLoading: true }));
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
