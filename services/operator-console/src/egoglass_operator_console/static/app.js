import { ImuSceneController } from "./imu-scene.js";
import {
  getSessionDisplayName,
  readJsonResponse,
  readRecordingLibrary,
  readRecordingStatus,
  recordingCommandEndpoint,
  recordingLibraryEndpoint,
  recordingSessionCommandEndpoint,
  recordingStatusEndpoint,
} from "./recordings-api.js";

const elements = {
  liveVideo: document.querySelector("#live-video-source"),
  viewerStage: document.querySelector("#viewer-stage"),
  viewerEmpty: document.querySelector("#viewer-empty"),
  connectionLight: document.querySelector("#connection-light"),
  connectionLabel: document.querySelector("#connection-label"),
  liveBadge: document.querySelector("#live-badge"),
  liveBadgeLabel: document.querySelector("#live-badge-label"),
  resolutionBadge: document.querySelector("#resolution-badge"),
  sourcePill: document.querySelector("#source-pill"),
  previewStatus: document.querySelector("#preview-status-property"),
  frameSize: document.querySelector("#frame-size-property"),
  previewFps: document.querySelector("#preview-fps"),
  lastFrameTime: document.querySelector("#last-frame-time"),
  controlStatusDot: document.querySelector("#control-status-dot"),
  controlStatus: document.querySelector("#stream-control-status"),
  controlError: document.querySelector("#control-error"),
  streamToggleButton: document.querySelector("#stream-toggle-button"),
  streamToggleIcon: document.querySelector("#stream-toggle-icon"),
  streamToggleLabel: document.querySelector("#stream-toggle-label"),
  recordingToggleButton: document.querySelector("#recording-toggle-button"),
  recordingToggleIcon: document.querySelector("#recording-toggle-icon"),
  recordingToggleLabel: document.querySelector("#recording-toggle-label"),
  recordingSummary: document.querySelector("#recording-summary"),
  recordingCountdown: document.querySelector("#recording-countdown"),
  recordingCountdownValue: document.querySelector("#recording-countdown-value"),
  sessionStatusDot: document.querySelector("#session-status-dot"),
  currentSessionName: document.querySelector("#current-session-name"),
  currentSessionState: document.querySelector("#current-session-state"),
  currentSessionImu: document.querySelector("#current-session-imu"),
  currentSessionMetadata: document.querySelector("#current-session-metadata"),
  currentSessionSync: document.querySelector("#current-session-sync"),
  newSessionButton: document.querySelector("#new-session-button"),
  fullscreenButton: document.querySelector("#fullscreen-button"),
  clearEventsButton: document.querySelector("#clear-events-button"),
  eventRows: document.querySelector("#event-rows"),
  eventEmpty: document.querySelector("#event-empty"),
  eventCount: document.querySelector("#event-count"),
  imuCanvas: document.querySelector("#imu-scene-canvas"),
  imuStatusPill: document.querySelector("#imu-status-pill"),
  imuEmpty: document.querySelector("#imu-empty"),
  imuEmptyDetail: document.querySelector("#imu-empty-detail"),
  imuRoll: document.querySelector("#imu-roll"),
  imuPitch: document.querySelector("#imu-pitch"),
  imuYaw: document.querySelector("#imu-yaw"),
  imuAcceleration: document.querySelector("#imu-acceleration"),
  imuAngularRate: document.querySelector("#imu-angular-rate"),
  imuAccelerationRate: document.querySelector("#imu-acceleration-rate"),
  imuGyroscopeRate: document.querySelector("#imu-gyroscope-rate"),
  resetImuButton: document.querySelector("#reset-imu-button"),
};

const state = {
  liveVideoReady: false,
  connecting: false,
  peer: null,
  reconnectTimer: null,
  frameCallbackId: null,
  fallbackFrameTimer: null,
  lastPresentedFrames: null,
  lastFpsSampleAt: null,
  lastDetailsAt: 0,
  lastSignalingError: null,
  controlState: "unavailable",
  controlDetail: null,
  controlPollTimer: null,
  controlPollInFlight: false,
  controlCommandInFlight: false,
  controlPollError: null,
  controlCommandError: null,
  recordingStatus: null,
  recordingPollTimer: null,
  recordingPollInFlight: false,
  recordingCommandInFlight: false,
  recordingPollError: null,
  recordingCommandError: null,
  recordingCountdownTimer: null,
  collectionLibrary: null,
  collectionPollTimer: null,
  collectionPollInFlight: false,
  collectionCommandInFlight: false,
  collectionPollError: null,
  collectionCommandError: null,
  imuPollTimer: null,
  imuPollInFlight: false,
  imuConnected: false,
  imuOrientationReady: false,
  imuSceneError: null,
  events: [],
};

const viewerSignalingEndpoint =
  "http://127.0.0.1:8770/api/v1/webrtc/viewer/sessions";
const streamControlEndpoint = "http://127.0.0.1:8770/api/v1/webrtc/control";
const streamControlCommandEndpoint = `${streamControlEndpoint}/commands`;
const imuStatusEndpoint = "http://127.0.0.1:8770/api/v1/webrtc/imu/status";
const streamControlStates = new Set([
  "unavailable",
  "ready",
  "starting",
  "streaming",
  "stopping",
  "stopped",
  "error",
]);
const controllableStreamStates = new Set(["ready", "streaming", "stopped"]);
const streamControlLabels = {
  unavailable: "眼镜控制未连接",
  ready: "控制通路已就绪",
  starting: "正在启动视频",
  streaming: "视频正在推流",
  stopping: "正在停止视频",
  stopped: "视频已停止",
  error: "眼镜端控制失败",
};
const recordingStateLabels = {
  unavailable: "等待 Glass3 视频",
  ready: "可录制 1280 × 720 · 30 FPS",
  countdown: "录制将在倒计时后开始",
  recording: "正在保存 1280 × 720 · 30 FPS",
  finalizing: "正在封装 MP4 文件",
  error: "录制服务异常",
};
const maxEventHistory = 50;
let imuScene = null;

if (window.location.search) {
  window.history.replaceState({}, "", window.location.pathname);
}

class ViewerSignalingError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function scheduleViewerRetry(delayMs = 1000) {
  window.clearTimeout(state.reconnectTimer);
  if (document.hidden) return;
  state.reconnectTimer = window.setTimeout(connectLiveVideo, delayMs);
}

function scheduleControlPoll(delayMs = 1000) {
  window.clearTimeout(state.controlPollTimer);
  if (document.hidden) return;
  state.controlPollTimer = window.setTimeout(pollStreamControlStatus, delayMs);
}

function scheduleRecordingPoll(delayMs = 500) {
  window.clearTimeout(state.recordingPollTimer);
  if (document.hidden) return;
  state.recordingPollTimer = window.setTimeout(pollRecordingStatus, delayMs);
}

function scheduleCollectionPoll(delayMs = 1500) {
  window.clearTimeout(state.collectionPollTimer);
  if (document.hidden) return;
  state.collectionPollTimer = window.setTimeout(pollCollectionLibrary, delayMs);
}

function readStreamControlStatus(payload) {
  if (
    payload === null ||
    typeof payload !== "object" ||
    payload.schema_version !== "1.0" ||
    payload.message_type !== "stream_control_status" ||
    !streamControlStates.has(payload.state) ||
    (payload.detail !== null && payload.detail !== undefined && typeof payload.detail !== "string")
  ) {
    throw new Error("接收网关返回了无效的控制状态");
  }
  return {
    state: payload.state,
    detail: payload.detail || null,
  };
}

function applyStreamControlStatus(payload) {
  const status = readStreamControlStatus(payload);
  const previousState = state.controlState;
  state.controlState = status.state;
  state.controlDetail = status.detail;
  state.controlPollError = null;
  renderStreamControl();

  if (previousState === status.state) return;
  if (status.state === "ready") {
    addEvent("OK", "眼镜控制通路已连接", status.detail || "可发送视频流控制命令");
  } else if (status.state === "streaming") {
    addEvent("OK", "眼镜端视频已启动", status.detail || "推流状态已确认");
  } else if (status.state === "stopped") {
    addEvent("INFO", "眼镜端视频已停止", status.detail || "控制通路保持在线");
  } else if (status.state === "error") {
    addEvent("ERROR", "眼镜端控制失败", status.detail || "设备返回控制错误");
  }
}

async function pollStreamControlStatus() {
  if (state.controlPollInFlight || state.controlCommandInFlight || document.hidden) {
    scheduleControlPoll();
    return;
  }
  state.controlPollInFlight = true;
  try {
    const response = await fetch(streamControlEndpoint, { cache: "no-store" });
    const payload = await readJsonResponse(response, `控制状态 HTTP ${response.status}`);
    applyStreamControlStatus(payload);
  } catch (error) {
    if (state.controlPollError !== error.message) {
      addEvent("WARN", "眼镜控制通路不可用", error.message);
    }
    state.controlPollError = error.message;
    state.controlState = "unavailable";
    state.controlDetail = null;
    renderStreamControl();
  } finally {
    state.controlPollInFlight = false;
    scheduleControlPoll();
  }
}

async function sendStreamControlCommand(action) {
  if (
    state.controlCommandInFlight ||
    !controllableStreamStates.has(state.controlState) ||
    (action === "start" && state.controlState === "streaming") ||
    (action === "stop" && state.controlState === "stopped")
  ) {
    return;
  }

  state.controlCommandInFlight = true;
  state.controlCommandError = null;
  renderStreamControl();
  try {
    const response = await fetch(streamControlCommandEndpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
    const payload = await readJsonResponse(response, `视频控制 HTTP ${response.status}`);
    applyStreamControlStatus(payload);
  } catch (error) {
    state.controlCommandError = error.message;
    addEvent(
      "ERROR",
      action === "start" ? "启动视频失败" : "停止视频失败",
      error.message,
    );
    console.warn("Glass3 stream command failed", error);
  } finally {
    state.controlCommandInFlight = false;
    renderStreamControl();
    scheduleControlPoll(0);
  }
}

function renderStreamControl() {
  const controlReady = controllableStreamStates.has(state.controlState);
  const busy = state.controlCommandInFlight || ["starting", "stopping"].includes(state.controlState);
  const recordingActive = ["countdown", "recording", "finalizing"].includes(
    state.recordingStatus?.state,
  );
  const shouldStop = state.controlState === "streaming";
  elements.streamToggleButton.disabled = !controlReady || busy || recordingActive;
  elements.streamToggleButton.dataset.action = shouldStop ? "stop" : "start";
  elements.streamToggleButton.title = recordingActive
    ? "停止录制后才能控制视频流"
    : shouldStop
    ? "停止眼镜端视频流"
    : "开始眼镜端视频流";
  elements.streamToggleButton.classList.toggle("stream-control-stop", shouldStop);
  elements.streamToggleButton.classList.toggle("stream-control-start", !shouldStop);
  elements.streamToggleIcon.textContent = shouldStop ? "■" : "▶";
  elements.streamToggleLabel.textContent = shouldStop ? "停止视频" : "开始视频";
  elements.controlStatus.textContent = state.controlCommandInFlight
    ? "命令发送中"
    : streamControlLabels[state.controlState];
  elements.controlStatus.title = state.controlDetail || "";
  elements.controlStatusDot.dataset.state = state.controlState;

  const visibleError = state.controlCommandError || state.recordingCommandError ||
    state.collectionCommandError || state.controlPollError || state.recordingPollError;
  elements.controlError.textContent = visibleError || "";
  elements.controlError.title = visibleError || "";
  elements.controlError.hidden = visibleError === null;
}

function applyRecordingStatus(payload) {
  const status = readRecordingStatus(payload);
  const previousState = state.recordingStatus?.state || null;
  state.recordingStatus = status;
  state.recordingPollError = null;
  renderRecordingControl();
  renderCollectionOverview();
  renderStreamControl();

  if (previousState === status.state) return;
  if (status.state === "countdown") {
    addEvent("INFO", "录制倒计时已开始", status.detail || "3 秒后开始保存视频");
  } else if (status.state === "recording") {
    addEvent("OK", "视频录制已开始", status.detail || "正在写入当前会话");
  } else if (status.state === "finalizing") {
    addEvent("INFO", "正在保存视频", status.detail || "正在封装 MP4 文件");
  } else if (status.state === "ready" && previousState === "countdown") {
    addEvent("INFO", "录制已取消", status.detail || "未生成视频文件");
  } else if (
    status.state === "ready" &&
    ["recording", "finalizing"].includes(previousState)
  ) {
    addEvent("OK", "视频已保存", status.detail || "视频已加入本地会话");
  } else if (status.state === "error") {
    addEvent("ERROR", "录制服务异常", status.detail || "无法继续录制");
  }
}

function updateRecordingCountdown() {
  window.clearTimeout(state.recordingCountdownTimer);
  const status = state.recordingStatus;
  if (status?.state !== "countdown") {
    elements.recordingCountdown.hidden = true;
    return;
  }
  const remainingMs = status.recording_starts_at_unix_ms - Date.now();
  elements.recordingCountdownValue.textContent = String(
    Math.min(3, Math.max(1, Math.ceil(remainingMs / 1000))),
  );
  elements.recordingCountdown.hidden = false;
  state.recordingCountdownTimer = window.setTimeout(updateRecordingCountdown, 50);
}

function renderRecordingControl() {
  const status = state.recordingStatus;
  const recordingState = status?.state || "unavailable";
  const canStart = recordingState === "ready";
  const canStop = ["countdown", "recording"].includes(recordingState);
  const busy = state.recordingCommandInFlight || recordingState === "finalizing";
  elements.recordingToggleButton.disabled = busy || (!canStart && !canStop);
  elements.recordingToggleButton.dataset.action = canStop ? "stop" : "start";
  elements.recordingToggleButton.classList.toggle("is-recording", recordingState === "recording");
  elements.recordingToggleButton.classList.toggle("is-countdown", recordingState === "countdown");

  if (state.recordingCommandInFlight) {
    elements.recordingToggleLabel.textContent = "命令发送中";
  } else if (recordingState === "countdown") {
    elements.recordingToggleLabel.textContent = "取消录制";
  } else if (recordingState === "recording") {
    elements.recordingToggleLabel.textContent = "停止录制";
  } else if (recordingState === "finalizing") {
    elements.recordingToggleLabel.textContent = "正在保存";
  } else {
    elements.recordingToggleLabel.textContent = "开始录制";
  }
  elements.recordingToggleIcon.textContent = canStop ? "■" : "●";
  elements.recordingToggleButton.title = canStop ? "停止当前录制" : "3 秒后开始录制";
  elements.recordingSummary.textContent = status?.detail || recordingStateLabels[recordingState];
  updateRecordingCountdown();
}

function findCurrentSession() {
  const sessionId = state.recordingStatus?.session_id;
  if (!sessionId || state.collectionLibrary === null) return null;
  return state.collectionLibrary.sessions.find((session) => session.session_id === sessionId) || null;
}

function formatMetadataCoverage(quality) {
  if (quality.metadata_match_coverage === null) return "等待视频帧";
  return `${(quality.metadata_match_coverage * 100).toFixed(1)}%`;
}

function renderCollectionOverview() {
  const status = state.recordingStatus;
  const session = findCurrentSession();
  const sessionState = status?.session_state || null;
  const recordingBusy = ["countdown", "recording", "finalizing"].includes(status?.state);
  const commandBusy = state.collectionCommandInFlight;
  const gatewayStatusKnown = status !== null && status.state !== "error";

  elements.newSessionButton.disabled =
    !gatewayStatusKnown || sessionState !== "active" || recordingBusy || commandBusy;
  elements.newSessionButton.title = recordingBusy
    ? "停止当前录制后才能开始新会话"
    : sessionState === "active"
    ? "结束当前会话，下一次录制自动开始新会话"
    : "当前没有可结束的采集会话";
  elements.sessionStatusDot.dataset.state = sessionState || "idle";

  if (state.collectionCommandInFlight) {
    elements.currentSessionName.textContent = "正在切换会话";
  } else if (state.collectionPollError) {
    elements.currentSessionName.textContent = "无法读取会话";
  } else if (session) {
    elements.currentSessionName.textContent = getSessionDisplayName(session);
  } else if (status?.session_id) {
    elements.currentSessionName.textContent = "正在读取会话";
  } else {
    elements.currentSessionName.textContent = "首次录制时自动创建";
  }

  const sessionStateLabels = {
    active: "采集中",
    finalizing: "正在完成",
    complete: "已完成",
    incomplete: "异常中断",
  };
  elements.currentSessionState.textContent = sessionStateLabels[sessionState] || "尚未开始";

  if (session === null) {
    elements.currentSessionImu.textContent = sessionState === "active" ? "正在读取" : "尚未保存";
    elements.currentSessionMetadata.textContent = "--";
    elements.currentSessionSync.textContent = "--";
    return;
  }

  const quality = session.quality;
  const hasTelemetry = session.telemetry_database === "telemetry/telemetry.sqlite";
  elements.currentSessionImu.textContent = hasTelemetry
    ? session.state === "active"
      ? `持续保存 · ${quality.imu_sample_count.toLocaleString("zh-CN")}`
      : `${quality.imu_sample_count.toLocaleString("zh-CN")} 样本`
    : "未保存";
  elements.currentSessionImu.title = hasTelemetry
    ? `加速度 ${quality.accelerometer_sample_count} · 陀螺仪 ${quality.gyroscope_sample_count} · 序号缺口 ${quality.imu_sequence_gap_count}`
    : "该会话没有 IMU 遥测数据库";
  elements.currentSessionMetadata.textContent = formatMetadataCoverage(quality);
  elements.currentSessionMetadata.title =
    `${quality.recorded_video_frame_metadata_match_count} / ${quality.recorded_video_frame_count} 个录制帧已匹配`;
  elements.currentSessionSync.textContent = quality.timestamp_alignment_state === "unverified"
    ? "原始时间 · 未验证"
    : quality.timestamp_alignment_state;
  elements.currentSessionSync.title =
    `${quality.timestamp_mapping_segment_count} 个原始时间映射分段`;
}

async function pollCollectionLibrary() {
  if (state.collectionPollInFlight || document.hidden) {
    scheduleCollectionPoll();
    return;
  }
  state.collectionPollInFlight = true;
  try {
    const response = await fetch(recordingLibraryEndpoint, { cache: "no-store" });
    const payload = await readJsonResponse(response, `采集会话 HTTP ${response.status}`);
    state.collectionLibrary = readRecordingLibrary(payload);
    state.collectionPollError = null;
  } catch (error) {
    state.collectionLibrary = null;
    state.collectionPollError = error.message;
  } finally {
    state.collectionPollInFlight = false;
    renderCollectionOverview();
    scheduleCollectionPoll();
  }
}

async function requestNewCollectionSession() {
  if (state.collectionCommandInFlight || elements.newSessionButton.disabled) return;
  state.collectionCommandInFlight = true;
  state.collectionCommandError = null;
  renderCollectionOverview();
  renderStreamControl();
  try {
    const response = await fetch(recordingSessionCommandEndpoint, {
      method: "POST",
      cache: "no-store",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "new" }),
    });
    const payload = await readJsonResponse(response, `新会话 HTTP ${response.status}`);
    applyRecordingStatus(payload);
    addEvent("OK", "当前会话已结束", "下一次录制会自动开始新会话并保存 IMU");
    await pollCollectionLibrary();
  } catch (error) {
    state.collectionCommandError = error.message;
    addEvent("ERROR", "会话切换失败", error.message);
  } finally {
    state.collectionCommandInFlight = false;
    renderCollectionOverview();
    renderStreamControl();
    scheduleRecordingPoll(0);
  }
}

async function pollRecordingStatus() {
  if (state.recordingPollInFlight || state.recordingCommandInFlight || document.hidden) {
    scheduleRecordingPoll();
    return;
  }
  state.recordingPollInFlight = true;
  try {
    const response = await fetch(recordingStatusEndpoint, { cache: "no-store" });
    const payload = await readJsonResponse(response, `录制状态 HTTP ${response.status}`);
    applyRecordingStatus(payload);
  } catch (error) {
    if (state.recordingPollError !== error.message) {
      addEvent("WARN", "录制服务不可用", error.message);
    }
    state.recordingStatus = null;
    state.recordingPollError = error.message;
    renderRecordingControl();
    renderCollectionOverview();
    renderStreamControl();
  } finally {
    state.recordingPollInFlight = false;
    scheduleRecordingPoll(state.recordingStatus?.state === "countdown" ? 200 : 500);
  }
}

function addEvent(level, event, detail) {
  state.events.unshift({
    time: new Date().toLocaleTimeString("zh-CN", { hour12: false }),
    level,
    event,
    detail,
  });
  state.events = state.events.slice(0, maxEventHistory);
  renderEvents();
}

function renderEvents() {
  elements.eventRows.replaceChildren();
  state.events.forEach((entry) => {
    const row = document.createElement("tr");
    const time = document.createElement("td");
    const level = document.createElement("td");
    const message = document.createElement("td");
    const eventTitle = document.createElement("strong");
    const eventDetail = document.createElement("span");
    time.textContent = entry.time;
    level.textContent = entry.level;
    level.className = `event-level ${entry.level.toLowerCase()}`;
    message.className = "event-message";
    eventTitle.textContent = entry.event;
    eventDetail.textContent = entry.detail;
    message.append(eventTitle, eventDetail);
    row.append(time, level, message);
    elements.eventRows.append(row);
  });
  elements.eventEmpty.hidden = state.events.length > 0;
  elements.eventCount.textContent = `${state.events.length} 条`;
}

async function sendRecordingCommand(action) {
  const recordingState = state.recordingStatus?.state;
  if (
    state.recordingCommandInFlight ||
    (action === "start" && recordingState !== "ready") ||
    (action === "stop" && !["countdown", "recording"].includes(recordingState))
  ) {
    return;
  }
  state.recordingCommandInFlight = true;
  state.recordingCommandError = null;
  renderRecordingControl();
  try {
    const response = await fetch(recordingCommandEndpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
    const payload = await readJsonResponse(response, `录制控制 HTTP ${response.status}`);
    applyRecordingStatus(payload);
  } catch (error) {
    state.recordingCommandError = error.message;
    addEvent(
      "ERROR",
      action === "start" ? "开始录制失败" : "停止录制失败",
      error.message,
    );
    console.warn("Recording command failed", error);
  } finally {
    state.recordingCommandInFlight = false;
    renderRecordingControl();
    renderStreamControl();
    scheduleRecordingPoll(0);
  }
}

function waitForIceGathering(peer) {
  if (peer.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      peer.removeEventListener("icegatheringstatechange", handleStateChange);
      reject(new Error("本机 WebRTC ICE 收集超时"));
    }, 5000);

    function handleStateChange() {
      if (peer.iceGatheringState !== "complete") return;
      window.clearTimeout(timeout);
      peer.removeEventListener("icegatheringstatechange", handleStateChange);
      resolve();
    }

    peer.addEventListener("icegatheringstatechange", handleStateChange);
  });
}

async function exchangeViewerSdp(description) {
  const response = await fetch(viewerSignalingEndpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      schema_version: "1.0",
      type: description.type,
      sdp: description.sdp,
    }),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new ViewerSignalingError(
      response.status,
      payload.detail || `Viewer signaling HTTP ${response.status}`,
    );
  }
  return response.json();
}

async function connectLiveVideo() {
  if (state.connecting || state.peer || document.hidden) return;
  state.connecting = true;
  window.clearTimeout(state.reconnectTimer);

  const peer = new RTCPeerConnection({
    iceServers: [],
    bundlePolicy: "max-bundle",
  });
  state.peer = peer;
  peer.addTransceiver("video", { direction: "recvonly" });

  peer.addEventListener("track", (event) => {
    if (state.peer !== peer || event.track.kind !== "video") return;
    const stream = event.streams[0] || new MediaStream([event.track]);
    elements.liveVideo.srcObject = stream;
    event.track.addEventListener("ended", () => handleViewerDisconnect(peer, "视频轨道已结束"));
    elements.liveVideo.play().catch((error) => {
      handleViewerDisconnect(peer, `视频播放失败: ${error.message}`);
    });
    startFrameMonitoring();
  });

  peer.addEventListener("connectionstatechange", () => {
    if (["failed", "disconnected", "closed"].includes(peer.connectionState)) {
      handleViewerDisconnect(peer, `WebRTC ${peer.connectionState}`);
    }
  });

  try {
    const offer = await peer.createOffer();
    await peer.setLocalDescription(offer);
    await waitForIceGathering(peer);
    if (peer.localDescription === null) throw new Error("本机 WebRTC offer 不可用");
    const answer = await exchangeViewerSdp(peer.localDescription);
    await peer.setRemoteDescription({ type: answer.type, sdp: answer.sdp });
    state.lastSignalingError = null;
  } catch (error) {
    const status = error instanceof ViewerSignalingError ? error.status : null;
    if (status !== 503 && state.lastSignalingError !== error.message) {
      addEvent("WARN", "本机预览连接失败", error.message);
      console.warn("Local preview connection failed", error);
      state.lastSignalingError = error.message;
    }
    closeViewerPeer(peer, error.message);
    scheduleViewerRetry(status === 503 ? 500 : 1500);
  } finally {
    state.connecting = false;
  }
}

function handleViewerDisconnect(peer, detail) {
  if (state.peer !== peer) return;
  closeViewerPeer(peer, detail);
  scheduleViewerRetry();
}

function closeViewerPeer(peer, detail = "等待重新连接") {
  if (state.peer !== peer) return;
  state.peer = null;
  peer.onconnectionstatechange = null;
  peer.close();
  stopFrameMonitoring();
  elements.liveVideo.srcObject = null;
  const wasReady = state.liveVideoReady;
  state.liveVideoReady = false;
  renderVideoState();
  if (wasReady) addEvent("WARN", "Glass3 视频已断开", detail);
}

function startFrameMonitoring() {
  stopFrameMonitoring();
  if ("requestVideoFrameCallback" in elements.liveVideo) {
    state.frameCallbackId = elements.liveVideo.requestVideoFrameCallback(handleVideoFrame);
    return;
  }
  state.fallbackFrameTimer = window.setInterval(readFallbackFrameStats, 1000);
}

function stopFrameMonitoring() {
  if (state.frameCallbackId !== null && "cancelVideoFrameCallback" in elements.liveVideo) {
    elements.liveVideo.cancelVideoFrameCallback(state.frameCallbackId);
  }
  window.clearInterval(state.fallbackFrameTimer);
  state.frameCallbackId = null;
  state.fallbackFrameTimer = null;
  state.lastPresentedFrames = null;
  state.lastFpsSampleAt = null;
  state.lastDetailsAt = 0;
  elements.previewFps.textContent = "--";
}

function handleVideoFrame(now, metadata) {
  markVideoReady();
  updateFrameDetails(now);
  updateDisplayedFps(now, metadata.presentedFrames);
  state.frameCallbackId = elements.liveVideo.requestVideoFrameCallback(handleVideoFrame);
}

function readFallbackFrameStats() {
  const quality = elements.liveVideo.getVideoPlaybackQuality?.();
  markVideoReady();
  updateFrameDetails(performance.now());
  if (quality) updateDisplayedFps(performance.now(), quality.totalVideoFrames);
}

function updateDisplayedFps(now, presentedFrames) {
  if (state.lastPresentedFrames === null || state.lastFpsSampleAt === null) {
    state.lastPresentedFrames = presentedFrames;
    state.lastFpsSampleAt = now;
    return;
  }
  const elapsedMs = now - state.lastFpsSampleAt;
  if (elapsedMs < 1000) return;
  const fps = ((presentedFrames - state.lastPresentedFrames) * 1000) / elapsedMs;
  elements.previewFps.textContent = `${fps.toFixed(1)} FPS`;
  state.lastPresentedFrames = presentedFrames;
  state.lastFpsSampleAt = now;
}

function updateFrameDetails(now) {
  if (now - state.lastDetailsAt < 500) return;
  state.lastDetailsAt = now;
  const frameSize = `${elements.liveVideo.videoWidth} × ${elements.liveVideo.videoHeight}`;
  elements.resolutionBadge.textContent = `${frameSize} · LIVE`;
  elements.frameSize.textContent = frameSize;
  elements.lastFrameTime.textContent = new Date().toLocaleTimeString("zh-CN", {
    hour12: false,
  });
}

function markVideoReady() {
  if (state.liveVideoReady) return;
  state.liveVideoReady = true;
  renderVideoState();
  addEvent("OK", "Glass3 视频已连接", "WebRTC H.264 实时轨道");
}

function renderVideoState() {
  const ready = state.liveVideoReady;
  elements.connectionLight.classList.toggle("is-live", ready);
  elements.connectionLabel.textContent = ready ? "Glass3 视频在线" : "等待 Glass3 视频";
  elements.liveBadge.classList.toggle("badge-live", ready);
  elements.liveBadgeLabel.textContent = ready ? "LIVE" : "WAITING";
  elements.sourcePill.classList.toggle("pill-success", ready);
  elements.sourcePill.textContent = ready ? "实时在线" : "等待首帧";
  elements.previewStatus.textContent = ready ? "实时画面已连接" : "等待首帧";
  elements.liveVideo.classList.toggle("is-ready", ready);
  elements.viewerEmpty.hidden = ready;

  if (!ready) {
    elements.resolutionBadge.textContent = "等待画面";
    elements.frameSize.textContent = "--";
    elements.lastFrameTime.textContent = "--";
  }
}

async function toggleFullscreen() {
  try {
    if (document.fullscreenElement) {
      await document.exitFullscreen();
    } else {
      await elements.viewerStage.requestFullscreen();
    }
  } catch (error) {
    addEvent("WARN", "全屏切换失败", error.message);
    console.warn("Fullscreen toggle failed", error);
  }
}

function scheduleImuPoll(delayMs = state.imuConnected ? 50 : 500) {
  window.clearTimeout(state.imuPollTimer);
  if (document.hidden) return;
  state.imuPollTimer = window.setTimeout(pollImuStatus, delayMs);
}

function readImuSample(sample, expectedType, expectedAndroidType) {
  if (
    sample === null ||
    typeof sample !== "object" ||
    sample.schema_version !== "0.1" ||
    sample.message_type !== "imu_sample" ||
    sample.sensor_type !== expectedType ||
    sample.android_sensor_type !== expectedAndroidType ||
    !Number.isSafeInteger(sample.sequence_number) ||
    !Number.isSafeInteger(sample.sensor_event_monotonic_ns) ||
    !Array.isArray(sample.values) ||
    sample.values.length !== 3 ||
    !sample.values.every(Number.isFinite)
  ) {
    throw new Error(`接收网关返回了无效的 ${expectedType} 样本`);
  }
  return sample;
}

function readImuStatus(payload) {
  const channelStates = new Set(["unavailable", "ready", "receiving"]);
  if (
    payload === null ||
    typeof payload !== "object" ||
    payload.schema_version !== "0.1" ||
    !channelStates.has(payload.channel_state) ||
    payload.sensors === null ||
    typeof payload.sensors !== "object"
  ) {
    throw new Error("接收网关返回了无效的 IMU 状态");
  }
  if (payload.channel_state !== "receiving") {
    return { ...payload, accelerometer: null, gyroscope: null };
  }
  const accelerometerStatus = payload.sensors.accelerometer;
  const gyroscopeStatus = payload.sensors.gyroscope;
  const accelerometer = readImuSample(
    accelerometerStatus?.last_sample,
    "accelerometer",
    1,
  );
  const gyroscope = readImuSample(gyroscopeStatus?.last_sample, "gyroscope", 4);
  return {
    ...payload,
    accelerometer,
    gyroscope,
    accelerometerRate: accelerometerStatus.observed_rate_hz,
    gyroscopeRate: gyroscopeStatus.observed_rate_hz,
  };
}

function vectorMagnitude(values) {
  return Math.hypot(...values);
}

function formatVector(values) {
  return values.map((value) => value.toFixed(3)).join(", ");
}

function renderImuOrientation(orientation) {
  state.imuOrientationReady = orientation.ready;
  if (!orientation.ready) {
    elements.imuRoll.textContent = "--°";
    elements.imuPitch.textContent = "--°";
    elements.imuYaw.textContent = "--°";
    elements.imuEmpty.hidden = false;
    elements.imuEmptyDetail.textContent = "正在建立相对姿态参考";
    return;
  }
  elements.imuRoll.textContent = `${orientation.roll.toFixed(1)}°`;
  elements.imuPitch.textContent = `${orientation.pitch.toFixed(1)}°`;
  elements.imuYaw.textContent = `${orientation.yaw.toFixed(1)}°`;
  elements.imuEmpty.hidden = true;
}

function setImuUnavailable(detail) {
  const wasConnected = state.imuConnected;
  state.imuConnected = false;
  state.imuOrientationReady = false;
  elements.imuStatusPill.classList.remove("pill-success");
  elements.imuStatusPill.textContent = "等待 IMU";
  elements.imuEmpty.hidden = false;
  elements.imuEmptyDetail.textContent = detail;
  elements.imuAcceleration.textContent = "-- m/s²";
  elements.imuAngularRate.textContent = "-- rad/s";
  elements.imuAccelerationRate.textContent = "-- Hz";
  elements.imuGyroscopeRate.textContent = "-- Hz";
  elements.resetImuButton.disabled = true;
  if (imuScene !== null) imuScene.setActive(false);
  if (wasConnected) addEvent("WARN", "Glass3 IMU 已断开", detail);
}

function applyImuStatus(payload) {
  const status = readImuStatus(payload);
  if (status.channel_state !== "receiving") {
    setImuUnavailable(
      status.channel_state === "ready" ? "IMU 通道已连接，等待首个样本" : "接收网关尚未收到姿态数据",
    );
    return;
  }

  const wasConnected = state.imuConnected;
  state.imuConnected = true;
  const accelerationMagnitude = vectorMagnitude(status.accelerometer.values);
  const angularRateMagnitude = vectorMagnitude(status.gyroscope.values);
  const accelerometerRate = Number.isFinite(status.accelerometerRate)
    ? status.accelerometerRate
    : null;
  const gyroscopeRate = Number.isFinite(status.gyroscopeRate) ? status.gyroscopeRate : null;
  const displayRate = Math.min(
    accelerometerRate ?? Number.POSITIVE_INFINITY,
    gyroscopeRate ?? Number.POSITIVE_INFINITY,
  );
  if (!wasConnected) {
    const detail = Number.isFinite(displayRate)
      ? `实时姿态样本 · ${displayRate.toFixed(0)} Hz`
      : "实时姿态样本已到达";
    addEvent("OK", "Glass3 IMU 已连接", detail);
  }
  elements.imuStatusPill.classList.add("pill-success");
  elements.imuStatusPill.textContent = Number.isFinite(displayRate)
    ? `LIVE ${displayRate.toFixed(0)} HZ`
    : "IMU LIVE";
  elements.imuAcceleration.textContent = `${accelerationMagnitude.toFixed(2)} m/s²`;
  elements.imuAcceleration.title = `x, y, z: ${formatVector(status.accelerometer.values)}`;
  elements.imuAngularRate.textContent = `${angularRateMagnitude.toFixed(3)} rad/s`;
  elements.imuAngularRate.title = `x, y, z: ${formatVector(status.gyroscope.values)}`;
  elements.imuAccelerationRate.textContent = accelerometerRate === null
    ? "-- Hz"
    : `${accelerometerRate.toFixed(1)} Hz`;
  elements.imuGyroscopeRate.textContent = gyroscopeRate === null
    ? "-- Hz"
    : `${gyroscopeRate.toFixed(1)} Hz`;
  elements.resetImuButton.disabled = imuScene === null;

  if (imuScene === null) {
    elements.imuEmpty.hidden = false;
    elements.imuEmptyDetail.textContent = state.imuSceneError || "3D 姿态视图不可用";
    return;
  }
  imuScene.beginSession(status.session_id || status.device_session_id || "active");
  imuScene.setActive(true);
  imuScene.update(status.accelerometer, status.gyroscope);
}

async function pollImuStatus() {
  if (state.imuPollInFlight || document.hidden) {
    scheduleImuPoll();
    return;
  }
  state.imuPollInFlight = true;
  try {
    const response = await fetch(imuStatusEndpoint, { cache: "no-store" });
    const payload = await readJsonResponse(response, `IMU 状态 HTTP ${response.status}`);
    applyImuStatus(payload);
  } catch (error) {
    setImuUnavailable(error.message);
  } finally {
    state.imuPollInFlight = false;
    scheduleImuPoll();
  }
}

try {
  imuScene = new ImuSceneController(elements.imuCanvas, renderImuOrientation);
} catch (error) {
  state.imuSceneError = error.message;
  addEvent("ERROR", "IMU 三维视图初始化失败", error.message);
  console.warn("IMU scene initialization failed", error);
}

function resetImuReference() {
  if (imuScene === null) return;
  imuScene.resetReference();
  addEvent("INFO", "IMU 姿态已归零", "当前眼镜方向已设为相对姿态原点");
}

elements.fullscreenButton.addEventListener("click", toggleFullscreen);
elements.streamToggleButton.addEventListener("click", () => {
  sendStreamControlCommand(elements.streamToggleButton.dataset.action);
});
elements.recordingToggleButton.addEventListener("click", () => {
  sendRecordingCommand(elements.recordingToggleButton.dataset.action);
});
elements.newSessionButton.addEventListener("click", requestNewCollectionSession);
elements.resetImuButton.addEventListener("click", resetImuReference);
elements.clearEventsButton.addEventListener("click", () => {
  state.events = [];
  renderEvents();
});
document.addEventListener("fullscreenchange", () => {
  elements.fullscreenButton.textContent = document.fullscreenElement ? "退出全屏" : "全屏";
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    if (state.peer === null) scheduleViewerRetry(0);
    scheduleControlPoll(0);
    scheduleRecordingPoll(0);
    scheduleCollectionPoll(0);
    scheduleImuPoll(0);
  }
});
window.addEventListener("beforeunload", () => {
  window.clearTimeout(state.reconnectTimer);
  window.clearTimeout(state.controlPollTimer);
  window.clearTimeout(state.recordingPollTimer);
  window.clearTimeout(state.recordingCountdownTimer);
  window.clearTimeout(state.collectionPollTimer);
  window.clearTimeout(state.imuPollTimer);
  if (state.peer !== null) closeViewerPeer(state.peer);
  imuScene?.dispose();
});

addEvent("INFO", "客户端已启动", "等待 Glass3 视频、控制通路和 IMU 数据");
renderVideoState();
renderStreamControl();
renderRecordingControl();
renderCollectionOverview();
setImuUnavailable("接收网关尚未收到姿态数据");
connectLiveVideo();
pollStreamControlStatus();
pollRecordingStatus();
pollCollectionLibrary();
pollImuStatus();
