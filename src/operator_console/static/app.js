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
  decodedPreview: document.querySelector("#decoded-preview-source"),
  handTrackingOverlay: document.querySelector("#hand-tracking-overlay"),
  viewerStage: document.querySelector("#viewer-stage"),
  viewerEmpty: document.querySelector("#viewer-empty"),
  viewerEmptyTitle: document.querySelector("#viewer-empty-title"),
  viewerEmptyDetail: document.querySelector("#viewer-empty-detail"),
  viewerModeLive: document.querySelector("#viewer-mode-live"),
  viewerModeReplay: document.querySelector("#viewer-mode-replay"),
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
  handTrackingState: document.querySelector("#hand-tracking-state"),
  handTrackingDetail: document.querySelector("#hand-tracking-detail"),
  decodedFrameCount: document.querySelector("#decoded-frame-count"),
  handReceivedCount: document.querySelector("#hand-received-count"),
  handInputFrame: document.querySelector("#hand-input-frame"),
  handInferenceTime: document.querySelector("#hand-inference-time"),
  handInferenceCount: document.querySelector("#hand-inference-count"),
  handDroppedCount: document.querySelector("#hand-dropped-count"),
  leftHandReadout: document.querySelector("#left-hand-readout"),
  rightHandReadout: document.querySelector("#right-hand-readout"),
  replaySession: document.querySelector("#replay-session"),
  startReplayButton: document.querySelector("#start-replay-button"),
  replayProgress: document.querySelector("#replay-progress"),
  handReplayVideo: document.querySelector("#hand-replay-video"),
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
  viewerMode: "live",
  liveVideoReady: false,
  decodedPreviewStatus: null,
  decodedPreviewError: null,
  decodedPreviewPollTimer: null,
  decodedPreviewPollInFlight: false,
  decodedPreviewReconnectTimer: null,
  decodedPreviewGeneration: 0,
  lastDecodedFrameIndex: null,
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
  handTrackingEventSource: null,
  handTrackingStatus: null,
  handTrackingError: null,
  replayRequestInFlight: false,
  imuPollTimer: null,
  imuPollInFlight: false,
  imuConnected: false,
  imuOrientationReady: false,
  imuSceneError: null,
};

const decodedPreviewEndpoint = "http://127.0.0.1:8770/api/v1/webrtc/decoded-preview.mjpg";
const decodedPreviewStatusEndpoint =
  "http://127.0.0.1:8770/api/v1/webrtc/decoded-preview/status";
const streamControlEndpoint = "http://127.0.0.1:8770/api/v1/webrtc/control";
const streamControlCommandEndpoint = `${streamControlEndpoint}/commands`;
const imuStatusEndpoint = "http://127.0.0.1:8770/api/v1/webrtc/imu/status";
const handTrackingEndpoint = "http://127.0.0.1:8770/api/v1/perception/hand-tracking";
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
let imuScene = null;

if (window.location.search) {
  window.history.replaceState({}, "", window.location.pathname);
}

function scheduleDecodedPreviewReconnect(delayMs = 1000) {
  window.clearTimeout(state.decodedPreviewReconnectTimer);
  if (document.hidden) return;
  state.decodedPreviewReconnectTimer = window.setTimeout(connectDecodedPreview, delayMs);
}

function scheduleDecodedPreviewPoll(delayMs = 500) {
  window.clearTimeout(state.decodedPreviewPollTimer);
  if (document.hidden) return;
  state.decodedPreviewPollTimer = window.setTimeout(pollDecodedPreviewStatus, delayMs);
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

function closeHandTrackingEvents() {
  const eventSource = state.handTrackingEventSource;
  state.handTrackingEventSource = null;
  eventSource?.close();
}

function connectHandTrackingEvents() {
  if (document.hidden || state.handTrackingEventSource !== null) return;
  const eventSource = new EventSource(`${handTrackingEndpoint}/events`);
  state.handTrackingEventSource = eventSource;
  eventSource.addEventListener("status", (event) => {
    if (state.handTrackingEventSource !== eventSource) return;
    try {
      state.handTrackingStatus = readHandTrackingStatus(JSON.parse(event.data));
      state.handTrackingError = null;
    } catch (error) {
      state.handTrackingError = error.message;
    }
    renderHandTrackingStatus();
  });
  eventSource.onopen = () => {
    if (state.handTrackingEventSource !== eventSource) return;
    state.handTrackingError = null;
    renderHandTrackingStatus();
  };
  eventSource.onerror = () => {
    if (state.handTrackingEventSource !== eventSource || document.hidden) return;
    state.handTrackingError = "手部感知推送连接中断，正在重连";
    renderHandTrackingStatus();
  };
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
    ? "源时间已保留"
    : quality.timestamp_alignment_state;
  elements.currentSessionSync.title =
    "跨模态对齐将在感知阶段执行";
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
    renderReplaySessionOptions();
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
  const writer = ["WARN", "ERROR"].includes(level) ? console.warn : console.info;
  writer(`[${level}] ${event}: ${detail}`);
}

function readHandTrackingStatus(payload) {
  if (
    payload === null ||
    typeof payload !== "object" ||
    payload.schema_version !== "1.0" ||
    typeof payload.state !== "string" ||
    payload.replay === null ||
    typeof payload.replay !== "object"
  ) {
    throw new Error("接收网关返回了无效的手部感知状态");
  }
  return payload;
}

function renderHandReadout(element, hand) {
  const title = element.querySelector("strong");
  const detail = element.querySelector(".hand-readout-meta");
  const confidenceFields = element.querySelectorAll("[data-confidence-field]");
  if (!hand) {
    title.textContent = "未检测";
    detail.textContent = "--";
    confidenceFields.forEach((field) => {
      field.textContent = "--";
    });
    return;
  }
  const finalConfidence = hand.final_confidence ?? hand.confidence;
  const confidenceValues = {
    detector_confidence: hand.detector_confidence,
    reconstruction_quality: hand.reconstruction_quality,
    depth_score: hand.depth_score,
    coverage_score: hand.coverage_score,
    compactness_score: hand.compactness_score,
    final_confidence: finalConfidence,
  };
  title.textContent = `${formatConfidence(finalConfidence)} · ${hand.reconstruction_backend}`;
  detail.textContent = `${hand.metric_depth_status} · ${hand.is_grasping ? "抓握" : "展开"}`;
  confidenceFields.forEach((field) => {
    field.textContent = formatConfidence(confidenceValues[field.dataset.confidenceField]);
  });
}

function formatConfidence(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "--";
}

const handBones = [
  [5, 6], [6, 7], [7, 0],
  [5, 8], [8, 9], [9, 10], [10, 1],
  [5, 11], [11, 12], [12, 13], [13, 2],
  [5, 14], [14, 15], [15, 16], [16, 3],
  [5, 17], [17, 18], [18, 19], [19, 4],
];

function clearHandTrackingOverlay() {
  const canvas = elements.handTrackingOverlay;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
}

function drawHandTrackingOverlay(result) {
  const canvas = elements.handTrackingOverlay;
  const bounds = elements.viewerStage.getBoundingClientRect();
  const sourceWidth = result?.source_image_width_px;
  const sourceHeight = result?.source_image_height_px;
  const hands = Array.isArray(result?.hands) ? result.hands : [];
  if (
    state.viewerMode !== "live"
    || bounds.width <= 0
    || bounds.height <= 0
    || !Number.isFinite(sourceWidth)
    || !Number.isFinite(sourceHeight)
  ) {
    clearHandTrackingOverlay();
    return;
  }

  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const canvasWidth = Math.max(1, Math.round(bounds.width * pixelRatio));
  const canvasHeight = Math.max(1, Math.round(bounds.height * pixelRatio));
  if (canvas.width !== canvasWidth || canvas.height !== canvasHeight) {
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
  }
  const context = canvas.getContext("2d");
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, bounds.width, bounds.height);

  const scale = Math.max(bounds.width / sourceWidth, bounds.height / sourceHeight);
  const offsetX = (bounds.width - sourceWidth * scale) / 2;
  const offsetY = (bounds.height - sourceHeight * scale) / 2;
  const mapPoint = (point) => [offsetX + point[0] * scale, offsetY + point[1] * scale];

  hands.forEach((hand) => {
    if (
      !Array.isArray(hand.source_keypoints_2d_px)
      || !Array.isArray(hand.source_bbox_xyxy_px)
    ) return;
    const color = hand.handedness === "left" ? "#f3c878" : "#64d98c";
    const points = hand.source_keypoints_2d_px.map(mapPoint);
    context.strokeStyle = color;
    context.fillStyle = color;
    context.lineWidth = 2;
    context.lineJoin = "round";
    context.lineCap = "round";
    handBones.forEach(([first, second]) => {
      if (!points[first] || !points[second]) return;
      context.beginPath();
      context.moveTo(...points[first]);
      context.lineTo(...points[second]);
      context.stroke();
    });
    points.slice(0, 20).forEach((point) => {
      context.beginPath();
      context.arc(point[0], point[1], 3, 0, Math.PI * 2);
      context.fill();
    });

    const [x1, y1] = mapPoint(hand.source_bbox_xyxy_px.slice(0, 2));
    const [x2, y2] = mapPoint(hand.source_bbox_xyxy_px.slice(2, 4));
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const finalConfidence = hand.final_confidence ?? hand.confidence;
    const label = `${hand.handedness.toUpperCase()} ${Math.round(finalConfidence * 100)}%${hand.is_grasping ? " · GRASP" : ""}`;
    context.font = "700 11px Consolas, monospace";
    const labelWidth = context.measureText(label).width + 12;
    const labelTop = Math.max(4, y1 - 23);
    context.fillStyle = "rgba(8, 10, 9, 0.88)";
    context.fillRect(x1, labelTop, labelWidth, 19);
    context.fillStyle = color;
    context.fillText(label, x1 + 6, labelTop + 13);
  });
}

function renderReplaySessionOptions() {
  const selected = elements.replaySession.value;
  const sessions = (state.collectionLibrary?.sessions || []).filter(
    (session) => session.state !== "active" && session.clips.length > 0,
  );
  elements.replaySession.replaceChildren();
  if (sessions.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "暂无可回放会话";
    elements.replaySession.append(option);
  } else {
    sessions.forEach((session) => {
      const option = document.createElement("option");
      option.value = session.session_id;
      option.textContent = getSessionDisplayName(session);
      elements.replaySession.append(option);
    });
    if (sessions.some((session) => session.session_id === selected)) {
      elements.replaySession.value = selected;
    }
  }
  elements.startReplayButton.disabled = sessions.length === 0 || state.replayRequestInFlight;
}

function replayMatchesSelectedSession() {
  return Boolean(
    elements.handReplayVideo.dataset.source
      && elements.handReplayVideo.dataset.sessionId === elements.replaySession.value,
  );
}

function renderViewerMedia() {
  const liveMode = state.viewerMode === "live";
  const showLiveVideo = liveMode && state.liveVideoReady;
  const showReplay = !liveMode && replayMatchesSelectedSession();

  elements.viewerStage.dataset.viewerMode = state.viewerMode;
  elements.viewerModeLive.classList.toggle("is-active", liveMode);
  elements.viewerModeLive.setAttribute("aria-pressed", String(liveMode));
  elements.viewerModeReplay.classList.toggle("is-active", !liveMode);
  elements.viewerModeReplay.setAttribute("aria-pressed", String(!liveMode));
  elements.decodedPreview.hidden = !liveMode;
  elements.handTrackingOverlay.hidden = !liveMode;
  elements.handReplayVideo.hidden = !showReplay;
  elements.viewerEmpty.hidden = showLiveVideo || showReplay;
  elements.liveBadge.classList.toggle("badge-live", liveMode && showLiveVideo);

  if (liveMode) {
    const hands = state.handTrackingStatus?.latest_result?.hands || [];
    elements.liveBadgeLabel.textContent = showLiveVideo ? "DECODED LIVE" : "WAITING";
    const status = state.decodedPreviewStatus;
    const frameSize = status?.width && status?.height ? `${status.width} × ${status.height}` : null;
    elements.resolutionBadge.textContent = frameSize
      ? `${frameSize} · ${hands.length > 0 ? "TRACKED" : "DECODED"}`
      : "等待画面";
    elements.viewerEmptyTitle.textContent = "等待 Glass3 首帧";
    elements.viewerEmptyDetail.textContent = state.decodedPreviewError
      || "接收网关完成首帧解码后，画面会自动出现";
    return;
  }

  clearHandTrackingOverlay();

  const replay = state.handTrackingStatus?.replay;
  const hasReplaySessions = elements.replaySession.options.length > 0
    && Boolean(elements.replaySession.value);
  elements.liveBadgeLabel.textContent = "REPLAY";
  elements.resolutionBadge.textContent = showReplay ? "离线识别结果" : "暂无回放";
  elements.viewerEmptyTitle.textContent = hasReplaySessions ? "等待回放结果" : "暂无可回放会话";
  elements.viewerEmptyDetail.textContent = replay?.state === "running"
    ? "正在生成手部识别回放"
    : hasReplaySessions ? "当前会话尚未生成识别回放" : "完成一次录制后可在这里回放";
}

function setViewerMode(mode) {
  if (mode !== "live" && mode !== "replay") return;
  state.viewerMode = mode;
  if (mode === "live") {
    elements.handReplayVideo.pause();
  }
  renderViewerMedia();
}

function clearReplayVideo() {
  elements.handReplayVideo.pause();
  elements.handReplayVideo.removeAttribute("src");
  elements.handReplayVideo.removeAttribute("data-source");
  elements.handReplayVideo.removeAttribute("data-session-id");
  elements.handReplayVideo.load();
}

function renderHandTrackingStatus() {
  const status = state.handTrackingStatus;
  const result = status?.latest_result || null;
  const labels = {
    disabled: "已禁用",
    idle: "等待视频",
    loading: "模型运行中",
    ready: "识别在线",
    error: "识别异常",
  };
  elements.handTrackingState.textContent = status ? labels[status.state] || status.state : "不可用";
  elements.handTrackingState.dataset.status = status?.state || "error";
  elements.handTrackingDetail.textContent = state.handTrackingError || status?.detail || "等待接收网关";
  elements.decodedFrameCount.textContent = (
    state.decodedPreviewStatus?.frames_received || 0
  ).toLocaleString("zh-CN");
  elements.handReceivedCount.textContent = (status?.live_frames_received || 0).toLocaleString(
    "zh-CN",
  );
  elements.handInputFrame.textContent = result ? result.frame_index.toLocaleString("zh-CN") : "--";
  elements.handInferenceTime.textContent = result
    ? `${(result.inference_duration_ns / 1_000_000).toFixed(0)} ms`
    : "--";
  elements.handInferenceCount.textContent = (status?.live_inferences || 0).toLocaleString("zh-CN");
  elements.handDroppedCount.textContent = (status?.live_frames_dropped || 0).toLocaleString("zh-CN");
  const hands = result?.hands || [];
  renderHandReadout(elements.leftHandReadout, hands.find((hand) => hand.handedness === "left"));
  renderHandReadout(elements.rightHandReadout, hands.find((hand) => hand.handedness === "right"));
  drawHandTrackingOverlay(result);

  const replay = status?.replay;
  if (replay?.state === "running") {
    const percent = replay.frame_total > 0
      ? Math.round((replay.frames_processed / replay.frame_total) * 100)
      : 0;
    elements.replayProgress.textContent = `${percent}% · ${replay.frames_processed}/${replay.frame_total}`;
  } else {
    elements.replayProgress.textContent = replay?.detail || "未运行";
  }
  const firstVideo = replay?.report?.videos?.[0];
  if (firstVideo) {
    const videoUrl = `${handTrackingEndpoint}/replays/${replay.report.session_id}/${replay.report.run_id}/${firstVideo.clip_id}`;
    if (elements.handReplayVideo.dataset.source !== videoUrl) {
      elements.handReplayVideo.dataset.source = videoUrl;
      elements.handReplayVideo.dataset.sessionId = replay.report.session_id;
      elements.handReplayVideo.src = videoUrl;
    }
  }
  elements.startReplayButton.disabled =
    !elements.replaySession.value || state.replayRequestInFlight || replay?.state === "running";
  renderViewerMedia();
}

async function startHandTrackingReplay() {
  const sessionId = elements.replaySession.value;
  if (!sessionId || state.replayRequestInFlight) return;
  state.replayRequestInFlight = true;
  clearReplayVideo();
  setViewerMode("replay");
  renderReplaySessionOptions();
  try {
    const response = await fetch(`${handTrackingEndpoint}/replays`, {
      method: "POST",
      cache: "no-store",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
    await readJsonResponse(response, `离线回放 HTTP ${response.status}`);
  } catch (error) {
    state.handTrackingError = error.message;
  } finally {
    state.replayRequestInFlight = false;
    renderReplaySessionOptions();
    renderHandTrackingStatus();
  }
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

function readDecodedPreviewStatus(payload) {
  const states = new Set(["waiting", "streaming", "error", "closed"]);
  if (
    payload === null
    || typeof payload !== "object"
    || payload.schema_version !== "1.0"
    || !states.has(payload.state)
    || !Number.isSafeInteger(payload.frames_received)
    || !Number.isSafeInteger(payload.frames_encoded)
    || !Number.isSafeInteger(payload.frames_dropped)
  ) {
    throw new Error("接收网关返回了无效的解码预览状态");
  }
  return payload;
}

function connectDecodedPreview() {
  if (document.hidden) return;
  window.clearTimeout(state.decodedPreviewReconnectTimer);
  state.decodedPreviewGeneration += 1;
  elements.decodedPreview.src = `${decodedPreviewEndpoint}?generation=${state.decodedPreviewGeneration}`;
}

function renderDecodedPreviewState() {
  const status = state.decodedPreviewStatus;
  const hasFrame = Boolean(status?.frames_encoded > 0);
  const wasReady = state.liveVideoReady;
  state.liveVideoReady = state.liveVideoReady || hasFrame;
  const ready = state.liveVideoReady;
  const stale = ready && Number.isFinite(status?.last_frame_age_ms)
    && status.last_frame_age_ms > 2000;
  const frameSize = status?.width && status?.height ? `${status.width} × ${status.height}` : "--";

  elements.connectionLight.classList.toggle("is-live", ready && !stale);
  elements.connectionLabel.textContent = !ready
    ? "等待 Glass3 视频"
    : stale ? "Glass3 画面暂停" : "Glass3 解码视频在线";
  elements.sourcePill.classList.toggle("pill-success", ready && !stale);
  elements.sourcePill.textContent = !ready ? "等待首帧" : stale ? "保留末帧" : "解码在线";
  elements.previewStatus.textContent = state.decodedPreviewError
    || (!ready ? "等待首帧" : stale ? "输入暂停，末帧保留" : "解码帧持续输出");
  elements.decodedPreview.classList.toggle("is-ready", ready);
  elements.frameSize.textContent = frameSize;
  elements.previewFps.textContent = Number.isFinite(status?.output_fps)
    ? `${status.output_fps.toFixed(1)} FPS`
    : "--";
  elements.decodedFrameCount.textContent = (status?.frames_received || 0).toLocaleString("zh-CN");
  if (status?.latest_frame_index !== null && status?.latest_frame_index !== undefined) {
    if (state.lastDecodedFrameIndex !== status.latest_frame_index) {
      state.lastDecodedFrameIndex = status.latest_frame_index;
      elements.lastFrameTime.textContent = new Date().toLocaleTimeString("zh-CN", {
        hour12: false,
      });
    }
  } else if (!ready) {
    elements.lastFrameTime.textContent = "--";
  }
  if (!wasReady && ready) {
    addEvent("OK", "Glass3 视频已连接", "网关 H.264 单次解码帧");
  }
  renderViewerMedia();
}

async function pollDecodedPreviewStatus() {
  if (state.decodedPreviewPollInFlight || document.hidden) {
    scheduleDecodedPreviewPoll();
    return;
  }
  state.decodedPreviewPollInFlight = true;
  try {
    const response = await fetch(decodedPreviewStatusEndpoint, { cache: "no-store" });
    const payload = await readJsonResponse(response, `解码预览 HTTP ${response.status}`);
    state.decodedPreviewStatus = readDecodedPreviewStatus(payload);
    state.decodedPreviewError = null;
  } catch (error) {
    state.decodedPreviewError = error.message;
  } finally {
    state.decodedPreviewPollInFlight = false;
    renderDecodedPreviewState();
    scheduleDecodedPreviewPoll();
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
elements.viewerModeLive.addEventListener("click", () => setViewerMode("live"));
elements.viewerModeReplay.addEventListener("click", () => setViewerMode("replay"));
elements.decodedPreview.addEventListener("error", () => {
  state.decodedPreviewError = "解码预览连接中断，正在恢复";
  renderDecodedPreviewState();
  scheduleDecodedPreviewReconnect();
});
elements.streamToggleButton.addEventListener("click", () => {
  sendStreamControlCommand(elements.streamToggleButton.dataset.action);
});
elements.recordingToggleButton.addEventListener("click", () => {
  sendRecordingCommand(elements.recordingToggleButton.dataset.action);
});
elements.newSessionButton.addEventListener("click", requestNewCollectionSession);
elements.resetImuButton.addEventListener("click", resetImuReference);
elements.startReplayButton.addEventListener("click", startHandTrackingReplay);
elements.replaySession.addEventListener("change", renderHandTrackingStatus);
document.addEventListener("fullscreenchange", () => {
  elements.fullscreenButton.textContent = document.fullscreenElement ? "退出全屏" : "全屏";
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    connectDecodedPreview();
    scheduleDecodedPreviewPoll(0);
    scheduleControlPoll(0);
    scheduleRecordingPoll(0);
    scheduleCollectionPoll(0);
    connectHandTrackingEvents();
    scheduleImuPoll(0);
  } else {
    closeHandTrackingEvents();
  }
});
window.addEventListener("beforeunload", () => {
  window.clearTimeout(state.decodedPreviewReconnectTimer);
  window.clearTimeout(state.decodedPreviewPollTimer);
  window.clearTimeout(state.controlPollTimer);
  window.clearTimeout(state.recordingPollTimer);
  window.clearTimeout(state.recordingCountdownTimer);
  window.clearTimeout(state.collectionPollTimer);
  window.clearTimeout(state.imuPollTimer);
  closeHandTrackingEvents();
  viewerResizeObserver.disconnect();
  imuScene?.dispose();
});

const viewerResizeObserver = new ResizeObserver(() => {
  drawHandTrackingOverlay(state.handTrackingStatus?.latest_result || null);
});
viewerResizeObserver.observe(elements.viewerStage);

addEvent("INFO", "客户端已启动", "等待 Glass3 视频、控制通路和 IMU 数据");
renderDecodedPreviewState();
renderStreamControl();
renderRecordingControl();
renderCollectionOverview();
renderReplaySessionOptions();
renderHandTrackingStatus();
setImuUnavailable("接收网关尚未收到姿态数据");
connectDecodedPreview();
pollDecodedPreviewStatus();
pollStreamControlStatus();
pollRecordingStatus();
pollCollectionLibrary();
connectHandTrackingEvents();
pollImuStatus();
