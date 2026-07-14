const elements = {
  canvas: document.querySelector("#scene-canvas"),
  viewerStage: document.querySelector("#viewer-stage"),
  connectionLight: document.querySelector("#connection-light"),
  connectionLabel: document.querySelector("#connection-label"),
  sessionLabel: document.querySelector("#session-label"),
  recordButton: document.querySelector("#record-button"),
  recordLabel: document.querySelector("#record-label"),
  recordingPill: document.querySelector("#recording-pill"),
  viewerEmpty: document.querySelector("#viewer-empty"),
  resumeSessionButton: document.querySelector("#resume-session-button"),
  stopSessionButton: document.querySelector("#stop-session-button"),
  fullscreenButton: document.querySelector("#fullscreen-button"),
  overlayButton: document.querySelector("#overlay-button"),
  leftToggle: document.querySelector("#left-trajectory-toggle"),
  rightToggle: document.querySelector("#right-trajectory-toggle"),
  resolutionBadge: document.querySelector("#resolution-badge"),
  calibrationBanner: document.querySelector("#calibration-banner"),
  calibrationProfile: document.querySelector("#calibration-profile"),
  calibrationStateCopy: document.querySelector("#calibration-state-copy"),
  reprojectionError: document.querySelector("#reprojection-error"),
  frameSeq: document.querySelector("#frame-seq"),
  sdkTime: document.querySelector("#sdk-time"),
  timelineProgress: document.querySelector("#timeline-progress"),
  timelineCursor: document.querySelector("#timeline-cursor"),
  feedbackLatencyProperty: document.querySelector("#feedback-latency-property"),
  modelInputProperty: document.querySelector("#model-input-property"),
  historyProperty: document.querySelector("#history-property"),
  horizonProperty: document.querySelector("#horizon-property"),
  leftConfidence: document.querySelector("#left-confidence"),
  rightConfidence: document.querySelector("#right-confidence"),
  mediaLatency: document.querySelector("#media-latency"),
  inferenceLatency: document.querySelector("#inference-latency"),
  captureRate: document.querySelector("#capture-rate"),
  inferenceRate: document.querySelector("#inference-rate"),
  droppedFrames: document.querySelector("#dropped-frames"),
  gpuMemory: document.querySelector("#gpu-memory"),
  sessionSize: document.querySelector("#session-size"),
  sourcePill: document.querySelector("#source-pill"),
  eventRows: document.querySelector("#event-rows"),
  clearEventsButton: document.querySelector("#clear-events-button"),
  settingsForm: document.querySelector("#settings-form"),
  settingsMessage: document.querySelector("#settings-message"),
  toastRegion: document.querySelector("#toast-region"),
};

const state = {
  server: null,
  telemetry: null,
  socket: null,
  reconnectTimer: null,
  reconnectDelayMs: 700,
  lastTelemetryAt: 0,
  overlayVisible: true,
  events: [],
  frameAccumulator: 0,
};

const context = elements.canvas.getContext("2d", { alpha: false });

if (window.location.search) {
  window.history.replaceState({}, "", window.location.pathname);
}

function apiUrl(path) {
  return `${window.location.origin}${path}`;
}

async function request(path, options = {}) {
  const response = await fetch(apiUrl(path), {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const message = payload.detail || `${response.status} ${response.statusText}`;
    throw new Error(typeof message === "string" ? message : "请求参数无效");
  }
  return response.json();
}

async function initialize() {
  bindControls();
  addEvent("INFO", "操作台启动", "正在加载服务端状态");
  try {
    state.server = await request("/api/v1/state");
    renderServerState();
    populateSettings(state.server.settings);
    addEvent("OK", "模拟源已就绪", state.server.session_id);
    addEvent("WARN", "使用模拟标定", state.server.calibration.profile_id);
  } catch (error) {
    setConnectionState("offline", "服务不可用");
    addEvent("WARN", "状态加载失败", error.message);
  }
  connectTelemetry();
  requestAnimationFrame(drawFrame);
  window.setInterval(checkFreshness, 500);
}

function bindControls() {
  document.querySelectorAll(".inspector-tab").forEach((tab) => {
    tab.addEventListener("click", () => activateInspectorPanel(tab.dataset.panel));
  });

  elements.overlayButton.addEventListener("click", () => {
    state.overlayVisible = !state.overlayVisible;
    elements.overlayButton.textContent = state.overlayVisible ? "隐藏轨迹" : "显示轨迹";
    elements.overlayButton.setAttribute("aria-pressed", String(state.overlayVisible));
  });

  elements.recordButton.addEventListener("click", toggleRecording);
  elements.stopSessionButton.addEventListener("click", () => setSessionActive(false));
  elements.resumeSessionButton.addEventListener("click", () => setSessionActive(true));
  elements.fullscreenButton.addEventListener("click", toggleFullscreen);
  elements.clearEventsButton.addEventListener("click", () => {
    state.events = [];
    renderEvents();
  });
  elements.settingsForm.addEventListener("submit", saveSettings);

  window.addEventListener("resize", resizeCanvas);
  document.addEventListener("fullscreenchange", () => {
    elements.fullscreenButton.textContent = document.fullscreenElement ? "退出全屏" : "全屏";
    resizeCanvas();
  });
}

function activateInspectorPanel(panelId) {
  document.querySelectorAll(".inspector-tab").forEach((tab) => {
    const active = tab.dataset.panel === panelId;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".inspector-panel").forEach((panel) => {
    const active = panel.id === panelId;
    panel.classList.toggle("is-active", active);
    panel.hidden = !active;
  });
}

function connectTelemetry() {
  if (state.socket && state.socket.readyState <= WebSocket.OPEN) {
    return;
  }
  setConnectionState("connecting", "正在连接");
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  state.socket = new WebSocket(`${protocol}//${window.location.host}/api/v1/telemetry`);

  state.socket.addEventListener("open", () => {
    state.reconnectDelayMs = 700;
    setConnectionState("live", "模拟流在线");
  });

  state.socket.addEventListener("message", (event) => {
    try {
      const payload = JSON.parse(event.data);
      state.telemetry = payload;
      state.lastTelemetryAt = Date.now();
      renderTelemetry(payload);
    } catch (error) {
      addEvent("WARN", "遥测解析失败", error.message);
    }
  });

  state.socket.addEventListener("close", () => {
    setConnectionState("offline", "连接已断开");
    window.clearTimeout(state.reconnectTimer);
    state.reconnectTimer = window.setTimeout(connectTelemetry, state.reconnectDelayMs);
    state.reconnectDelayMs = Math.min(state.reconnectDelayMs * 1.8, 8000);
  });

  state.socket.addEventListener("error", () => state.socket.close());
}

function checkFreshness() {
  if (state.lastTelemetryAt && Date.now() - state.lastTelemetryAt > 1500) {
    setConnectionState("offline", "遥测已过期");
  }
}

function setConnectionState(mode, label) {
  elements.connectionLight.classList.toggle("is-live", mode === "live");
  elements.connectionLight.classList.toggle("is-offline", mode === "offline");
  elements.connectionLabel.textContent = label;
  if (elements.sourcePill) {
    elements.sourcePill.textContent = mode === "live" ? "模拟在线" : "无数据";
    elements.sourcePill.classList.toggle("pill-success", mode === "live");
  }
}

function renderServerState() {
  if (!state.server) return;
  const server = state.server;
  elements.sessionLabel.textContent = server.session_id;
  elements.viewerEmpty.hidden = server.session_phase === "live";
  elements.stopSessionButton.disabled = server.session_phase !== "live";
  elements.recordButton.disabled = server.session_phase !== "live";
  renderRecording(server.recording);
  renderSettingsSummary(server.settings);
  renderCalibration(server.calibration);
}

function renderTelemetry(telemetry) {
  const metrics = telemetry.metrics;
  elements.frameSeq.textContent = String(telemetry.frame_seq).padStart(6, "0");
  elements.sdkTime.textContent = formatDuration(telemetry.captured_at_sdk_ms);
  elements.mediaLatency.textContent = metrics.media_latency_ms.toFixed(1);
  elements.inferenceLatency.textContent = metrics.inference_latency_ms.toFixed(1);
  elements.captureRate.textContent = metrics.capture_fps.toFixed(1);
  elements.inferenceRate.textContent = metrics.inference_fps.toFixed(1);
  elements.droppedFrames.textContent = String(metrics.dropped_frames);
  elements.gpuMemory.textContent = metrics.gpu_memory_gb.toFixed(1);
  elements.feedbackLatencyProperty.textContent = `${metrics.feedback_latency_ms.toFixed(1)} ms`;

  const left = telemetry.hands.find((hand) => hand.side === "left");
  const right = telemetry.hands.find((hand) => hand.side === "right");
  elements.leftConfidence.textContent = left ? `${Math.round(left.confidence * 100)}%` : "--";
  elements.rightConfidence.textContent = right ? `${Math.round(right.confidence * 100)}%` : "--";

  const settings = state.server?.settings;
  if (settings) {
    const elapsedMinutes = telemetry.captured_at_sdk_ms / 60_000;
    const limit = settings.session_limit_minutes || 30;
    const progress = Math.min(100, (elapsedMinutes / limit) * 100);
    elements.timelineProgress.style.width = `${progress}%`;
    elements.timelineCursor.style.left = `${progress}%`;
    const bytes = (telemetry.captured_at_sdk_ms / 1000) * (settings.target_bitrate_kbps * 1000 / 8);
    elements.sessionSize.textContent = formatBytes(bytes);
  }

  if (state.server && state.server.recording !== telemetry.recording) {
    state.server.recording = telemetry.recording;
    renderRecording(telemetry.recording);
  }
}

function renderSettingsSummary(settings) {
  elements.resolutionBadge.textContent = `${settings.video_width} × ${settings.video_height} · ${settings.capture_fps} FPS`;
  elements.modelInputProperty.textContent = `${settings.history_frames} × 224² RGB`;
  elements.historyProperty.textContent = `${(settings.history_frames / settings.inference_fps).toFixed(1)} s`;
  elements.horizonProperty.textContent = `${(settings.prediction_steps * settings.prediction_interval_ms / 1000).toFixed(1)} s`;
}

function renderCalibration(calibration) {
  elements.calibrationProfile.textContent = calibration.profile_id;
  elements.calibrationBanner.querySelector("span").textContent = calibration.profile_id;
  elements.reprojectionError.textContent = calibration.reprojection_error_px == null
    ? "--"
    : `${calibration.reprojection_error_px.toFixed(2)} px`;
  const messages = {
    simulated: "模拟配置，不代表真机空间标定结果",
    verified: "真机空间标定已验证",
    missing: "缺少空间标定配置",
    invalid: "空间标定配置无效",
  };
  elements.calibrationStateCopy.textContent = messages[calibration.state] || "标定状态未知";
  elements.calibrationBanner.querySelector("strong").textContent = calibration.state === "verified"
    ? "空间标定已验证"
    : "模拟标定";
}

function renderRecording(recording) {
  elements.recordButton.classList.toggle("is-recording", recording);
  elements.recordLabel.textContent = recording ? "停止录制" : "开始录制";
  elements.recordingPill.textContent = recording ? "录制中" : "待机";
  elements.recordingPill.classList.toggle("pill-success", recording);
}

async function toggleRecording() {
  if (!state.server) return;
  const next = !state.server.recording;
  elements.recordButton.disabled = true;
  try {
    state.server = await request(`/api/v1/recording/${next ? "start" : "stop"}`, { method: "POST" });
    renderServerState();
    addEvent(next ? "OK" : "INFO", next ? "开始录制" : "停止录制", state.server.session_id);
    showToast(next ? "录制已开始" : "录制已停止");
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    elements.recordButton.disabled = state.server?.session_phase !== "live";
  }
}

async function setSessionActive(active) {
  try {
    state.server = await request(`/api/v1/session/${active ? "start" : "stop"}`, { method: "POST" });
    renderServerState();
    addEvent(active ? "OK" : "INFO", active ? "会话已启动" : "会话已停止", state.server.session_id);
    showToast(active ? "模拟会话已启动" : "会话已停止");
  } catch (error) {
    showToast(error.message, "error");
  }
}

async function saveSettings(event) {
  event.preventDefault();
  const current = state.server?.settings;
  if (!current) return;
  const [videoWidth, videoHeight] = document.querySelector("#video-resolution").value.split("x").map(Number);
  const payload = {
    ...current,
    video_width: videoWidth,
    video_height: videoHeight,
    capture_fps: numberValue("#capture-fps"),
    target_bitrate_kbps: numberValue("#target-bitrate"),
    inference_fps: numberValue("#inference-fps"),
    history_frames: numberValue("#history-frames"),
    prediction_steps: numberValue("#prediction-steps"),
    prediction_interval_ms: numberValue("#prediction-interval"),
    max_feedback_age_ms: numberValue("#feedback-age"),
    recording_segment_seconds: numberValue("#segment-seconds"),
    session_limit_minutes: numberValue("#session-limit"),
    min_free_disk_gb: numberValue("#min-disk"),
    retain_recordings: document.querySelector("#retain-recordings").checked,
  };
  elements.settingsMessage.textContent = "正在保存";
  try {
    state.server = await request("/api/v1/settings", {
      method: "PUT",
      body: JSON.stringify(payload),
    });
    renderServerState();
    populateSettings(state.server.settings);
    elements.settingsMessage.textContent = `配置版本 ${state.server.settings_revision}`;
    addEvent("OK", "参数已更新", `revision ${state.server.settings_revision}`);
    showToast("运行参数已保存");
  } catch (error) {
    elements.settingsMessage.textContent = error.message;
    showToast(error.message, "error");
  }
}

function populateSettings(settings) {
  document.querySelector("#video-resolution").value = `${settings.video_width}x${settings.video_height}`;
  document.querySelector("#capture-fps").value = settings.capture_fps;
  document.querySelector("#target-bitrate").value = settings.target_bitrate_kbps;
  document.querySelector("#inference-fps").value = settings.inference_fps;
  document.querySelector("#history-frames").value = settings.history_frames;
  document.querySelector("#prediction-steps").value = settings.prediction_steps;
  document.querySelector("#prediction-interval").value = settings.prediction_interval_ms;
  document.querySelector("#feedback-age").value = settings.max_feedback_age_ms;
  document.querySelector("#segment-seconds").value = settings.recording_segment_seconds;
  document.querySelector("#session-limit").value = settings.session_limit_minutes;
  document.querySelector("#min-disk").value = settings.min_free_disk_gb;
  document.querySelector("#retain-recordings").checked = settings.retain_recordings;
}

function numberValue(selector) {
  return Number(document.querySelector(selector).value);
}

async function toggleFullscreen() {
  try {
    if (document.fullscreenElement) {
      await document.exitFullscreen();
    } else {
      await elements.viewerStage.requestFullscreen();
    }
  } catch (error) {
    showToast(error.message, "error");
  }
}

function addEvent(level, event, detail) {
  state.events.unshift({
    time: new Date().toLocaleTimeString("zh-CN", { hour12: false }),
    level,
    event,
    detail,
  });
  state.events = state.events.slice(0, 8);
  renderEvents();
}

function renderEvents() {
  elements.eventRows.replaceChildren();
  state.events.forEach((entry) => {
    const row = document.createElement("tr");
    const time = document.createElement("td");
    const level = document.createElement("td");
    const event = document.createElement("td");
    const detail = document.createElement("td");
    time.textContent = entry.time;
    level.textContent = entry.level;
    level.className = `event-level ${entry.level.toLowerCase()}`;
    event.textContent = entry.event;
    detail.textContent = entry.detail;
    row.append(time, level, event, detail);
    elements.eventRows.append(row);
  });
}

function showToast(message, mode = "success") {
  const toast = document.createElement("div");
  toast.className = "toast";
  if (mode === "error") toast.style.borderLeftColor = "var(--red)";
  toast.textContent = message;
  elements.toastRegion.append(toast);
  window.setTimeout(() => toast.remove(), 3200);
}

function resizeCanvas() {
  const bounds = elements.canvas.getBoundingClientRect();
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(bounds.width * pixelRatio));
  const height = Math.max(1, Math.round(bounds.height * pixelRatio));
  if (elements.canvas.width !== width || elements.canvas.height !== height) {
    elements.canvas.width = width;
    elements.canvas.height = height;
  }
}

function drawFrame() {
  resizeCanvas();
  const width = elements.canvas.width;
  const height = elements.canvas.height;
  drawSyntheticScene(context, width, height, state.telemetry);
  requestAnimationFrame(drawFrame);
}

function drawSyntheticScene(ctx, width, height, telemetry) {
  ctx.fillStyle = "#a5b4ae";
  ctx.fillRect(0, 0, width, height);

  const horizon = height * 0.47;
  ctx.fillStyle = "#bbc5bd";
  ctx.fillRect(0, 0, width, horizon);
  ctx.fillStyle = "#6e5f50";
  ctx.beginPath();
  ctx.moveTo(0, height);
  ctx.lineTo(width, height);
  ctx.lineTo(width * 0.82, horizon);
  ctx.lineTo(width * 0.18, horizon);
  ctx.closePath();
  ctx.fill();

  drawPerspectiveGrid(ctx, width, height, horizon);
  drawWorkObjects(ctx, width, height);

  const left = telemetry?.hands?.find((hand) => hand.side === "left");
  const right = telemetry?.hands?.find((hand) => hand.side === "right");
  drawHand(ctx, projectFirst(left, width, height), "#315966", -1, width, height);
  drawHand(ctx, projectFirst(right, width, height), "#7a443f", 1, width, height);

  if (state.overlayVisible && telemetry && telemetry.session_phase === "live") {
    if (elements.leftToggle.checked && left) drawTrajectory(ctx, left, width, height, "#32bdd0");
    if (elements.rightToggle.checked && right) drawTrajectory(ctx, right, width, height, "#ff7168");
  }

  ctx.strokeStyle = "rgba(242,245,241,0.32)";
  ctx.lineWidth = Math.max(1, width / 1000);
  ctx.beginPath();
  ctx.moveTo(width / 2 - 10, height / 2);
  ctx.lineTo(width / 2 + 10, height / 2);
  ctx.moveTo(width / 2, height / 2 - 10);
  ctx.lineTo(width / 2, height / 2 + 10);
  ctx.stroke();
}

function drawPerspectiveGrid(ctx, width, height, horizon) {
  ctx.strokeStyle = "rgba(232,238,233,0.16)";
  ctx.lineWidth = Math.max(1, width / 1200);
  for (let i = 0; i <= 8; i += 1) {
    const bottomX = (i / 8) * width;
    const topX = width * 0.18 + (i / 8) * width * 0.64;
    ctx.beginPath();
    ctx.moveTo(bottomX, height);
    ctx.lineTo(topX, horizon);
    ctx.stroke();
  }
  for (let i = 1; i <= 5; i += 1) {
    const t = i / 6;
    const y = horizon + (height - horizon) * t * t;
    const inset = width * 0.18 * (1 - t * t);
    ctx.beginPath();
    ctx.moveTo(inset, y);
    ctx.lineTo(width - inset, y);
    ctx.stroke();
  }
}

function drawWorkObjects(ctx, width, height) {
  const objects = [
    { x: 0.39, y: 0.50, w: 0.10, h: 0.12, color: "#3a7b60" },
    { x: 0.52, y: 0.48, w: 0.09, h: 0.14, color: "#d7ae4b" },
    { x: 0.63, y: 0.53, w: 0.11, h: 0.10, color: "#4b6c91" },
  ];
  objects.forEach((object) => {
    const x = object.x * width;
    const y = object.y * height;
    const w = object.w * width;
    const h = object.h * height;
    ctx.fillStyle = "rgba(20,22,21,0.18)";
    ctx.fillRect(x + w * 0.06, y + h * 0.12, w, h);
    ctx.fillStyle = object.color;
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = "rgba(255,255,255,0.28)";
    ctx.strokeRect(x, y, w, h);
  });
}

function projectFirst(hand, width, height) {
  if (!hand || !hand.waypoints.length) return null;
  return projectPoint(hand.waypoints[0], width, height);
}

function projectPoint(point, width, height) {
  const fx = width * 0.84;
  const fy = height * 1.5;
  return {
    x: width * 0.5 + fx * (point.x_m / point.z_m),
    y: height * 0.47 + fy * (point.y_m / point.z_m),
  };
}

function drawHand(ctx, point, color, direction, width, height) {
  if (!point) return;
  const scale = Math.max(0.65, width / 1100);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineCap = "round";
  ctx.lineWidth = 18 * scale;
  ctx.beginPath();
  ctx.moveTo(point.x - direction * 70 * scale, height + 10);
  ctx.quadraticCurveTo(point.x - direction * 38 * scale, point.y + 55 * scale, point.x, point.y);
  ctx.stroke();
  ctx.beginPath();
  ctx.ellipse(point.x, point.y, 23 * scale, 17 * scale, direction * 0.25, 0, Math.PI * 2);
  ctx.fill();
}

function drawTrajectory(ctx, hand, width, height, color) {
  if (!hand.present || !hand.waypoints.length) return;
  const points = hand.waypoints.map((point) => projectPoint(point, width, height));
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(3, width / 300);
  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point.x, point.y);
    else ctx.lineTo(point.x, point.y);
  });
  ctx.stroke();

  points.forEach((point, index) => {
    const confidence = hand.waypoints[index].confidence;
    const radius = Math.max(3, width / 400) * (0.75 + confidence * 0.35);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(10,12,11,0.75)";
    ctx.lineWidth = Math.max(1, width / 900);
    ctx.stroke();
  });

  const end = points.at(-1);
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(1.5, width / 600);
  ctx.beginPath();
  ctx.arc(end.x, end.y, Math.max(8, width / 170), 0, Math.PI * 2);
  ctx.stroke();
}

function formatDuration(milliseconds) {
  const minutes = Math.floor(milliseconds / 60_000);
  const seconds = Math.floor((milliseconds % 60_000) / 1000);
  const millis = milliseconds % 1000;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

initialize();
