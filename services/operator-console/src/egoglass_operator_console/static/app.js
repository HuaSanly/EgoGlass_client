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
  startStreamButton: document.querySelector("#start-stream-button"),
  stopStreamButton: document.querySelector("#stop-stream-button"),
  fullscreenButton: document.querySelector("#fullscreen-button"),
  clearEventsButton: document.querySelector("#clear-events-button"),
  eventRows: document.querySelector("#event-rows"),
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
  events: [],
};

const viewerSignalingEndpoint =
  "http://127.0.0.1:8770/api/v1/webrtc/viewer/sessions";
const streamControlEndpoint = "http://127.0.0.1:8770/api/v1/webrtc/control";
const streamControlCommandEndpoint = `${streamControlEndpoint}/commands`;
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

async function readJsonResponse(response, fallbackMessage) {
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload && typeof payload.detail === "string" ? payload.detail : fallbackMessage;
    throw new Error(detail);
  }
  return payload;
}

function applyStreamControlStatus(payload) {
  const previousState = state.controlState;
  const status = readStreamControlStatus(payload);
  state.controlState = status.state;
  state.controlDetail = status.detail;
  state.controlPollError = null;
  renderStreamControl();

  if (previousState !== status.state && status.state === "streaming") {
    addEvent("OK", "眼镜端视频已启动", status.detail || "控制状态已确认");
  } else if (previousState !== status.state && status.state === "stopped") {
    addEvent("INFO", "眼镜端视频已停止", status.detail || "控制通路保持在线");
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
    addEvent("WARN", action === "start" ? "启动视频失败" : "停止视频失败", error.message);
  } finally {
    state.controlCommandInFlight = false;
    renderStreamControl();
    scheduleControlPoll(0);
  }
}

function renderStreamControl() {
  const controlReady = controllableStreamStates.has(state.controlState);
  const busy = state.controlCommandInFlight || ["starting", "stopping"].includes(state.controlState);
  elements.startStreamButton.disabled =
    !controlReady || busy || state.controlState === "streaming";
  elements.stopStreamButton.disabled = !controlReady || busy || state.controlState === "stopped";
  elements.controlStatus.textContent = state.controlCommandInFlight
    ? "命令发送中"
    : streamControlLabels[state.controlState];
  elements.controlStatus.title = state.controlDetail || "";
  elements.controlStatusDot.dataset.state = state.controlState;

  const visibleError = state.controlCommandError || state.controlPollError;
  elements.controlError.textContent = visibleError || "";
  elements.controlError.title = visibleError || "";
  elements.controlError.hidden = visibleError === null;
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
  }
}

function addEvent(level, event, detail) {
  state.events.unshift({
    time: new Date().toLocaleTimeString("zh-CN", { hour12: false }),
    level,
    event,
    detail,
  });
  state.events = state.events.slice(0, 12);
  renderEvents();
}

function renderEvents() {
  elements.eventRows.replaceChildren();
  state.events.forEach((entry) => {
    const row = document.createElement("tr");
    const time = document.createElement("td");
    const level = document.createElement("td");
    const event = document.createElement("td");
    const eventTitle = document.createElement("strong");
    const eventDetail = document.createElement("span");
    time.textContent = entry.time;
    level.textContent = entry.level;
    level.className = `event-level ${entry.level.toLowerCase()}`;
    event.className = "event-message";
    eventTitle.textContent = entry.event;
    eventDetail.textContent = entry.detail;
    event.append(eventTitle, eventDetail);
    row.append(time, level, event);
    elements.eventRows.append(row);
  });
}

elements.fullscreenButton.addEventListener("click", toggleFullscreen);
elements.startStreamButton.addEventListener("click", () => sendStreamControlCommand("start"));
elements.stopStreamButton.addEventListener("click", () => sendStreamControlCommand("stop"));
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
  }
});
window.addEventListener("beforeunload", () => {
  window.clearTimeout(state.reconnectTimer);
  window.clearTimeout(state.controlPollTimer);
  if (state.peer !== null) closeViewerPeer(state.peer);
});

addEvent("INFO", "客户端已启动", "等待 Glass3 首帧");
renderVideoState();
renderStreamControl();
connectLiveVideo();
pollStreamControlStatus();
