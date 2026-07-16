import { ImuSceneController } from "./imu-scene.js";

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
  imuPollTimer: null,
  imuPollInFlight: false,
  imuConnected: false,
  imuOrientationReady: false,
  imuSceneError: null,
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
  const status = readStreamControlStatus(payload);
  state.controlState = status.state;
  state.controlDetail = status.detail;
  state.controlPollError = null;
  renderStreamControl();
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

function closeViewerPeer(peer) {
  if (state.peer !== peer) return;
  state.peer = null;
  peer.onconnectionstatechange = null;
  peer.close();
  stopFrameMonitoring();
  elements.liveVideo.srcObject = null;
  state.liveVideoReady = false;
  renderVideoState();
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
}

function applyImuStatus(payload) {
  const status = readImuStatus(payload);
  if (status.channel_state !== "receiving") {
    setImuUnavailable(
      status.channel_state === "ready" ? "IMU 通道已连接，等待首个样本" : "接收网关尚未收到姿态数据",
    );
    return;
  }

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
  console.warn("IMU scene initialization failed", error);
}

elements.fullscreenButton.addEventListener("click", toggleFullscreen);
elements.startStreamButton.addEventListener("click", () => sendStreamControlCommand("start"));
elements.stopStreamButton.addEventListener("click", () => sendStreamControlCommand("stop"));
elements.resetImuButton.addEventListener("click", () => imuScene?.resetReference());
document.addEventListener("fullscreenchange", () => {
  elements.fullscreenButton.textContent = document.fullscreenElement ? "退出全屏" : "全屏";
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    if (state.peer === null) scheduleViewerRetry(0);
    scheduleControlPoll(0);
    scheduleImuPoll(0);
  }
});
window.addEventListener("beforeunload", () => {
  window.clearTimeout(state.reconnectTimer);
  window.clearTimeout(state.controlPollTimer);
  window.clearTimeout(state.imuPollTimer);
  if (state.peer !== null) closeViewerPeer(state.peer);
  imuScene?.dispose();
});

renderVideoState();
renderStreamControl();
setImuUnavailable("接收网关尚未收到姿态数据");
connectLiveVideo();
pollStreamControlStatus();
pollImuStatus();
