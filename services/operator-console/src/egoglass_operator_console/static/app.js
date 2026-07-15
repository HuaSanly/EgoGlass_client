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
  lastFrameTime: document.querySelector("#last-frame-time"),
  linkState: document.querySelector("#link-state"),
  fullscreenButton: document.querySelector("#fullscreen-button"),
  clearEventsButton: document.querySelector("#clear-events-button"),
  eventRows: document.querySelector("#event-rows"),
};

const state = {
  liveVideoReady: false,
  liveVideoTimer: null,
  events: [],
};

const liveVideoEndpoint = "http://127.0.0.1:8770/api/v1/webrtc/frame.jpg";

if (window.location.search) {
  window.history.replaceState({}, "", window.location.pathname);
}

function scheduleFrame(delayMs) {
  window.clearTimeout(state.liveVideoTimer);
  if (document.hidden) return;
  state.liveVideoTimer = window.setTimeout(() => {
    elements.liveVideo.src = `${liveVideoEndpoint}?frame=${Date.now()}`;
  }, delayMs);
}

function connectLiveVideo() {
  elements.liveVideo.addEventListener("load", () => {
    const becameReady = !state.liveVideoReady;
    state.liveVideoReady = true;
    renderVideoState();
    if (becameReady) {
      addEvent("OK", "Glass3 视频已连接", "WebRTC H.264");
    }
    scheduleFrame(100);
  });

  elements.liveVideo.addEventListener("error", () => {
    const wasReady = state.liveVideoReady;
    state.liveVideoReady = false;
    renderVideoState();
    if (wasReady) {
      addEvent("WARN", "Glass3 视频已断开", "等待新的首帧");
    }
    scheduleFrame(500);
  });

  scheduleFrame(0);
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
  elements.linkState.classList.toggle("is-live", ready);

  if (ready && elements.liveVideo.naturalWidth > 0) {
    const frameSize = `${elements.liveVideo.naturalWidth} × ${elements.liveVideo.naturalHeight}`;
    elements.resolutionBadge.textContent = `${frameSize} · LIVE`;
    elements.frameSize.textContent = frameSize;
    elements.lastFrameTime.textContent = new Date().toLocaleTimeString("zh-CN", {
      hour12: false,
    });
  } else {
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

elements.fullscreenButton.addEventListener("click", toggleFullscreen);
elements.clearEventsButton.addEventListener("click", () => {
  state.events = [];
  renderEvents();
});
document.addEventListener("fullscreenchange", () => {
  elements.fullscreenButton.textContent = document.fullscreenElement ? "退出全屏" : "全屏";
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    window.clearTimeout(state.liveVideoTimer);
  } else {
    scheduleFrame(0);
  }
});
window.addEventListener("beforeunload", () => window.clearTimeout(state.liveVideoTimer));

addEvent("INFO", "客户端已启动", "等待 Glass3 首帧");
renderVideoState();
connectLiveVideo();
