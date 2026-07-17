export const recordingStatusEndpoint =
  "http://127.0.0.1:8770/api/v1/recordings/status";
export const recordingCommandEndpoint =
  "http://127.0.0.1:8770/api/v1/recordings/commands";
export const recordingLibraryEndpoint =
  "http://127.0.0.1:8770/api/v1/recordings/library";

const recordingIdPattern = /^[0-9a-f]{32}$/;

export function recordingDeleteEndpoint(sessionId, clipId) {
  if (!recordingIdPattern.test(sessionId) || !recordingIdPattern.test(clipId)) {
    throw new Error("录制片段标识无效");
  }
  return `http://127.0.0.1:8770/api/v1/recordings/clips/${sessionId}/${clipId}`;
}

export function recordingSessionEndpoint(sessionId) {
  if (!recordingIdPattern.test(sessionId)) {
    throw new Error("录制会话标识无效");
  }
  return `http://127.0.0.1:8770/api/v1/recordings/sessions/${sessionId}`;
}

export const recordingStates = new Set([
  "unavailable",
  "ready",
  "countdown",
  "recording",
  "finalizing",
  "error",
]);

const gatewayOrigin = new URL(recordingStatusEndpoint).origin;

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isNullableString(value) {
  return value === null || (typeof value === "string" && value.length > 0);
}

function isNullableUnixMs(value) {
  return value === null || (Number.isSafeInteger(value) && value >= 0);
}

function requireFiniteNumber(value, name, minimum = 0) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum) {
    throw new Error(`录制服务返回了无效的 ${name}`);
  }
  return value;
}

function requireSafeInteger(value, name, minimum = 0) {
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new Error(`录制服务返回了无效的 ${name}`);
  }
  return value;
}

function readOutput(output) {
  if (
    !isRecord(output) ||
    output.width !== 1280 ||
    output.height !== 720 ||
    output.fps !== 30 ||
    output.container !== "mp4" ||
    output.video_codec !== "h264"
  ) {
    throw new Error("录制服务返回了无效的输出格式");
  }
  return {
    width: output.width,
    height: output.height,
    fps: output.fps,
    container: output.container,
    video_codec: output.video_codec,
  };
}

export function readRecordingStatus(payload) {
  if (
    !isRecord(payload) ||
    payload.schema_version !== "1.0" ||
    !recordingStates.has(payload.state) ||
    !isNullableString(payload.session_id) ||
    !isNullableString(payload.clip_id) ||
    !isNullableUnixMs(payload.recording_starts_at_unix_ms) ||
    !isNullableUnixMs(payload.recording_started_at_unix_ms) ||
    typeof payload.detail !== "string"
  ) {
    throw new Error("录制服务返回了无效的状态");
  }

  if (payload.state === "countdown" && payload.recording_starts_at_unix_ms === null) {
    throw new Error("录制倒计时缺少开始时间");
  }
  if (payload.state === "recording" && payload.recording_started_at_unix_ms === null) {
    throw new Error("录制状态缺少实际开始时间");
  }

  return {
    schema_version: payload.schema_version,
    state: payload.state,
    session_id: payload.session_id,
    clip_id: payload.clip_id,
    recording_starts_at_unix_ms: payload.recording_starts_at_unix_ms,
    recording_started_at_unix_ms: payload.recording_started_at_unix_ms,
    detail: payload.detail,
    output: readOutput(payload.output),
  };
}

function readMediaUrl(value) {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error("录制片段缺少媒体地址");
  }
  let mediaUrl;
  try {
    mediaUrl = new URL(value, gatewayOrigin);
  } catch {
    throw new Error("录制片段媒体地址无效");
  }
  if (mediaUrl.origin !== gatewayOrigin || !["http:", "https:"].includes(mediaUrl.protocol)) {
    throw new Error("录制片段媒体地址不属于本机接收网关");
  }
  return mediaUrl.href;
}

function readClip(clip) {
  if (!isRecord(clip) || !recordingIdPattern.test(clip.clip_id)) {
    throw new Error("录制服务返回了无效的片段");
  }
  return {
    clip_id: clip.clip_id,
    recorded_at_unix_ms: requireSafeInteger(
      clip.recorded_at_unix_ms,
      "recorded_at_unix_ms",
    ),
    duration_ms: requireSafeInteger(clip.duration_ms, "duration_ms"),
    width: requireSafeInteger(clip.width, "width", 1),
    height: requireSafeInteger(clip.height, "height", 1),
    fps: requireFiniteNumber(clip.fps, "fps", 0.001),
    file_size_bytes: requireSafeInteger(clip.file_size_bytes, "file_size_bytes"),
    media_url: readMediaUrl(clip.media_url),
  };
}

function readSession(session) {
  if (
    !isRecord(session) ||
    !recordingIdPattern.test(session.session_id) ||
    !Array.isArray(session.clips)
  ) {
    throw new Error("录制服务返回了无效的会话");
  }
  const displayName = session.display_name ?? null;
  if (
    displayName !== null &&
    (
      typeof displayName !== "string" ||
      displayName.length === 0 ||
      displayName.length > 64 ||
      displayName !== displayName.trim() ||
      /[\u0000-\u001f\u007f]/u.test(displayName)
    )
  ) {
    throw new Error("录制服务返回了无效的会话名称");
  }
  return {
    session_id: session.session_id,
    started_at_unix_ms: requireSafeInteger(
      session.started_at_unix_ms,
      "started_at_unix_ms",
    ),
    display_name: displayName,
    clips: session.clips.map(readClip),
  };
}

export function readRecordingLibrary(payload) {
  if (
    !isRecord(payload) ||
    payload.schema_version !== "1.0" ||
    !Array.isArray(payload.sessions)
  ) {
    throw new Error("录制服务返回了无效的媒体库");
  }
  return {
    schema_version: payload.schema_version,
    sessions: payload.sessions.map(readSession),
  };
}

export async function readJsonResponse(response, fallbackMessage) {
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = isRecord(payload) && typeof payload.detail === "string"
      ? payload.detail
      : fallbackMessage;
    throw new Error(detail);
  }
  if (!isRecord(payload)) throw new Error(fallbackMessage);
  return payload;
}
