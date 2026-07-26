const idPattern = /^[0-9a-f]{32}$/;

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function requireId(value, label) {
  if (!idPattern.test(value)) throw new Error(`${label}标识无效`);
  return value;
}

export async function readWorkspace() {
  const response = await fetch("/api/v1/annotations/workspace", { cache: "no-store" });
  const payload = await readJsonResponse(response, "无法读取标注工作区");
  if (
    payload.schema_version !== "1.0" ||
    !Array.isArray(payload.sessions) ||
    !Array.isArray(payload.implemented_strategies)
  ) {
    throw new Error("标注工作区格式无效");
  }
  return payload;
}

export async function readAnnotationSession(sessionId) {
  requireId(sessionId, "会话");
  const response = await fetch(`/api/v1/annotations/sessions/${sessionId}`, {
    cache: "no-store",
  });
  const payload = await readJsonResponse(response, "无法读取标注会话");
  if (!isRecord(payload.session) || !isRecord(payload.draft)) {
    throw new Error("标注会话格式无效");
  }
  return payload;
}

export async function requestEpisodeProposals(sessionId, request) {
  requireId(sessionId, "会话");
  const response = await fetch(`/api/v1/annotations/sessions/${sessionId}/proposals`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  const payload = await readJsonResponse(response, "无法生成切分候选");
  if (payload.contract_id !== "episode-proposal-v1" || !Array.isArray(payload.proposals)) {
    throw new Error("切分候选格式无效");
  }
  return payload;
}

export async function writeAnnotationDraft(sessionId, request) {
  requireId(sessionId, "会话");
  const response = await fetch(`/api/v1/annotations/sessions/${sessionId}/draft`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  return readJsonResponse(response, "无法保存标注草稿");
}

export async function publishAnnotation(sessionId, baseRevision) {
  requireId(sessionId, "会话");
  const response = await fetch(`/api/v1/annotations/sessions/${sessionId}/publish`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ base_revision: baseRevision }),
  });
  return readJsonResponse(response, "无法发布标注版本");
}

export async function annotationMediaUrl(path) {
  if (typeof path !== "string" || !path.startsWith("/api/v1/annotations/media/")) {
    throw new Error("标注视频地址无效");
  }
  return path;
}

export async function readJsonResponse(response, fallbackMessage) {
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    if (isRecord(payload) && typeof payload.detail === "string") {
      throw new Error(payload.detail);
    }
    if (
      isRecord(payload) &&
      isRecord(payload.detail) &&
      Array.isArray(payload.detail.issues)
    ) {
      throw new Error(payload.detail.issues.join("；"));
    }
    if (isRecord(payload) && Array.isArray(payload.detail)) {
      const messages = payload.detail
        .map((issue) => isRecord(issue) && typeof issue.msg === "string" ? issue.msg : null)
        .filter((message) => message !== null);
      if (messages.length > 0) throw new Error(messages.join("；"));
    }
    throw new Error(fallbackMessage);
  }
  if (!isRecord(payload)) throw new Error(fallbackMessage);
  return payload;
}
