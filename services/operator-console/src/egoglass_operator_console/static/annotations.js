import {
  annotationMediaUrl,
  publishAnnotation,
  readAnnotationSession,
  readWorkspace,
  requestEpisodeProposals,
  writeAnnotationDraft,
} from "./annotations-api.js";

const elements = {
  strategy: document.querySelector("#segmentation-strategy"),
  fixedWindowFields: [...document.querySelectorAll(".fixed-window-field")],
  windowDuration: document.querySelector("#window-duration"),
  windowStride: document.querySelector("#window-stride"),
  generateProposals: document.querySelector("#generate-proposals-button"),
  acceptProposals: document.querySelector("#accept-proposals-button"),
  undo: document.querySelector("#undo-annotation-button"),
  redo: document.querySelector("#redo-annotation-button"),
  saveDot: document.querySelector("#annotation-save-dot"),
  saveLabel: document.querySelector("#annotation-save-label"),
  publish: document.querySelector("#publish-annotation-button"),
  refresh: document.querySelector("#refresh-annotations-button"),
  sessionSummary: document.querySelector("#annotation-session-summary"),
  sessionList: document.querySelector("#annotation-session-list"),
  sessionEmpty: document.querySelector("#annotation-session-empty"),
  activeSessionName: document.querySelector("#active-session-name"),
  activeSessionDetail: document.querySelector("#active-session-detail"),
  clipSelect: document.querySelector("#annotation-clip-select"),
  video: document.querySelector("#annotation-video"),
  videoEmpty: document.querySelector("#annotation-video-empty"),
  jumpStart: document.querySelector("#jump-start-button"),
  previousFrame: document.querySelector("#previous-frame-button"),
  play: document.querySelector("#annotation-play-button"),
  nextFrame: document.querySelector("#next-frame-button"),
  jumpEnd: document.querySelector("#jump-end-button"),
  timecode: document.querySelector("#annotation-timecode"),
  playbackRate: document.querySelector("#annotation-playback-rate"),
  timeline: document.querySelector("#annotation-timeline"),
  ruler: document.querySelector("#timeline-ruler"),
  episodeTrack: document.querySelector("#episode-track"),
  phaseTrack: document.querySelector("#phase-track"),
  playhead: document.querySelector("#timeline-playhead"),
  markReadout: document.querySelector("#episode-mark-readout"),
  setEpisodeIn: document.querySelector("#set-episode-in-button"),
  createEpisode: document.querySelector("#create-episode-button"),
  splitEpisode: document.querySelector("#split-episode-button"),
  mergeEpisode: document.querySelector("#merge-episode-button"),
  deleteEpisode: document.querySelector("#delete-episode-button"),
  inspectorSummary: document.querySelector("#selected-episode-summary"),
  inspectorEmpty: document.querySelector("#annotation-inspector-empty"),
  inspectorForm: document.querySelector("#annotation-inspector-form"),
  taskId: document.querySelector("#label-task-id"),
  instruction: document.querySelector("#label-instruction"),
  verb: document.querySelector("#label-verb"),
  object: document.querySelector("#label-object"),
  target: document.querySelector("#label-target"),
  hand: document.querySelector("#label-hand"),
  outcome: document.querySelector("#label-outcome"),
  qualityFlags: [...document.querySelectorAll("#quality-flag-group input")],
  notes: document.querySelector("#label-notes"),
  phaseList: document.querySelector("#phase-list"),
  phaseMarkReadout: document.querySelector("#phase-mark-readout"),
  phaseKind: document.querySelector("#phase-kind"),
  phaseHand: document.querySelector("#phase-hand"),
  phaseVerb: document.querySelector("#phase-verb"),
  phaseObject: document.querySelector("#phase-object"),
  setPhaseIn: document.querySelector("#set-phase-in-button"),
  createPhase: document.querySelector("#create-phase-button"),
  message: document.querySelector("#annotation-message"),
};

const state = {
  workspace: null,
  detail: null,
  selectedSessionId: null,
  selectedClipId: null,
  selectedEpisodeId: null,
  proposalBatch: null,
  episodeMarkIn: null,
  phaseMarkIn: null,
  undoStack: [],
  redoStack: [],
  changeSerial: 0,
  dirty: false,
  saveInFlight: false,
  saveTimer: null,
};

function newId() {
  return crypto.randomUUID().replaceAll("-", "");
}

function clone(value) {
  return structuredClone(value);
}

function currentClip() {
  return state.detail?.session.clips.find((clip) => clip.clip_id === state.selectedClipId) ?? null;
}

function currentEpisode() {
  return state.detail?.draft.episodes.find(
    (episode) => episode.episode_id === state.selectedEpisodeId,
  ) ?? null;
}

function currentFrame() {
  const clip = currentClip();
  if (clip === null) return 0;
  return Math.max(0, Math.min(clip.frame_count, Math.round(elements.video.currentTime * clip.fps)));
}

function frameToSeconds(frame, clip = currentClip()) {
  return clip === null ? 0 : frame / clip.fps;
}

function formatTime(seconds) {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const minutes = Math.floor(safe / 60);
  const remainder = safe - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(3).padStart(6, "0")}`;
}

function formatFrame(frame) {
  return `F${String(frame).padStart(6, "0")}`;
}

function formatSessionTime(unixMs) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(new Date(unixMs));
}

function setMessage(message, type = "info") {
  elements.message.hidden = !message;
  elements.message.dataset.type = type;
  elements.message.textContent = message;
}

function setSaveState(label, status = "idle") {
  elements.saveLabel.textContent = label;
  elements.saveDot.dataset.status = status;
}

function updateCommandState() {
  const clip = currentClip();
  const episode = currentEpisode();
  const editable = state.detail?.session.editable === true;
  const hasClip = clip !== null;
  for (const control of [
    elements.jumpStart,
    elements.previousFrame,
    elements.play,
    elements.nextFrame,
    elements.jumpEnd,
    elements.playbackRate,
    elements.setEpisodeIn,
  ]) {
    control.disabled = !hasClip;
  }
  elements.createEpisode.disabled = !hasClip || !editable || state.episodeMarkIn === null;
  elements.splitEpisode.disabled = episode === null || !editable;
  elements.mergeEpisode.disabled = episode === null || !editable || nextEpisode() === null;
  elements.deleteEpisode.disabled = episode === null || !editable;
  elements.generateProposals.disabled = !hasClip || !editable || elements.strategy.value === "manual";
  elements.publish.disabled = !editable || state.detail?.draft.draft_revision < 1 || state.dirty;
  elements.undo.disabled = state.undoStack.length === 0;
  elements.redo.disabled = state.redoStack.length === 0;
}

function snapshotDraft() {
  return {
    segmentation_strategy: state.detail.draft.segmentation_strategy,
    default_labels: clone(state.detail.draft.default_labels),
    episodes: clone(state.detail.draft.episodes),
    selectedEpisodeId: state.selectedEpisodeId,
  };
}

function pushHistory() {
  if (state.detail === null) return;
  state.undoStack.push(snapshotDraft());
  if (state.undoStack.length > 80) state.undoStack.shift();
  state.redoStack = [];
}

function restoreSnapshot(snapshot) {
  state.detail.draft.segmentation_strategy = snapshot.segmentation_strategy;
  state.detail.draft.default_labels = clone(snapshot.default_labels);
  state.detail.draft.episodes = clone(snapshot.episodes);
  state.selectedEpisodeId = snapshot.selectedEpisodeId;
  markChanged();
  renderEditor();
}

function markChanged() {
  state.changeSerial += 1;
  state.dirty = true;
  setSaveState("有未保存修改", "dirty");
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(() => saveDraft(), 700);
  updateCommandState();
}

async function saveDraft() {
  if (state.detail === null || !state.dirty || state.saveInFlight) return;
  state.saveInFlight = true;
  const serial = state.changeSerial;
  setSaveState("正在保存草稿", "saving");
  try {
    const draft = await writeAnnotationDraft(state.selectedSessionId, {
      base_revision: state.detail.draft.draft_revision,
      segmentation_strategy: state.detail.draft.segmentation_strategy,
      default_labels: state.detail.draft.default_labels,
      episodes: state.detail.draft.episodes,
    });
    state.detail.draft.draft_revision = draft.draft_revision;
    state.detail.draft.updated_at_unix_ns = draft.updated_at_unix_ns;
    state.detail.draft.latest_published_revision_id = draft.latest_published_revision_id;
    if (serial === state.changeSerial) {
      state.dirty = false;
      setSaveState(`草稿已保存 · r${draft.draft_revision}`, "saved");
    } else {
      state.saveTimer = setTimeout(() => saveDraft(), 200);
    }
    updateSessionStatus("draft", draft.draft_revision, draft.latest_published_revision_id);
    renderSessionList();
  } catch (error) {
    setSaveState("草稿保存失败", "error");
    setMessage(error.message, "error");
  } finally {
    state.saveInFlight = false;
    updateCommandState();
  }
}

function updateSessionStatus(annotationStatus, draftRevision, latestId) {
  const session = state.workspace?.sessions.find(
    (item) => item.session_id === state.selectedSessionId,
  );
  if (session === undefined) return;
  session.annotation_status = latestId === null ? annotationStatus : "published";
  session.draft_revision = draftRevision;
  session.latest_published_revision_id = latestId;
}

async function loadWorkspace({ preserveSelection = true } = {}) {
  const previous = preserveSelection ? state.selectedSessionId : null;
  setMessage("");
  try {
    state.workspace = await readWorkspace();
    const skipped = state.workspace.skipped_session_count ?? 0;
    elements.sessionSummary.textContent = skipped === 0
      ? `${state.workspace.sessions.length} 个会话`
      : `${state.workspace.sessions.length} 个会话 · 跳过 ${skipped} 个旧格式`;
    elements.sessionEmpty.hidden = state.workspace.sessions.length !== 0;
    renderSessionList();
    const target = state.workspace.sessions.find((session) => session.session_id === previous)
      ?? state.workspace.sessions[0];
    if (target !== undefined) await loadSession(target.session_id);
    else resetEditor();
  } catch (error) {
    elements.sessionSummary.textContent = "数据平台不可用";
    setSaveState("数据平台未连接", "error");
    setMessage(error.message, "error");
  }
}

function renderSessionList() {
  elements.sessionList.replaceChildren();
  for (const session of state.workspace?.sessions ?? []) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "annotation-session-item";
    button.classList.toggle("is-active", session.session_id === state.selectedSessionId);
    const title = document.createElement("strong");
    title.textContent = session.display_name;
    const detail = document.createElement("span");
    detail.textContent = `${formatSessionTime(session.started_at_unix_ms)} · ${session.clips.length} 段视频`;
    const status = document.createElement("span");
    status.className = `annotation-status annotation-status-${session.annotation_status}`;
    status.textContent = {
      unannotated: "未标注",
      draft: `草稿 r${session.draft_revision}`,
      published: "已发布",
    }[session.annotation_status];
    button.append(title, detail, status);
    button.addEventListener("click", () => loadSession(session.session_id));
    elements.sessionList.append(button);
  }
}

async function loadSession(sessionId) {
  if (state.dirty) await saveDraft();
  setMessage("");
  setSaveState("正在载入会话", "saving");
  try {
    state.detail = await readAnnotationSession(sessionId);
    state.selectedSessionId = sessionId;
    state.selectedClipId = state.detail.session.clips[0]?.clip_id ?? null;
    state.selectedEpisodeId = null;
    state.proposalBatch = null;
    state.episodeMarkIn = null;
    state.phaseMarkIn = null;
    state.undoStack = [];
    state.redoStack = [];
    state.dirty = false;
    state.changeSerial = 0;
    elements.strategy.value = state.detail.draft.segmentation_strategy;
    renderSessionList();
    renderEditor();
    await loadClip();
    const revision = state.detail.draft.draft_revision;
    setSaveState(revision === 0 ? "尚未创建草稿" : `草稿已保存 · r${revision}`, "saved");
  } catch (error) {
    setSaveState("会话载入失败", "error");
    setMessage(error.message, "error");
  }
}

function resetEditor() {
  state.detail = null;
  state.selectedSessionId = null;
  state.selectedClipId = null;
  state.selectedEpisodeId = null;
  elements.activeSessionName.textContent = "请选择会话";
  elements.activeSessionDetail.textContent = "视频和草稿尚未载入";
  elements.clipSelect.replaceChildren();
  elements.clipSelect.disabled = true;
  elements.video.removeAttribute("src");
  elements.video.load();
  elements.videoEmpty.hidden = false;
  renderTimeline();
  renderInspector();
  updateCommandState();
}

function renderEditor() {
  if (state.detail === null) return resetEditor();
  const session = state.detail.session;
  elements.activeSessionName.textContent = session.display_name;
  elements.activeSessionDetail.textContent = `${session.clips.length} 段视频 · ${state.detail.draft.episodes.length} 个 Episode · ${session.editable ? "可编辑" : "采集中，只读"}`;
  elements.clipSelect.replaceChildren();
  for (const [index, clip] of session.clips.entries()) {
    const option = document.createElement("option");
    option.value = clip.clip_id;
    option.textContent = `片段 ${index + 1} · ${formatTime(clip.duration_ms / 1000)} · ${clip.frame_count} 帧`;
    elements.clipSelect.append(option);
  }
  elements.clipSelect.value = state.selectedClipId ?? "";
  elements.clipSelect.disabled = session.clips.length === 0;
  renderTimeline();
  renderInspector();
  renderMarkReadouts();
  updateStrategyControls();
  updateCommandState();
}

async function loadClip() {
  const clip = currentClip();
  elements.video.pause();
  if (clip === null) {
    elements.video.removeAttribute("src");
    elements.video.load();
    elements.videoEmpty.hidden = false;
    return;
  }
  try {
    elements.video.src = await annotationMediaUrl(clip.media_url);
    elements.video.load();
    elements.videoEmpty.hidden = true;
    updateTimecode();
  } catch (error) {
    elements.videoEmpty.hidden = false;
    setMessage(error.message, "error");
  }
}

function renderRuler(clip) {
  elements.ruler.replaceChildren();
  if (clip === null) return;
  for (let index = 0; index <= 5; index += 1) {
    const mark = document.createElement("span");
    mark.style.left = `${index * 20}%`;
    mark.textContent = formatTime((clip.duration_ms / 1000) * index / 5);
    elements.ruler.append(mark);
  }
}

function barPosition(start, end, frameCount) {
  return {
    left: `${start / frameCount * 100}%`,
    width: `${Math.max(0.2, (end - start) / frameCount * 100)}%`,
  };
}

function renderTimeline() {
  const clip = currentClip();
  elements.episodeTrack.replaceChildren();
  elements.phaseTrack.replaceChildren();
  renderRuler(clip);
  if (clip === null) {
    elements.playhead.hidden = true;
    return;
  }
  for (const proposal of state.proposalBatch?.proposals ?? []) {
    if (proposal.clip_id !== clip.clip_id) continue;
    const bar = document.createElement("span");
    bar.className = "timeline-proposal";
    Object.assign(
      bar.style,
      barPosition(proposal.start_frame_index, proposal.end_frame_index_exclusive, clip.frame_count),
    );
    bar.title = `${formatFrame(proposal.start_frame_index)}–${formatFrame(proposal.end_frame_index_exclusive)}`;
    elements.episodeTrack.append(bar);
  }
  const episodes = state.detail?.draft.episodes.filter(
    (episode) => episode.clip_id === clip.clip_id,
  ) ?? [];
  for (const [index, episode] of episodes.entries()) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "timeline-episode";
    button.classList.toggle("is-selected", episode.episode_id === state.selectedEpisodeId);
    button.textContent = `E${index + 1}`;
    button.title = `${episode.labels.instruction || "未标注任务"} · ${formatFrame(episode.start_frame_index)}–${formatFrame(episode.end_frame_index_exclusive)}`;
    Object.assign(
      button.style,
      barPosition(episode.start_frame_index, episode.end_frame_index_exclusive, clip.frame_count),
    );
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      selectEpisode(episode.episode_id);
    });
    elements.episodeTrack.append(button);
  }
  const selected = currentEpisode();
  for (const phase of selected?.phases ?? []) {
    const bar = document.createElement("button");
    bar.type = "button";
    bar.className = `timeline-phase timeline-phase-${phase.phase}`;
    bar.textContent = phaseLabel(phase.phase);
    bar.title = `${phaseLabel(phase.phase)} · ${formatFrame(phase.start_frame_index)}–${formatFrame(phase.end_frame_index_exclusive)}`;
    Object.assign(
      bar.style,
      barPosition(phase.start_frame_index, phase.end_frame_index_exclusive, clip.frame_count),
    );
    bar.addEventListener("click", (event) => {
      event.stopPropagation();
      seekFrame(phase.start_frame_index);
    });
    elements.phaseTrack.append(bar);
  }
  elements.playhead.hidden = false;
  updatePlayhead();
}

function phaseLabel(value) {
  return {
    prepare: "准备",
    approach: "接近",
    contact: "接触",
    manipulate: "操作",
    release: "释放",
    complete: "完成",
    other: "其他",
  }[value] ?? value;
}

function selectEpisode(episodeId) {
  state.selectedEpisodeId = episodeId;
  state.phaseMarkIn = null;
  const episode = currentEpisode();
  if (episode !== null && episode.clip_id !== state.selectedClipId) {
    state.selectedClipId = episode.clip_id;
    elements.clipSelect.value = episode.clip_id;
    loadClip();
  }
  renderTimeline();
  renderInspector();
  renderMarkReadouts();
  updateCommandState();
}

function renderInspector() {
  const episode = currentEpisode();
  elements.inspectorEmpty.hidden = episode !== null;
  elements.inspectorForm.hidden = episode === null;
  if (episode === null) {
    elements.inspectorSummary.textContent = "未选择 Episode";
    elements.phaseList.replaceChildren();
    return;
  }
  elements.inspectorSummary.textContent = `${formatFrame(episode.start_frame_index)}–${formatFrame(episode.end_frame_index_exclusive)}`;
  elements.taskId.value = episode.labels.task_id ?? "";
  elements.instruction.value = episode.labels.instruction;
  elements.verb.value = episode.labels.verb;
  elements.object.value = episode.labels.object;
  elements.target.value = episode.labels.target ?? "";
  elements.hand.value = episode.labels.hand;
  elements.outcome.value = episode.labels.outcome;
  elements.notes.value = episode.labels.notes;
  for (const checkbox of elements.qualityFlags) {
    checkbox.checked = episode.labels.quality_flags.includes(checkbox.value);
  }
  renderPhaseList(episode);
}

function renderPhaseList(episode) {
  elements.phaseList.replaceChildren();
  for (const phase of episode.phases) {
    const row = document.createElement("div");
    row.className = "phase-list-item";
    const label = document.createElement("button");
    label.type = "button";
    label.className = "phase-list-jump";
    label.textContent = `${phaseLabel(phase.phase)} · ${formatFrame(phase.start_frame_index)}–${formatFrame(phase.end_frame_index_exclusive)}`;
    label.addEventListener("click", () => seekFrame(phase.start_frame_index));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "phase-list-delete";
    remove.title = "删除阶段";
    remove.setAttribute("aria-label", "删除阶段");
    remove.textContent = "×";
    remove.addEventListener("click", () => {
      pushHistory();
      episode.phases = episode.phases.filter((item) => item.phase_id !== phase.phase_id);
      markChanged();
      renderTimeline();
      renderInspector();
    });
    row.append(label, remove);
    elements.phaseList.append(row);
  }
  if (episode.phases.length === 0) {
    const empty = document.createElement("span");
    empty.className = "phase-list-empty";
    empty.textContent = "尚未添加内部阶段";
    elements.phaseList.append(empty);
  }
}

function renderMarkReadouts() {
  elements.markReadout.textContent = state.episodeMarkIn === null
    ? "入点未设置"
    : `入点 ${formatFrame(state.episodeMarkIn)} · ${formatTime(frameToSeconds(state.episodeMarkIn))}`;
  elements.phaseMarkReadout.textContent = state.phaseMarkIn === null
    ? "阶段入点未设置"
    : `阶段入点 ${formatFrame(state.phaseMarkIn)}`;
}

function seekFrame(frame) {
  const clip = currentClip();
  if (clip === null) return;
  const bounded = Math.max(0, Math.min(clip.frame_count - 1, frame));
  elements.video.currentTime = bounded / clip.fps;
  updateTimecode();
}

function updateTimecode() {
  const frame = currentFrame();
  elements.timecode.textContent = `${formatTime(elements.video.currentTime)} · ${formatFrame(frame)}`;
  elements.play.textContent = elements.video.paused ? "▶" : "❚❚";
  updatePlayhead();
}

function updatePlayhead() {
  const clip = currentClip();
  if (clip === null || elements.playhead.hidden) return;
  const timelineRect = elements.timeline.getBoundingClientRect();
  const trackRect = elements.episodeTrack.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, currentFrame() / clip.frame_count));
  elements.playhead.style.left = `${trackRect.left - timelineRect.left + trackRect.width * ratio}px`;
}

function seekFromPointer(event) {
  const clip = currentClip();
  if (clip === null) return;
  const rect = event.currentTarget.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  seekFrame(Math.round(ratio * (clip.frame_count - 1)));
}

function createEmptyLabels() {
  return clone(state.detail?.draft.default_labels ?? {
    task_id: null,
    instruction: "",
    verb: "",
    object: "",
    target: null,
    hand: "unspecified",
    outcome: "unreviewed",
    quality_flags: [],
    notes: "",
  });
}

function intervalOverlaps(start, end, ignoredId = null) {
  return state.detail.draft.episodes.some(
    (episode) =>
      episode.clip_id === state.selectedClipId &&
      episode.episode_id !== ignoredId &&
      start < episode.end_frame_index_exclusive &&
      end > episode.start_frame_index,
  );
}

function addEpisode(start, end, sourceStrategy = "manual") {
  const clip = currentClip();
  if (clip === null || start >= end || end > clip.frame_count) {
    setMessage("Episode 边界无效", "error");
    return;
  }
  if (intervalOverlaps(start, end)) {
    setMessage("Episode 不能与当前视频中的已有 Episode 重叠", "error");
    return;
  }
  pushHistory();
  const episode = {
    episode_id: newId(),
    clip_id: clip.clip_id,
    start_frame_index: start,
    end_frame_index_exclusive: end,
    source_strategy: sourceStrategy,
    labels: createEmptyLabels(),
    phases: [],
  };
  state.detail.draft.episodes.push(episode);
  state.detail.draft.episodes.sort((left, right) =>
    left.clip_id.localeCompare(right.clip_id) || left.start_frame_index - right.start_frame_index
  );
  state.selectedEpisodeId = episode.episode_id;
  state.episodeMarkIn = null;
  markChanged();
  renderEditor();
  setMessage("");
}

function nextEpisode() {
  const selected = currentEpisode();
  if (selected === null) return null;
  const ordered = state.detail.draft.episodes
    .filter((episode) => episode.clip_id === selected.clip_id)
    .sort((left, right) => left.start_frame_index - right.start_frame_index);
  const index = ordered.findIndex((episode) => episode.episode_id === selected.episode_id);
  return index >= 0 ? ordered[index + 1] ?? null : null;
}

function splitSelectedEpisode() {
  const episode = currentEpisode();
  const frame = currentFrame();
  if (episode === null || frame <= episode.start_frame_index || frame >= episode.end_frame_index_exclusive) {
    setMessage("播放头必须位于所选 Episode 内部", "error");
    return;
  }
  if (episode.phases.some(
    (phase) => phase.start_frame_index < frame && phase.end_frame_index_exclusive > frame,
  )) {
    setMessage("播放头穿过一个内部阶段，请先调整或删除该阶段", "error");
    return;
  }
  pushHistory();
  const second = clone(episode);
  second.episode_id = newId();
  second.start_frame_index = frame;
  second.labels.outcome = "unreviewed";
  second.phases = second.phases.filter((phase) => phase.start_frame_index >= frame);
  episode.end_frame_index_exclusive = frame;
  episode.labels.outcome = "unreviewed";
  episode.phases = episode.phases.filter((phase) => phase.end_frame_index_exclusive <= frame);
  state.detail.draft.episodes.push(second);
  state.selectedEpisodeId = second.episode_id;
  markChanged();
  renderEditor();
  setMessage("");
}

function mergeSelectedWithNext() {
  const episode = currentEpisode();
  const following = nextEpisode();
  if (episode === null || following === null) return;
  pushHistory();
  episode.end_frame_index_exclusive = following.end_frame_index_exclusive;
  episode.labels.outcome = "unreviewed";
  episode.phases = [...episode.phases, ...following.phases].sort(
    (left, right) => left.start_frame_index - right.start_frame_index,
  );
  state.detail.draft.episodes = state.detail.draft.episodes.filter(
    (item) => item.episode_id !== following.episode_id,
  );
  markChanged();
  renderEditor();
}

function deleteSelectedEpisode() {
  const episode = currentEpisode();
  if (episode === null || !window.confirm("删除所选 Episode 及其阶段标注？")) return;
  pushHistory();
  state.detail.draft.episodes = state.detail.draft.episodes.filter(
    (item) => item.episode_id !== episode.episode_id,
  );
  state.selectedEpisodeId = null;
  markChanged();
  renderEditor();
}

function addPhase() {
  const episode = currentEpisode();
  const end = currentFrame();
  const start = state.phaseMarkIn;
  if (episode === null || start === null) return;
  if (start >= end || start < episode.start_frame_index || end > episode.end_frame_index_exclusive) {
    setMessage("阶段边界必须按顺序位于所选 Episode 内", "error");
    return;
  }
  if (episode.phases.some(
    (phase) => start < phase.end_frame_index_exclusive && end > phase.start_frame_index,
  )) {
    setMessage("内部阶段不能重叠", "error");
    return;
  }
  pushHistory();
  episode.phases.push({
    phase_id: newId(),
    start_frame_index: start,
    end_frame_index_exclusive: end,
    phase: elements.phaseKind.value,
    action_verb: elements.phaseVerb.value.trim() || null,
    active_hand: elements.phaseHand.value,
    object: elements.phaseObject.value.trim() || null,
  });
  episode.phases.sort((left, right) => left.start_frame_index - right.start_frame_index);
  state.phaseMarkIn = null;
  markChanged();
  renderTimeline();
  renderInspector();
  renderMarkReadouts();
  setMessage("");
}

function updateSelectedLabels() {
  const episode = currentEpisode();
  if (episode === null) return;
  pushHistory();
  episode.labels = {
    task_id: elements.taskId.value.trim() || null,
    instruction: elements.instruction.value.trim(),
    verb: elements.verb.value.trim(),
    object: elements.object.value.trim(),
    target: elements.target.value.trim() || null,
    hand: elements.hand.value,
    outcome: elements.outcome.value,
    quality_flags: elements.qualityFlags
      .filter((checkbox) => checkbox.checked)
      .map((checkbox) => checkbox.value),
    notes: elements.notes.value.trim(),
  };
  markChanged();
  renderTimeline();
}

function updateStrategyControls() {
  const fixed = elements.strategy.value === "fixed_window";
  for (const field of elements.fixedWindowFields) field.hidden = !fixed;
  updateCommandState();
}

async function generateProposals() {
  const clip = currentClip();
  if (clip === null || elements.strategy.value === "manual") return;
  setMessage("");
  elements.generateProposals.disabled = true;
  try {
    const request = {
      strategy: elements.strategy.value,
      clip_id: clip.clip_id,
    };
    if (elements.strategy.value === "fixed_window") {
      request.window_duration_ms = Math.round(Number(elements.windowDuration.value) * 1000);
      request.stride_duration_ms = Math.round(Number(elements.windowStride.value) * 1000);
      if (request.stride_duration_ms < request.window_duration_ms) {
        throw new Error("固定窗口的步长不能小于窗口长度，正式 Episode 不允许重叠");
      }
    }
    state.proposalBatch = await requestEpisodeProposals(state.selectedSessionId, request);
    elements.acceptProposals.hidden = state.proposalBatch.proposals.length === 0;
    renderTimeline();
    setMessage(`已生成 ${state.proposalBatch.proposals.length} 个候选，尚未写入草稿`, "info");
  } catch (error) {
    setMessage(error.message, "error");
  } finally {
    updateCommandState();
  }
}

function acceptProposals() {
  const clip = currentClip();
  if (clip === null || state.proposalBatch === null) return;
  const existing = state.detail.draft.episodes.filter((episode) => episode.clip_id === clip.clip_id);
  if (
    existing.length > 0 &&
    !window.confirm("接受候选将替换当前视频已有的 Episode，是否继续？")
  ) return;
  pushHistory();
  state.detail.draft.episodes = state.detail.draft.episodes.filter(
    (episode) => episode.clip_id !== clip.clip_id,
  );
  for (const proposal of state.proposalBatch.proposals) {
    state.detail.draft.episodes.push({
      episode_id: newId(),
      clip_id: proposal.clip_id,
      start_frame_index: proposal.start_frame_index,
      end_frame_index_exclusive: proposal.end_frame_index_exclusive,
      source_strategy: state.proposalBatch.strategy,
      labels: createEmptyLabels(),
      phases: [],
    });
  }
  state.detail.draft.segmentation_strategy = state.proposalBatch.strategy;
  state.selectedEpisodeId = state.detail.draft.episodes.find(
    (episode) => episode.clip_id === clip.clip_id,
  )?.episode_id ?? null;
  state.proposalBatch = null;
  elements.acceptProposals.hidden = true;
  markChanged();
  renderEditor();
  setMessage("候选已进入草稿，请逐段检查边界和标签", "info");
}

async function publishCurrentDraft() {
  if (state.detail === null) return;
  clearTimeout(state.saveTimer);
  if (state.dirty) await saveDraft();
  if (state.dirty) return;
  elements.publish.disabled = true;
  setMessage("");
  try {
    const revision = await publishAnnotation(
      state.selectedSessionId,
      state.detail.draft.draft_revision,
    );
    state.detail.draft.latest_published_revision_id = revision.annotation_revision_id;
    updateSessionStatus("published", state.detail.draft.draft_revision, revision.annotation_revision_id);
    renderSessionList();
    setSaveState(`已发布 · ${revision.annotation_revision_id.slice(0, 8)}`, "published");
    setMessage(
      `发布完成：${revision.quality.episode_count} 个 Episode，${revision.quality.phase_count} 个阶段`,
      "success",
    );
  } catch (error) {
    setMessage(error.message, "error");
  } finally {
    updateCommandState();
  }
}

elements.strategy.addEventListener("change", () => {
  state.proposalBatch = null;
  elements.acceptProposals.hidden = true;
  updateStrategyControls();
  renderTimeline();
});
elements.generateProposals.addEventListener("click", generateProposals);
elements.acceptProposals.addEventListener("click", acceptProposals);
elements.refresh.addEventListener("click", () => loadWorkspace());
elements.publish.addEventListener("click", publishCurrentDraft);
elements.clipSelect.addEventListener("change", async () => {
  state.selectedClipId = elements.clipSelect.value;
  state.selectedEpisodeId = null;
  state.proposalBatch = null;
  state.episodeMarkIn = null;
  state.phaseMarkIn = null;
  elements.acceptProposals.hidden = true;
  await loadClip();
  renderEditor();
});
elements.video.addEventListener("timeupdate", updateTimecode);
elements.video.addEventListener("play", updateTimecode);
elements.video.addEventListener("pause", updateTimecode);
elements.video.addEventListener("ended", updateTimecode);
elements.video.addEventListener("loadedmetadata", updateTimecode);
elements.play.addEventListener("click", () => {
  if (elements.video.paused) elements.video.play().catch((error) => setMessage(error.message, "error"));
  else elements.video.pause();
});
elements.previousFrame.addEventListener("click", () => seekFrame(currentFrame() - 1));
elements.nextFrame.addEventListener("click", () => seekFrame(currentFrame() + 1));
elements.jumpStart.addEventListener("click", () => seekFrame(0));
elements.jumpEnd.addEventListener("click", () => seekFrame((currentClip()?.frame_count ?? 1) - 1));
elements.playbackRate.addEventListener("change", () => {
  elements.video.playbackRate = Number(elements.playbackRate.value);
});
for (const target of [elements.ruler, elements.episodeTrack, elements.phaseTrack]) {
  target.addEventListener("click", seekFromPointer);
}
elements.setEpisodeIn.addEventListener("click", () => {
  state.episodeMarkIn = currentFrame();
  renderMarkReadouts();
  updateCommandState();
});
elements.createEpisode.addEventListener("click", () => addEpisode(state.episodeMarkIn, currentFrame()));
elements.splitEpisode.addEventListener("click", splitSelectedEpisode);
elements.mergeEpisode.addEventListener("click", mergeSelectedWithNext);
elements.deleteEpisode.addEventListener("click", deleteSelectedEpisode);
elements.setPhaseIn.addEventListener("click", () => {
  const episode = currentEpisode();
  const frame = currentFrame();
  if (episode === null || frame < episode.start_frame_index || frame >= episode.end_frame_index_exclusive) {
    setMessage("阶段入点必须位于所选 Episode 内", "error");
    return;
  }
  state.phaseMarkIn = frame;
  renderMarkReadouts();
});
elements.createPhase.addEventListener("click", addPhase);
for (const control of [
  elements.taskId,
  elements.instruction,
  elements.verb,
  elements.object,
  elements.target,
  elements.hand,
  elements.outcome,
  elements.notes,
  ...elements.qualityFlags,
]) {
  control.addEventListener("change", updateSelectedLabels);
}
elements.undo.addEventListener("click", () => {
  if (state.undoStack.length === 0 || state.detail === null) return;
  state.redoStack.push(snapshotDraft());
  restoreSnapshot(state.undoStack.pop());
});
elements.redo.addEventListener("click", () => {
  if (state.redoStack.length === 0 || state.detail === null) return;
  state.undoStack.push(snapshotDraft());
  restoreSnapshot(state.redoStack.pop());
});
window.addEventListener("resize", updatePlayhead);
window.addEventListener("beforeunload", (event) => {
  if (!state.dirty) return;
  event.preventDefault();
  event.returnValue = "";
});

loadWorkspace({ preserveSelection: false });
