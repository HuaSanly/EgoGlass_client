from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import statistics
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .annotation_models import (
    AnnotationDraft,
    AnnotationQuality,
    AnnotationQualityCheck,
    AnnotationSession,
    AnnotationSessionSummary,
    AnnotationWorkspace,
    ClipSummary,
    EpisodeDraft,
    EpisodeOutcome,
    EpisodeProposal,
    FrameBoundary,
    HandLabel,
    ProposalBatch,
    ProposalRequest,
    PublishedEpisode,
    PublishedRevision,
    SaveDraftRequest,
)


class AnnotationNotFoundError(RuntimeError):
    pass


class RevisionConflictError(RuntimeError):
    pass


class AnnotationReadOnlyError(RuntimeError):
    pass


class AnnotationValidationError(RuntimeError):
    def __init__(self, issues: list[str]) -> None:
        super().__init__("annotation validation failed")
        self.issues = issues


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as target:
            json.dump(value, target, ensure_ascii=False, indent=2)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class AnnotationStore:
    def __init__(
        self,
        recordings_root: Path,
        *,
        clock_ns: Callable[[], int] = time.time_ns,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self.recordings_root = recordings_root.resolve()
        self._clock_ns = clock_ns
        self._id_factory = id_factory
        self._lock = threading.RLock()

    def workspace(self) -> AnnotationWorkspace:
        sessions: list[AnnotationSessionSummary] = []
        skipped: Counter[str] = Counter()
        directories = self.recordings_root.iterdir() if self.recordings_root.is_dir() else ()
        for directory in directories:
            if not directory.is_dir() or not _is_id(directory.name):
                continue
            try:
                sessions.append(self._session_summary(directory.name))
            except AnnotationNotFoundError:
                skipped["unsupported_session_contract"] += 1
            except (OSError, ValueError, KeyError, json.JSONDecodeError, sqlite3.Error):
                skipped["unreadable_session"] += 1
        sessions.sort(key=lambda item: item.started_at_unix_ms, reverse=True)
        return AnnotationWorkspace(
            implemented_strategies=["manual", "clip_as_episode", "fixed_window"],
            planned_strategies=[
                "event_marker",
                "motion_change",
                "hand_object_interaction",
                "vlm_semantic",
            ],
            skipped_session_count=sum(skipped.values()),
            skipped_session_reasons=dict(sorted(skipped.items())),
            sessions=sessions,
        )

    def session(self, session_id: str) -> AnnotationSession:
        summary = self._session_summary(session_id)
        return AnnotationSession(session=summary, draft=self._read_draft(session_id))

    def save_draft(self, session_id: str, request: SaveDraftRequest) -> AnnotationDraft:
        with self._lock:
            summary = self._session_summary(session_id)
            if not summary.editable:
                raise AnnotationReadOnlyError("active or finalizing sessions cannot be annotated")
            current = self._read_draft(session_id)
            if current.draft_revision != request.base_revision:
                raise RevisionConflictError(
                    f"draft revision changed from {request.base_revision} "
                    f"to {current.draft_revision}"
                )
            self._validate_structure(summary, request.episodes)
            draft = AnnotationDraft(
                session_id=session_id,
                draft_revision=current.draft_revision + 1,
                segmentation_strategy=request.segmentation_strategy,
                default_labels=request.default_labels,
                episodes=request.episodes,
                updated_at_unix_ns=self._clock_ns(),
                latest_published_revision_id=current.latest_published_revision_id,
            )
            _write_json_atomic(self._draft_path(session_id), draft.model_dump(mode="json"))
            return draft

    def proposals(self, session_id: str, request: ProposalRequest) -> ProposalBatch:
        summary = self._session_summary(session_id)
        clips = summary.clips
        if request.clip_id is not None:
            clips = [clip for clip in clips if clip.clip_id == request.clip_id]
            if not clips:
                raise AnnotationNotFoundError("annotation clip not found")

        proposals: list[EpisodeProposal] = []
        if request.strategy == "clip_as_episode":
            for clip in clips:
                proposals.append(
                    EpisodeProposal(
                        proposal_id=self._id_factory(),
                        clip_id=clip.clip_id,
                        start_frame_index=0,
                        end_frame_index_exclusive=clip.frame_count,
                        evidence=["complete source clip selected as one task-attempt candidate"],
                    )
                )
        else:
            for clip in clips:
                window_frames = max(1, round(request.window_duration_ms * clip.fps / 1000))
                stride_frames = max(1, round(request.stride_duration_ms * clip.fps / 1000))
                starts: Iterable[int] = range(0, clip.frame_count, stride_frames)
                for start in starts:
                    end = min(clip.frame_count, start + window_frames)
                    proposals.append(
                        EpisodeProposal(
                            proposal_id=self._id_factory(),
                            clip_id=clip.clip_id,
                            start_frame_index=start,
                            end_frame_index_exclusive=end,
                            evidence=[
                                f"deterministic {request.window_duration_ms} ms window "
                                f"at nominal {clip.fps:g} FPS"
                            ],
                        )
                    )
        return ProposalBatch(
            proposal_batch_id=self._id_factory(),
            session_id=session_id,
            strategy=request.strategy,
            generated_at_unix_ns=self._clock_ns(),
            config=request.model_dump(mode="json", exclude={"strategy"}),
            proposals=proposals,
        )

    def publish(self, session_id: str, base_revision: int) -> PublishedRevision:
        with self._lock:
            summary = self._session_summary(session_id)
            if not summary.editable:
                raise AnnotationReadOnlyError("active or finalizing sessions cannot be published")
            draft = self._read_draft(session_id)
            if draft.draft_revision != base_revision:
                raise RevisionConflictError(
                    f"draft revision changed from {base_revision} to {draft.draft_revision}"
                )
            checks = self._validate_publish(summary, draft.episodes)
            manifest = self._capture_manifest(session_id)
            episodes = [
                self._published_episode(session_id, manifest, episode) for episode in draft.episodes
            ]
            content = {
                "session_id": session_id,
                "taxonomy_version": draft.taxonomy_version,
                "segmentation_strategy": draft.segmentation_strategy,
                "episodes": [episode.model_dump(mode="json") for episode in episodes],
            }
            canonical = json.dumps(
                content,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            content_sha256 = hashlib.sha256(canonical).hexdigest()
            revision_id = content_sha256[:32]
            revision_path = self._revisions_path(session_id) / f"{revision_id}.json"
            if revision_path.is_file():
                revision = PublishedRevision.model_validate(_read_json(revision_path))
            else:
                revision = PublishedRevision(
                    annotation_revision_id=revision_id,
                    parent_draft_revision=draft.draft_revision,
                    session_id=session_id,
                    segmentation_strategy=draft.segmentation_strategy,
                    published_at_unix_ns=self._clock_ns(),
                    content_sha256=content_sha256,
                    episodes=episodes,
                    quality=AnnotationQuality(
                        episode_count=len(episodes),
                        phase_count=sum(len(episode.phases) for episode in episodes),
                        checks=checks,
                    ),
                )
                _write_json_atomic(revision_path, revision.model_dump(mode="json"))
            _write_json_atomic(
                self._latest_path(session_id),
                {
                    "schema_version": "1.0",
                    "annotation_revision_id": revision.annotation_revision_id,
                    "content_sha256": revision.content_sha256,
                },
            )
            updated_draft = draft.model_copy(
                update={"latest_published_revision_id": revision.annotation_revision_id}
            )
            _write_json_atomic(
                self._draft_path(session_id), updated_draft.model_dump(mode="json")
            )
            return revision

    def media_path(self, session_id: str, clip_id: str) -> Path:
        summary = self._session_summary(session_id)
        if clip_id not in {clip.clip_id for clip in summary.clips}:
            raise AnnotationNotFoundError("annotation clip not found")
        manifest = self._capture_manifest(session_id)
        for clip in manifest["clips"]:
            if clip["clip_id"] != clip_id:
                continue
            path = (self._session_directory(session_id) / clip["relative_media_path"]).resolve()
            if not path.is_relative_to(self._session_directory(session_id)) or not path.is_file():
                break
            return path
        raise AnnotationNotFoundError("annotation media not found")

    def _session_summary(self, session_id: str) -> AnnotationSessionSummary:
        manifest = self._capture_manifest(session_id)
        lifecycle = manifest["lifecycle"]
        state = lifecycle["state"]
        clips: list[ClipSummary] = []
        database_path = self._session_directory(session_id) / "telemetry" / "telemetry.sqlite"
        exact_clip_ids = self._indexed_clip_ids(database_path)
        for raw_clip in manifest["clips"]:
            if raw_clip["state"] not in {"complete", "incomplete"}:
                continue
            frame_count = raw_clip.get("frame_count")
            if not isinstance(frame_count, int) or frame_count < 1:
                continue
            relative_media_path = raw_clip["relative_media_path"]
            media_path = (self._session_directory(session_id) / relative_media_path).resolve()
            if (
                not media_path.is_relative_to(self._session_directory(session_id))
                or not media_path.is_file()
            ):
                continue
            profile = raw_clip["video_profile"]
            fps = float(profile["nominal_fps"])
            started_ns = raw_clip.get("started_at_session_time_ns")
            ended_ns = raw_clip.get("ended_at_session_time_ns")
            if isinstance(started_ns, int) and isinstance(ended_ns, int) and ended_ns >= started_ns:
                duration_ms = round((ended_ns - started_ns) / 1_000_000)
            else:
                duration_ms = round(frame_count * 1000 / fps)
            clip_id = raw_clip["clip_id"]
            clips.append(
                ClipSummary(
                    clip_id=clip_id,
                    state=raw_clip["state"],
                    duration_ms=duration_ms,
                    frame_count=frame_count,
                    fps=fps,
                    width=profile["width"],
                    height=profile["height"],
                    media_url=f"/api/v1/annotations/media/{session_id}/{clip_id}",
                    exact_frame_index_available=clip_id in exact_clip_ids,
                )
            )
        draft = self._read_draft(session_id)
        annotation_status = (
            "published"
            if draft.latest_published_revision_id is not None
            else ("draft" if draft.draft_revision > 0 else "unannotated")
        )
        return AnnotationSessionSummary(
            session_id=session_id,
            display_name=manifest["display_name"],
            state=state,
            started_at_unix_ms=lifecycle["started_at_unix_ns"] // 1_000_000,
            editable=state in {"complete", "incomplete"},
            annotation_status=annotation_status,
            draft_revision=draft.draft_revision,
            latest_published_revision_id=draft.latest_published_revision_id,
            clips=clips,
        )

    def _read_draft(self, session_id: str) -> AnnotationDraft:
        path = self._draft_path(session_id)
        latest_id = self._latest_revision_id(session_id)
        if not path.is_file():
            return AnnotationDraft(
                session_id=session_id,
                draft_revision=0,
                latest_published_revision_id=latest_id,
            )
        draft = AnnotationDraft.model_validate(_read_json(path))
        if draft.session_id != session_id:
            raise ValueError("annotation draft belongs to another session")
        if draft.latest_published_revision_id != latest_id:
            draft = draft.model_copy(update={"latest_published_revision_id": latest_id})
        return draft

    def _validate_structure(
        self, summary: AnnotationSessionSummary, episodes: list[EpisodeDraft]
    ) -> None:
        clip_counts = {clip.clip_id: clip.frame_count for clip in summary.clips}
        issues: list[str] = []
        episode_ids: set[str] = set()
        phase_ids: set[str] = set()
        by_clip: dict[str, list[EpisodeDraft]] = {}
        for episode in episodes:
            if episode.episode_id in episode_ids:
                issues.append(f"重复 episode_id：{episode.episode_id}")
            episode_ids.add(episode.episode_id)
            frame_count = clip_counts.get(episode.clip_id)
            if frame_count is None:
                issues.append(f"episode 引用了未知视频：{episode.clip_id}")
                continue
            if episode.end_frame_index_exclusive > frame_count:
                issues.append(f"episode {episode.episode_id} 超出视频帧范围")
            by_clip.setdefault(episode.clip_id, []).append(episode)
            for phase in episode.phases:
                if phase.phase_id in phase_ids:
                    issues.append(f"重复 phase_id：{phase.phase_id}")
                phase_ids.add(phase.phase_id)
                if not (
                    episode.start_frame_index
                    <= phase.start_frame_index
                    < phase.end_frame_index_exclusive
                    <= episode.end_frame_index_exclusive
                ):
                    issues.append(f"阶段 {phase.phase_id} 没有完全位于 episode 内")
            ordered_phases = sorted(
                episode.phases, key=lambda phase: phase.start_frame_index
            )
            if any(
                previous.end_frame_index_exclusive > current.start_frame_index
                for previous, current in zip(
                    ordered_phases, ordered_phases[1:], strict=False
                )
            ):
                issues.append(f"episode {episode.episode_id} 内存在重叠阶段")
        for clip_id, clip_episodes in by_clip.items():
            ordered = sorted(clip_episodes, key=lambda episode: episode.start_frame_index)
            if any(
                previous.end_frame_index_exclusive > current.start_frame_index
                for previous, current in zip(ordered, ordered[1:], strict=False)
            ):
                issues.append(f"视频 {clip_id} 中存在重叠 episode")
        if issues:
            raise AnnotationValidationError(issues)

    def _validate_publish(
        self, summary: AnnotationSessionSummary, episodes: list[EpisodeDraft]
    ) -> list[AnnotationQualityCheck]:
        self._validate_structure(summary, episodes)
        issues: list[str] = []
        warnings: list[str] = []
        if not episodes:
            issues.append("至少需要一个 episode")
        clip_fps = {clip.clip_id: clip.fps for clip in summary.clips}
        for index, episode in enumerate(episodes, start=1):
            prefix = f"episode {index}"
            if not episode.labels.instruction:
                issues.append(f"{prefix} 缺少任务描述")
            if not episode.labels.verb:
                issues.append(f"{prefix} 缺少动作动词")
            if not episode.labels.object:
                issues.append(f"{prefix} 缺少操作对象")
            if episode.labels.hand == HandLabel.UNSPECIFIED:
                issues.append(f"{prefix} 未选择参与手")
            if episode.labels.outcome == EpisodeOutcome.UNREVIEWED:
                issues.append(f"{prefix} 未标注任务结果")
            if not episode.phases:
                issues.append(f"{prefix} 至少需要一个内部阶段")
            fps = clip_fps.get(episode.clip_id, 30.0)
            duration_seconds = (
                episode.end_frame_index_exclusive - episode.start_frame_index
            ) / fps
            if duration_seconds < 1:
                warnings.append(f"{prefix} 短于 1 秒")
            if duration_seconds > 120:
                warnings.append(f"{prefix} 长于 120 秒")
        if issues:
            raise AnnotationValidationError(issues)
        checks = [
            AnnotationQualityCheck(
                check_id="episode_bounds",
                status="pass",
                evidence="all episode intervals are ordered, non-overlapping, and clip-local",
            ),
            AnnotationQualityCheck(
                check_id="phase_containment",
                status="pass",
                evidence="all phase intervals are ordered and contained by their episode",
            ),
            AnnotationQualityCheck(
                check_id="required_labels",
                status="pass",
                evidence="all episodes include instruction, verb, object, hand, and outcome",
            ),
        ]
        if warnings:
            checks.append(
                AnnotationQualityCheck(
                    check_id="duration_review",
                    status="warn",
                    evidence="; ".join(warnings),
                )
            )
        return checks

    def _published_episode(
        self,
        session_id: str,
        manifest: dict[str, Any],
        episode: EpisodeDraft,
    ) -> PublishedEpisode:
        clip = next(item for item in manifest["clips"] if item["clip_id"] == episode.clip_id)
        timing_rows = self._timing_rows(session_id, episode.clip_id)
        fps = float(clip["video_profile"]["nominal_fps"])
        return PublishedEpisode(
            episode_id=episode.episode_id,
            clip_id=episode.clip_id,
            start=self._resolve_boundary(
                episode.start_frame_index,
                clip,
                timing_rows,
                fps,
                is_end=False,
            ),
            end_exclusive=self._resolve_boundary(
                episode.end_frame_index_exclusive,
                clip,
                timing_rows,
                fps,
                is_end=True,
            ),
            source_strategy=episode.source_strategy,
            labels=episode.labels,
            phases=episode.phases,
        )

    def _resolve_boundary(
        self,
        frame_index: int,
        clip: dict[str, Any],
        rows: dict[int, tuple[int, int, int, int | None]],
        fps: float,
        *,
        is_end: bool,
    ) -> FrameBoundary:
        row = rows.get(frame_index)
        if row is not None:
            return FrameBoundary(
                frame_index=frame_index,
                mp4_pts=row[0],
                mp4_time_base_numerator=row[1],
                mp4_time_base_denominator=row[2],
                session_time_ns=row[3],
                timing_status="exact" if row[3] is not None else "unmapped",
            )
        if rows:
            sorted_indices = sorted(rows)
            last_index = sorted_indices[-1]
            last = rows[last_index]
            deltas = [
                rows[current][0] - rows[previous][0]
                for previous, current in zip(
                    sorted_indices, sorted_indices[1:], strict=False
                )
                if rows[current][0] > rows[previous][0]
            ]
            pts_delta = round(statistics.median(deltas)) if deltas else round(last[2] / fps)
            estimated_pts = last[0] + (frame_index - last_index) * pts_delta
            session_time_ns = None
            if is_end and frame_index == clip.get("frame_count"):
                session_time_ns = clip.get("ended_at_session_time_ns")
            return FrameBoundary(
                frame_index=frame_index,
                mp4_pts=estimated_pts,
                mp4_time_base_numerator=last[1],
                mp4_time_base_denominator=last[2],
                session_time_ns=session_time_ns,
                timing_status="estimated",
            )
        started_ns = clip.get("started_at_session_time_ns")
        session_time_ns = (
            None
            if not isinstance(started_ns, int)
            else started_ns + round(frame_index * 1_000_000_000 / fps)
        )
        return FrameBoundary(
            frame_index=frame_index,
            mp4_pts=round(frame_index * 90000 / fps),
            mp4_time_base_numerator=1,
            mp4_time_base_denominator=90000,
            session_time_ns=session_time_ns,
            timing_status="estimated",
        )

    def _timing_rows(
        self, session_id: str, clip_id: str
    ) -> dict[int, tuple[int, int, int, int | None]]:
        path = self._session_directory(session_id) / "telemetry" / "telemetry.sqlite"
        if not path.is_file():
            return {}
        try:
            with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as connection:
                rows = connection.execute(
                    """
                    SELECT frame_index, mp4_pts, mp4_time_base_numerator,
                           mp4_time_base_denominator, session_time_ns
                    FROM video_frame_index WHERE clip_id = ? ORDER BY frame_index
                    """,
                    (clip_id,),
                ).fetchall()
        except sqlite3.Error:
            return {}
        return {
            int(frame_index): (int(pts), int(numerator), int(denominator), session_time)
            for frame_index, pts, numerator, denominator, session_time in rows
        }

    def _indexed_clip_ids(self, path: Path) -> set[str]:
        if not path.is_file():
            return set()
        try:
            with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as connection:
                return {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT DISTINCT clip_id FROM video_frame_index"
                    ).fetchall()
                }
        except sqlite3.Error:
            return set()

    def _capture_manifest(self, session_id: str) -> dict[str, Any]:
        if not _is_id(session_id):
            raise AnnotationNotFoundError("annotation session not found")
        path = self._session_directory(session_id) / "session.json"
        if not path.is_file():
            raise AnnotationNotFoundError("annotation session not found")
        value = _read_json(path)
        if (
            value.get("contract_id") != "capture-session-v1"
            or value.get("session_id") != session_id
        ):
            raise AnnotationNotFoundError("annotation session is not a capture-session-v1 session")
        return value

    def _latest_revision_id(self, session_id: str) -> str | None:
        path = self._latest_path(session_id)
        if not path.is_file():
            return None
        value = _read_json(path).get("annotation_revision_id")
        return value if isinstance(value, str) and _is_id(value) else None

    def _session_directory(self, session_id: str) -> Path:
        directory = (self.recordings_root / session_id).resolve()
        if not directory.is_relative_to(self.recordings_root):
            raise AnnotationNotFoundError("annotation session not found")
        return directory

    def _annotation_directory(self, session_id: str) -> Path:
        return self._session_directory(session_id) / "annotations" / "episode-annotation-v1"

    def _draft_path(self, session_id: str) -> Path:
        return self._annotation_directory(session_id) / "draft.json"

    def _latest_path(self, session_id: str) -> Path:
        return self._annotation_directory(session_id) / "latest.json"

    def _revisions_path(self, session_id: str) -> Path:
        path = self._annotation_directory(session_id) / "revisions"
        path.mkdir(parents=True, exist_ok=True)
        return path


def _is_id(value: str) -> bool:
    return len(value) == 32 and all(character in "0123456789abcdef" for character in value)
