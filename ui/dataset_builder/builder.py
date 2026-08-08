from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path

from schemas import (
    ArtifactReference,
    DatasetEpisode,
    DatasetFrame,
    DatasetManifest,
    DatasetSplit,
    HandSample,
    MotionPhase,
    ObjectMaskObservation,
    ObjectObservation,
    ObjectPose,
    PhaseAnalysisResult,
    SensorSampleReference,
    VirtualVideoSpan,
)
from sensor_preprocessing import CaptureSessionReader
from ui.annotation.models import PublishedRevision
from ui.processing.results import ProcessingResultStore

from .episodes import EpisodeInterval, split_valid_intervals
from .models import DatasetBuildResult, DatasetCandidate, DatasetQualityReport
from .quality import DatasetQualityChecker


class DatasetBuildError(RuntimeError):
    """A candidate cannot be assembled or an immutable dataset already exists."""


class DatasetBuilder:
    """Assemble final offline runs into a reproducible, reference-only dataset."""

    def __init__(
        self,
        recordings_root: str | Path,
        *,
        random_seed: int = 20260808,
        minimum_episode_frames: int = 10,
        quality_checker: DatasetQualityChecker | None = None,
    ) -> None:
        self.recordings_root = Path(recordings_root).expanduser().resolve()
        self.random_seed = random_seed
        self.minimum_episode_frames = minimum_episode_frames
        self.quality_checker = quality_checker or DatasetQualityChecker()

    def candidate(
        self,
        session_id: str,
        processing_run_id: str,
        *,
        annotation_revision_id: str,
        manual_intervals: Mapping[str, Iterable[EpisodeInterval]] | None = None,
        quality_report: DatasetQualityReport | None = None,
    ) -> DatasetCandidate:
        session_path = self._session_path(session_id)
        run_directory = self._run_path(session_path, processing_run_id)
        annotation_revision = _load_annotation_revision(
            session_path,
            annotation_revision_id,
        )
        checked_quality = self.quality_checker.check(session_path, run_directory)
        quality = _validated_quality_report(checked_quality, quality_report)
        try:
            reader = CaptureSessionReader.open(session_path, verify_media_hashes=True)
            store = ProcessingResultStore(run_directory / "results.sqlite", read_only=True)
            rows = store.iter_results()
        except (OSError, ValueError, RuntimeError) as error:
            raise DatasetBuildError(str(error)) from error
        rows_by_clip: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            clip_id = row.get("sequence_id")
            if isinstance(clip_id, str):
                rows_by_clip.setdefault(clip_id, []).append(row)
        object_result = _read_json(run_directory / "objects" / "object-result.json") or {}
        phase_result = _read_json(run_directory / "phase-analysis.json") or {}
        episodes: list[DatasetEpisode] = []
        requested = manual_intervals or {}
        for clip in reader.session.clips:
            clip_rows = sorted(rows_by_clip.get(clip.clip_id, ()), key=_frame_key)
            if not clip_rows:
                continue
            ranges = tuple(requested.get(clip.clip_id, ()))
            if ranges:
                for interval in ranges:
                    if interval.clip_id != clip.clip_id:
                        raise DatasetBuildError("manual episode clip does not match selected clip")
                    if (
                        interval.start_frame_index < 0
                        or interval.end_frame_index_exclusive > clip.frame_count
                    ):
                        raise DatasetBuildError("manual episode exceeds clip bounds")
                selected_ranges = ranges
            else:
                selected_ranges = split_valid_intervals(
                    clip.clip_id,
                    tuple(int(row["frame_index"]) for row in clip_rows),
                    tuple(issue for issue in quality.soft_issues if issue.clip_id == clip.clip_id),
                    minimum_frames=self.minimum_episode_frames,
                )
            for interval in selected_ranges:
                episode_rows = tuple(
                    row
                    for row in clip_rows
                    if interval.start_frame_index
                    <= int(row["frame_index"])
                    < interval.end_frame_index_exclusive
                )
                if len(episode_rows) < self.minimum_episode_frames:
                    continue
                episodes.append(
                    self._episode(
                        session_id,
                        processing_run_id,
                        annotation_revision_id,
                        clip,
                        interval,
                        episode_rows,
                        phase_result,
                        object_result,
                        quality,
                        annotation_revision,
                    )
                )
        return DatasetCandidate(
            session_id=session_id,
            run_id=processing_run_id,
            session_path=session_path,
            run_directory=run_directory,
            quality=quality,
            episodes=tuple(episodes),
            annotation_revision_id=annotation_revision_id,
        )

    def publish(
        self,
        dataset_id: str,
        candidates: Iterable[DatasetCandidate],
        *,
        output_directory: str | Path | None = None,
    ) -> DatasetBuildResult:
        _validate_id(dataset_id)
        values = tuple(candidates)
        if not values:
            raise DatasetBuildError("at least one dataset candidate is required")
        if any(not value.publishable for value in values):
            raise DatasetBuildError("dataset contains a hard quality failure or no valid episodes")
        session_ids = tuple(sorted({value.session_id for value in values}))
        target = (
            Path(output_directory).expanduser().resolve()
            if output_directory
            else self.recordings_root / "datasets" / dataset_id
        )
        if target.exists():
            raise DatasetBuildError(f"dataset version already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        staging.mkdir(parents=False, exist_ok=False)
        episodes = tuple(episode for candidate in values for episode in candidate.episodes)
        samples = tuple(
            frame for episode in episodes for frame in _frames_for_episode(episode, values)
        )
        episode_path = staging / "episodes.jsonl"
        sample_path = staging / "samples.jsonl"
        quality_path = staging / "quality-report.json"
        provenance_path = staging / "provenance.json"
        splits_path = staging / "splits.json"
        try:
            _write_jsonl(episode_path, (episode.model_dump(mode="json") for episode in episodes))
            _write_jsonl(sample_path, (frame.model_dump(mode="json") for frame in samples))
            quality_payload = {
                "schema_version": "1.0",
                "hard_issues": [
                    issue.model_dump(mode="json")
                    for candidate in values
                    for issue in candidate.quality.hard_issues
                ],
                "soft_issues": [
                    issue.model_dump(mode="json")
                    for candidate in values
                    for issue in candidate.quality.soft_issues
                ],
                "metrics": {
                    key: value for candidate in values for key, value in candidate.quality.metrics
                },
            }
            _write_json(quality_path, quality_payload)
            _write_json(
                provenance_path,
                {
                    "schema_version": "1.0",
                    "sessions": [_provenance(candidate) for candidate in values],
                },
            )
            splits = _splits(session_ids, episodes, self.random_seed)
            _write_json(
                splits_path,
                [split.model_dump(mode="json") for split in splits],
            )
            manifest = DatasetManifest(
                dataset_id=dataset_id,
                created_at_unix_ns=time.time_ns(),
                random_seed=self.random_seed,
                episodes_artifact=_artifact(staging, episode_path),
                samples_artifact=_artifact(staging, sample_path),
                quality_report_artifact=_artifact(staging, quality_path),
                provenance_artifact=_artifact(staging, provenance_path),
                splits_artifact=_artifact(staging, splits_path),
                splits=splits,
                source_session_ids=session_ids,
            )
            _write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
            os.replace(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return DatasetBuildResult(dataset_id, target, manifest, len(episodes), len(samples))

    def _episode(
        self,
        session_id: str,
        run_id: str,
        annotation_revision_id: str,
        clip: object,
        interval: EpisodeInterval,
        rows: tuple[dict[str, object], ...],
        phase_payload: dict[str, object],
        _object_payload: dict[str, object],
        quality: object,
        annotation_revision: PublishedRevision,
    ) -> DatasetEpisode:
        first = rows[0]
        last = rows[-1]
        clip_ref = clip
        media = ArtifactReference(
            relative_path=Path(clip_ref.media_path)
            .relative_to(self._session_path(session_id))
            .as_posix(),
            sha256=clip_ref.sha256,
            media_type="video/mp4",
        )
        episode_id = hashlib.sha256(
            f"{session_id}:{run_id}:{interval.clip_id}:{interval.start_frame_index}:{interval.end_frame_index_exclusive}".encode()
        ).hexdigest()[:32]
        span = VirtualVideoSpan(
            session_id=session_id,
            clip_id=interval.clip_id,
            start_frame_index=interval.start_frame_index,
            end_frame_index_exclusive=interval.end_frame_index_exclusive,
            start_session_time_ns=int(first["session_time_ns"]),
            end_session_time_ns=int(last["session_time_ns"]),
            media=media,
        )
        if phase_payload.get("processing_run_id"):
            phase_result = PhaseAnalysisResult.model_validate_json(
                json.dumps(phase_payload)
            )
            phase_summary = tuple(
                dict.fromkeys(
                    item.phase
                    for item in phase_result.segments
                    if item.clip_id == interval.clip_id
                    and item.end_frame_index_exclusive > interval.start_frame_index
                    and item.start_frame_index < interval.end_frame_index_exclusive
                )
            )
        else:
            phase_summary = ()
        annotation_episode = next(
            (
                episode
                for episode in annotation_revision.episodes
                if episode.clip_id == interval.clip_id
                and episode.start.frame_index <= interval.start_frame_index
                and episode.end_exclusive.frame_index >= interval.end_frame_index_exclusive
            ),
            None,
        )
        if annotation_episode is None:
            raise DatasetBuildError(
                "annotation revision does not cover the complete dataset episode"
            )
        return DatasetEpisode(
            episode_id=episode_id,
            source_episode_id=annotation_episode.episode_id,
            span=span,
            processing_run_id=run_id,
            vio_run_id=_run_vio_id(self._run_path(self._session_path(session_id), run_id)),
            annotation_revision_id=annotation_revision_id,
            labels={
                "episode": annotation_episode.labels.model_dump(mode="json"),
                "phases": [
                    phase.model_dump(mode="json") for phase in annotation_episode.phases
                ],
                "source_strategy": annotation_episode.source_strategy.value,
            },
            phase_summary=phase_summary,
            quality_issues=tuple(
                issue for issue in quality.all_issues() if issue.clip_id == interval.clip_id
            ),
        )

    def _session_path(self, session_id: str) -> Path:
        candidate = (self.recordings_root / session_id).resolve()
        if not candidate.is_relative_to(self.recordings_root) or not candidate.is_dir():
            raise DatasetBuildError("recording session is unavailable")
        return candidate

    def _run_path(self, session_path: Path, run_id: str) -> Path:
        candidate = (session_path / "derived" / "video-processing" / run_id).resolve()
        root = (session_path / "derived" / "video-processing").resolve()
        if not candidate.is_relative_to(root) or not candidate.is_dir():
            raise DatasetBuildError("processing run is unavailable")
        return candidate


def _frames_for_episode(
    episode: DatasetEpisode, candidates: tuple[DatasetCandidate, ...]
) -> tuple[DatasetFrame, ...]:
    candidate = next(
        item
        for item in candidates
        if item.run_id == episode.processing_run_id and item.session_id == episode.span.session_id
    )
    store = ProcessingResultStore(candidate.run_directory / "results.sqlite", read_only=True)
    rows = store.iter_results()
    object_payload = _read_json(candidate.run_directory / "objects" / "object-result.json") or {}
    poses = {
        (item.get("clip_id"), int(item.get("frame_index", -1)), item.get("object_id")): item
        for item in object_payload.get("poses", ())
        if isinstance(item, dict)
    }
    masks = {
        (item.get("clip_id"), int(item.get("frame_index", -1)), item.get("object_id")): item
        for item in object_payload.get("masks", ())
        if isinstance(item, dict)
    }
    track_visibility: dict[tuple[object, int, object], float] = {}
    for track in object_payload.get("tracks", ()):
        if not isinstance(track, dict):
            continue
        clip_id = track.get("clip_id")
        object_id = track.get("object_id")
        frame_indices = track.get("frame_indices")
        visibility = track.get("visibility")
        if not isinstance(frame_indices, list) or not isinstance(visibility, list):
            continue
        for offset, frame_index in enumerate(frame_indices):
            if not isinstance(frame_index, int) or frame_index < 0:
                continue
            if offset >= len(visibility) or not isinstance(visibility[offset], list):
                continue
            values = [float(value) for value in visibility[offset]]
            if values:
                key = (clip_id, frame_index, object_id)
                track_visibility[key] = sum(values) / len(values)
    phase_payload = _read_json(candidate.run_directory / "phase-analysis.json") or {}
    phase_by_key = {
        (item.get("clip_id"), int(item.get("frame_index", -1))): item
        for item in phase_payload.get("frames", ())
        if isinstance(item, dict)
    }
    source_clip = next(
        clip
        for clip in CaptureSessionReader.open(candidate.session_path).session.clips
        if clip.clip_id == episode.span.clip_id
    )
    output: list[DatasetFrame] = []
    for row in rows:
        if (
            row.get("sequence_id") != episode.span.clip_id
            or not episode.span.start_frame_index
            <= int(row["frame_index"])
            < episode.span.end_frame_index_exclusive
        ):
            continue
        hands = tuple(_hand_sample(item) for item in row.get("hands", ()) if isinstance(item, dict))
        frame_index = int(row["frame_index"])
        object_keys = {
            key
            for key in (*masks.keys(), *poses.keys(), *track_visibility.keys())
            if key[0] == episode.span.clip_id and key[1] == frame_index
        }
        objects = tuple(
            _object_observation(
                object_id=str(object_id),
                mask_payload=masks.get(key),
                pose_payload=poses.get(key),
                track_visibility=track_visibility.get(key),
            )
            for key in sorted(object_keys, key=lambda value: str(value[2]))
            for object_id in (key[2],)
        )
        phase = phase_by_key.get((episode.span.clip_id, int(row["frame_index"])), {}).get(
            "phase", MotionPhase.STOP.value
        )
        output.append(
            DatasetFrame(
                session_id=episode.span.session_id,
                clip_id=episode.span.clip_id,
                episode_id=episode.episode_id,
                frame_index=int(row["frame_index"]),
                session_time_ns=int(row["session_time_ns"]),
                rgb_reference=ArtifactReference(
                    relative_path=source_clip.media_path.relative_to(
                        candidate.session_path
                    ).as_posix(),
                    sha256=source_clip.sha256,
                    media_type="video/mp4",
                ),
                sensor=SensorSampleReference(
                    telemetry_relative_path="telemetry/telemetry.sqlite",
                    session_time_ns=int(row["session_time_ns"]),
                    imu_window_start_ns=int(row["session_time_ns"]),
                    imu_window_end_ns=int(row["session_time_ns"]),
                ),
                processing_run_id=episode.processing_run_id,
                vio_run_id=episode.vio_run_id,
                hand_result_reference="results.sqlite#frame_results",
                object_result_reference="objects/object-result.json",
                annotation_revision_id=episode.annotation_revision_id,
                configuration_revision=_run_revision(candidate.run_directory),
                source_sha256=source_clip.sha256,
                phase=MotionPhase(phase),
                hands=hands,
                objects=objects,
                quality_state="valid",
            )
        )
    return tuple(output)


def _hand_sample(payload: dict[str, object]) -> HandSample:
    keypoints = tuple(
        tuple(float(value) for value in point) for point in payload.get("keypoints_3d_camera_m", ())
    )
    if len(keypoints) != 21:
        raise DatasetBuildError("final hand result has invalid 3D keypoints")
    kinematics = payload.get("kinematics")
    world = None
    if isinstance(kinematics, dict):
        value = kinematics.get("keypoints_3d_world_m")
        if isinstance(value, list) and len(value) == 21:
            world = tuple(tuple(float(item) for item in point) for point in value)
    return HandSample(
        handedness=str(payload.get("handedness", "unknown")),
        final_confidence=float(payload.get("final_confidence", payload.get("confidence", 0.0))),
        grasp_ratio=float(payload.get("grasp_ratio", 0.0)),
        is_grasping=bool(payload.get("is_grasping", False)),
        keypoints_3d_camera_m=keypoints,
        keypoints_3d_world_m=world,
        temporal_source=(payload.get("temporal") or {}).get("source")
        if isinstance(payload.get("temporal"), dict)
        else None,
    )


def _object_observation(
    *,
    object_id: str,
    mask_payload: dict[str, object] | None,
    pose_payload: dict[str, object] | None,
    track_visibility: float | None,
) -> ObjectObservation:
    mask = (
        ObjectMaskObservation.model_validate_json(json.dumps(mask_payload))
        if mask_payload
        else None
    )
    pose = ObjectPose.model_validate_json(json.dumps(pose_payload)) if pose_payload else None
    source = (
        str(pose_payload.get("source", "unknown"))
        if pose_payload is not None
        else "mask" if mask is not None else "track"
    )
    return ObjectObservation(
        object_id=object_id,
        mask=mask,
        pose=pose,
        track_visibility=track_visibility,
        source=source,
    )


def _run_vio_id(run_directory: Path) -> str:
    payload = _read_json(run_directory / "run.json") or {}
    value = payload.get("vio_run_id")
    if not isinstance(value, str) or not value:
        raise DatasetBuildError("processing run has no VIO run id")
    return value


def _run_revision(run_directory: Path) -> int:
    payload = _read_json(run_directory / "run.json") or {}
    configuration = payload.get("configuration")
    return int(configuration.get("revision", 0)) if isinstance(configuration, dict) else 0


def _load_annotation_revision(
    session_path: Path,
    revision_id: str,
) -> PublishedRevision:
    if len(revision_id) != 32 or any(
        character not in "0123456789abcdef" for character in revision_id
    ):
        raise DatasetBuildError("annotation revision id is invalid")
    path = (
        session_path
        / "annotations"
        / "episode-annotation-v1"
        / "revisions"
        / f"{revision_id}.json"
    )
    payload = _read_json(path)
    if payload is None:
        raise DatasetBuildError("annotation revision is unavailable")
    try:
        revision = PublishedRevision.model_validate(payload)
    except ValueError as error:
        raise DatasetBuildError("annotation revision is invalid") from error
    if revision.session_id != session_path.name or revision.annotation_revision_id != revision_id:
        raise DatasetBuildError("annotation revision provenance does not match the session")
    return revision


def _splits(
    session_ids: tuple[str, ...], episodes: tuple[DatasetEpisode, ...], seed: int
) -> tuple[DatasetSplit, ...]:
    shuffled = list(session_ids)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    train_end = max(1, round(count * 0.8)) if count else 0
    validation_end = min(count, train_end + (1 if count >= 3 else 0))
    groups = {
        "train": tuple(shuffled[:train_end]),
        "validation": tuple(shuffled[train_end:validation_end]),
        "test": tuple(shuffled[validation_end:]),
    }
    return tuple(
        DatasetSplit(
            name=name,
            session_ids=values,
            episode_ids=tuple(
                episode.episode_id for episode in episodes if episode.span.session_id in values
            ),
        )
        for name, values in groups.items()
    )


def _artifact(root: Path, path: Path) -> ArtifactReference:
    return ArtifactReference(
        relative_path=path.relative_to(root).as_posix(),
        sha256=_sha256(path),
        media_type="application/jsonl" if path.suffix == ".jsonl" else "application/json",
    )


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance(candidate: DatasetCandidate) -> dict[str, object]:
    run_path = candidate.run_directory / "run.json"
    run = _read_json(run_path)
    if run is None:
        raise DatasetBuildError("processing run manifest became unavailable during publication")
    vio_run_id = _run_vio_id(candidate.run_directory)
    vio_path = candidate.session_path / "derived" / "vio" / "basalt" / vio_run_id / "run.json"
    vio = _read_json(vio_path)
    if vio is None:
        raise DatasetBuildError("VIO run manifest became unavailable during publication")
    annotation_path = (
        candidate.session_path
        / "annotations"
        / "episode-annotation-v1"
        / "revisions"
        / f"{candidate.annotation_revision_id}.json"
    )
    annotation = _load_annotation_revision(
        candidate.session_path,
        candidate.annotation_revision_id,
    )
    media = {
        (episode.span.clip_id, episode.span.media.sha256)
        for episode in candidate.episodes
    }
    artifact_paths = (
        candidate.run_directory / "results.sqlite",
        candidate.run_directory / "phases.jsonl",
        candidate.run_directory / "phase-analysis.json",
        candidate.run_directory / "objects" / "object-result.json",
        candidate.run_directory / "objects" / "selected-keypoints.jsonl",
        candidate.run_directory / "objects" / "tracks.json",
        candidate.run_directory / "objects" / "triangulation.json",
        candidate.run_directory / "objects" / "object-qa.json",
        *sorted((candidate.run_directory / "objects" / "masks").glob("*.png")),
    )
    return {
        "session_id": candidate.session_id,
        "processing_run_id": candidate.run_id,
        "annotation_revision_id": candidate.annotation_revision_id,
        "annotation": {
            "relative_path": annotation_path.relative_to(candidate.session_path).as_posix(),
            "sha256": _sha256(annotation_path),
            "content_sha256": annotation.content_sha256,
        },
        "run_manifest_sha256": _sha256(run_path),
        "configuration": run.get("configuration"),
        "task_profile_id": run.get("task_profile_id"),
        "task_profile": run.get("task_profile"),
        "object_tracking": run.get("object_tracking"),
        "vio": {
            "run_id": vio_run_id,
            "manifest_sha256": _sha256(vio_path),
            "calibration_profile_id": vio.get("calibration_profile_id"),
            "calibration_verified": vio.get("calibration_verified"),
            "sensor_config_sha256": vio.get("sensor_config_sha256"),
            "basalt_config_sha256": vio.get("basalt_config_sha256"),
            "basalt_revision": vio.get("basalt_revision"),
        },
        "source_media": [
            {"clip_id": clip_id, "sha256": sha256}
            for clip_id, sha256 in sorted(media)
        ],
        "artifacts": [
            {
                "relative_path": path.relative_to(candidate.run_directory).as_posix(),
                "sha256": _sha256(path),
            }
            for path in artifact_paths
            if path.is_file()
        ],
    }


def _frame_key(row: dict[str, object]) -> tuple[int, int]:
    return int(row.get("frame_index", 0)), int(row.get("session_time_ns", 0))


def _validate_id(value: str) -> None:
    if not value or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_.-" for character in value.lower()
    ):
        raise DatasetBuildError("dataset id contains unsupported characters")


def _validated_quality_report(
    checked: DatasetQualityReport,
    supplied: DatasetQualityReport | None,
) -> DatasetQualityReport:
    """Accept only audited restoration fields from a previously checked report."""

    if supplied is None:
        return checked
    if (supplied.session_id, supplied.run_id) != (checked.session_id, checked.run_id):
        raise DatasetBuildError("quality report does not belong to the selected run")
    if supplied.hard_issues != checked.hard_issues or supplied.metrics != checked.metrics:
        raise DatasetBuildError("hard gates and quality metrics cannot be overridden")
    checked_soft = {issue.issue_id: issue for issue in checked.soft_issues}
    supplied_soft = {issue.issue_id: issue for issue in supplied.soft_issues}
    if checked_soft.keys() != supplied_soft.keys():
        raise DatasetBuildError("quality report issue set changed during review")
    immutable_fields = (
        "gate",
        "severity",
        "message",
        "clip_id",
        "start_frame_index",
        "end_frame_index_exclusive",
        "restorable",
    )
    for issue_id, original in checked_soft.items():
        reviewed = supplied_soft[issue_id]
        if any(getattr(original, field) != getattr(reviewed, field) for field in immutable_fields):
            raise DatasetBuildError("quality review may only restore an existing soft gate")
        if reviewed.restored:
            reviewed.validate_restore()
    return supplied
