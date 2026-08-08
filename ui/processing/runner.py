from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from hand_tracking import (
    HandTrackingConfig,
    HandTrackingResult,
    HumanEgoHandTrackingPipeline,
    OfflineHandTemporalConfig,
    OfflineHandTemporalProcessor,
    TemporalProcessingStats,
    release_pipeline_resources,
)
from object_tracking import (
    ObjectFrameInput,
    ObjectTrackingConfig,
    ObjectTrackingError,
    OfflineObjectProcessing,
    TaskProfile,
    load_object_tracking_config,
)
from phase_analysis import PhaseAnalysisConfig, PhaseAnalysisService, PhaseInputFrame
from schemas.phase import ObjectCentricWindow, PhaseAnalysisResult
from sensor_preprocessing import (
    CaptureSessionReader,
    SensorCalibration,
    SensorPreprocessingConfig,
    SensorPreprocessingPipeline,
    derive_recorded_clock_mapping,
)

from .models import ProcessingJob, ProcessingRunSummary
from .results import ProcessingResultStore
from .vio import VioRunInfo


class ProcessingCanceled(RuntimeError):
    pass


class SessionProcessingRunner:
    """Validate, preprocess, and index one immutable capture-session run."""

    def __init__(
        self,
        *,
        sensor_config_path: str | Path = "config/sensor-preprocessing.yaml",
        offline_hand_tracking_config_path: str | Path = "config/offline-hand-tracking.yaml",
        object_tracking_config_path: str | Path = "config/object-tracking.yaml",
        tracker_factory: Callable[[], HumanEgoHandTrackingPipeline] | None = None,
    ) -> None:
        self.sensor_config_path = Path(sensor_config_path).resolve()
        self.offline_hand_tracking_config_path = Path(offline_hand_tracking_config_path).resolve()
        self.object_tracking_config_path = Path(object_tracking_config_path).resolve()
        self._tracker_factory = tracker_factory
        self._tracker: HumanEgoHandTrackingPipeline | None = None

    def inspect(self, session_path: Path, clip_id: str | None) -> int:
        reader = CaptureSessionReader.open(session_path)
        clips = (
            reader.session.clips
            if clip_id is None
            else tuple(clip for clip in reader.session.clips if clip.clip_id == clip_id)
        )
        if not clips:
            raise KeyError(f"unknown complete clip {clip_id!r}")
        return sum(clip.frame_count for clip in clips)

    def run(
        self,
        job: ProcessingJob,
        session_path: Path,
        output_directory: Path,
        *,
        progress: Callable[[int, int], None],
        is_canceled: Callable[[], bool],
        vio_run_info: VioRunInfo | None = None,
        vio_error: str | None = None,
    ) -> ProcessingRunSummary:
        started_ns = time.time_ns()
        output_directory.mkdir(parents=True, exist_ok=False)
        run_manifest_path = output_directory / "run.json"
        run_log_path = output_directory / "run.log"
        result_store = ProcessingResultStore(output_directory / "results.sqlite")
        self._write_log(run_log_path, "run preparing")
        self._write_manifest(
            run_manifest_path,
            job,
            state="running",
            started_at_unix_ns=started_ns,
        )

        reader = CaptureSessionReader.open(session_path)
        frame_evidence = tuple(
            frame for clip in reader.session.clips for frame in reader.iter_frames(clip.clip_id)
        )
        imu_evidence = tuple(reader.iter_imu_samples())
        recorded_mapping = derive_recorded_clock_mapping(
            reader.session.session_id,
            frame_evidence,
            imu_evidence,
        )
        snapshotted = _processing_configuration(job)
        if snapshotted is None:
            preprocessing = SensorPreprocessingPipeline.from_config_file(
                self.sensor_config_path,
                recorded_mapping.mapper,
            )
            hand_config = None
        else:
            sensor_config, calibration, hand_config = snapshotted
            preprocessing = SensorPreprocessingPipeline(
                calibration,
                recorded_mapping.mapper,
                recorded_config=sensor_config.recorded,
                image_config=sensor_config.image,
                live_config=sensor_config.live,
            )
        selected = None if job.clip_id is None else {job.clip_id}
        total = sum(
            clip.frame_count
            for clip in reader.session.clips
            if selected is None or clip.clip_id in selected
        )
        if total < 1:
            raise ValueError("processing selection contains no frames")

        input_count = 0
        inferred_count = 0
        raw_detected_count = 0
        detected_count = 0
        effective_hand_config = hand_config or HandTrackingConfig.load(
            self.offline_hand_tracking_config_path
        )
        temporal_config = effective_hand_config.temporal_processing or OfflineHandTemporalConfig()
        temporal_processor = OfflineHandTemporalProcessor(
            temporal_config,
            grasp_ratio_threshold=effective_hand_config.grasp_ratio_threshold,
        )
        tracker = self._tracker_for_worker(effective_hand_config)
        raw_results_by_clip: dict[str, list[HandTrackingResult]] = {}
        final_results_by_clip: dict[str, list[HandTrackingResult]] = {}
        temporal_stats: list[TemporalProcessingStats] = []
        partial = vio_error is not None or vio_run_info is None or not vio_run_info.is_viewable
        phase_payload: dict[str, object] = {
            "state": "unavailable",
            "reason": "VIO unavailable",
        }
        object_payload: dict[str, object] = {"state": "not_requested"}
        try:
            for bundle in preprocessing.iter_recorded_session(
                session_path,
                verify_media_hashes=False,
                clip_ids=selected,
            ):
                if is_canceled():
                    raise ProcessingCanceled("processing canceled")
                input_count += 1
                if bundle.frame_index % job.preset.inference_stride_frames == 0:
                    result = tracker.process_frame(bundle)
                    result_store.put_raw(result.to_json_dict())
                    raw_results_by_clip.setdefault(result.sequence_id, []).append(result)
                    inferred_count += 1
                    raw_detected_count += len(result.hands)
                progress(input_count, total)
            for clip_results in raw_results_by_clip.values():
                if is_canceled():
                    raise ProcessingCanceled("processing canceled")
                output = temporal_processor.process_clip(
                    clip_results,
                    trajectory=vio_run_info.trajectory if vio_run_info is not None else None,
                    transform_camera_to_imu=(
                        vio_run_info.transform_camera_to_imu if vio_run_info is not None else None
                    ),
                )
                partial = partial or output.partial_world_coverage
                temporal_stats.append(output.stats)
                for final_result in output.final_results:
                    result_store.put_final(final_result.to_json_dict())
                    final_results_by_clip.setdefault(final_result.sequence_id, []).append(
                        final_result
                    )
                    detected_count += len(final_result.hands)
            if vio_run_info is not None and vio_run_info.is_viewable:
                phase_started_ns = time.perf_counter_ns()
                phase_result = self._run_phase_analysis(
                    output_directory,
                    final_results_by_clip,
                    vio_run_info,
                )
                phase_payload = {
                    "state": "completed",
                    "frame_count": len(phase_result.frames),
                    "segment_count": len(phase_result.segments),
                    "object_window_count": len(phase_result.object_centric_windows),
                    "artifact": "phases.jsonl",
                    "duration_ns": time.perf_counter_ns() - phase_started_ns,
                    "configuration": PhaseAnalysisConfig().model_dump(mode="json"),
                    "configuration_sha256": _json_sha256(
                        PhaseAnalysisConfig().model_dump(mode="json")
                    ),
                }
                if job.task_profile_id is not None:
                    object_started_ns = time.perf_counter_ns()
                    try:
                        object_payload = self._run_object_tracking(
                            output_directory,
                            job.task_profile_id,
                            job.task_profile_snapshot_json,
                            job.configuration_snapshot_json,
                            phase_result.object_centric_windows,
                            final_results_by_clip,
                            preprocessing,
                            session_path,
                            vio_run_info,
                            is_canceled,
                        )
                        object_payload["duration_ns"] = time.perf_counter_ns() - object_started_ns
                    except ProcessingCanceled:
                        raise
                    except ObjectTrackingError as error:
                        partial = True
                        object_payload = {"state": "failed", "error": str(error)}
                        self._write_log(run_log_path, f"object stage partial: {error}")
            elif job.task_profile_id is not None:
                partial = True
                object_payload = {"state": "blocked", "error": "VIO unavailable"}
                self._write_log(run_log_path, "object stage blocked: VIO unavailable")
        except BaseException as error:
            state = "canceled" if isinstance(error, ProcessingCanceled) else "failed"
            self._write_log(run_log_path, f"run {state}: {error}")
            self._write_manifest(
                run_manifest_path,
                job,
                state=state,
                started_at_unix_ns=started_ns,
                completed_at_unix_ns=time.time_ns(),
                error=str(error),
                input_frame_count=input_count,
                inferred_frame_count=inferred_count,
                detected_hand_count=detected_count,
                raw_detected_hand_count=raw_detected_count,
                vio_run_id=vio_run_info.run_id if vio_run_info is not None else None,
                vio_error=vio_error,
                temporal_stats=_aggregate_temporal_stats(temporal_stats),
                phase_analysis=phase_payload,
                object_tracking=object_payload,
            )
            raise

        completed_ns = time.time_ns()
        state = "partial" if partial else "completed"
        self._write_log(run_log_path, f"run {state}")
        self._write_manifest(
            run_manifest_path,
            job,
            state=state,
            started_at_unix_ns=started_ns,
            completed_at_unix_ns=completed_ns,
            input_frame_count=input_count,
            inferred_frame_count=inferred_count,
            detected_hand_count=detected_count,
            raw_detected_hand_count=raw_detected_count,
            vio_run_id=vio_run_info.run_id if vio_run_info is not None else None,
            vio_error=vio_error,
            temporal_stats=_aggregate_temporal_stats(temporal_stats),
            phase_analysis=phase_payload,
            object_tracking=object_payload,
        )
        return ProcessingRunSummary(
            run_id=output_directory.name,
            session_id=job.session_id,
            clip_id=job.clip_id,
            output_directory=output_directory,
            input_frame_count=input_count,
            inferred_frame_count=inferred_count,
            detected_hand_count=detected_count,
            started_at_unix_ns=started_ns,
            completed_at_unix_ns=completed_ns,
            partial=partial,
        )

    def _tracker_for_worker(
        self,
        config: HandTrackingConfig | None,
    ) -> HumanEgoHandTrackingPipeline:
        if self._tracker is None:
            if self._tracker_factory is not None:
                self._tracker = self._tracker_factory()
            elif config is not None:
                self._tracker = HumanEgoHandTrackingPipeline.from_config(config)
            else:
                self._tracker = HumanEgoHandTrackingPipeline.from_config_file(
                    str(self.offline_hand_tracking_config_path)
                )
        return self._tracker

    def release_gpu(self) -> None:
        """Release the offline model before live inference may resume."""

        tracker = self._tracker
        self._tracker = None
        release_pipeline_resources(tracker)

    def _run_phase_analysis(
        self,
        output_directory: Path,
        final_results_by_clip: dict[str, list[HandTrackingResult]],
        vio_run_info: VioRunInfo,
    ) -> PhaseAnalysisResult:
        inputs: list[PhaseInputFrame] = []
        for clip_id, results in final_results_by_clip.items():
            for result in sorted(results, key=lambda item: item.frame_index):
                pose = vio_run_info.pose_at(result.session_time_ns)
                if pose is None:
                    continue
                hand_speeds = [
                    float(np.linalg.norm(hand.kinematics.midpoint_linear_velocity_optimized_m_s))
                    for hand in result.hands
                    if hand.kinematics is not None
                ]
                inputs.append(
                    PhaseInputFrame(
                        clip_id=clip_id,
                        frame_index=result.frame_index,
                        session_time_ns=result.session_time_ns,
                        head_position_m=pose.position_m,
                        head_quaternion_wxyz=pose.quaternion_wxyz,
                        hand_linear_speed_m_s=max(hand_speeds, default=0.0),
                        grasping=any(hand.is_grasping for hand in result.hands),
                    )
                )
        phase_result = PhaseAnalysisService(PhaseAnalysisConfig()).analyze(
            output_directory.name,
            inputs,
        )
        with (output_directory / "phases.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            for frame in phase_result.frames:
                stream.write(json.dumps(frame.model_dump(mode="json"), ensure_ascii=False) + "\n")
        (output_directory / "phase-analysis.json").write_text(
            json.dumps(
                phase_result.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return phase_result

    def _run_object_tracking(
        self,
        output_directory: Path,
        task_profile_id: str,
        task_profile_snapshot_json: str,
        configuration_snapshot_json: str,
        windows: tuple[ObjectCentricWindow, ...],
        final_results_by_clip: dict[str, list[HandTrackingResult]],
        preprocessing: SensorPreprocessingPipeline,
        session_path: Path,
        vio_run_info: VioRunInfo,
        is_canceled: Callable[[], bool],
    ) -> dict[str, object]:
        configuration_snapshot = json.loads(configuration_snapshot_json)
        object_config_snapshot = (
            configuration_snapshot.get("object_tracking")
            if isinstance(configuration_snapshot, dict)
            else None
        )
        config = (
            ObjectTrackingConfig.model_validate_json(json.dumps(object_config_snapshot))
            if isinstance(object_config_snapshot, dict)
            else load_object_tracking_config(self.object_tracking_config_path)
        )
        snapshot = json.loads(task_profile_snapshot_json)
        profile = (
            TaskProfile.model_validate(snapshot)
            if isinstance(snapshot, dict) and snapshot
            else config.profile(task_profile_id)
        )
        if profile.profile_id != task_profile_id:
            raise ObjectTrackingError("task profile snapshot id does not match queued job")
        final_hands = {
            (clip_id, result.frame_index): result.hands
            for clip_id, results in final_results_by_clip.items()
            for result in results
        }
        window_keys = {
            (window.clip_id, frame_index)
            for window in windows
            for frame_index in range(window.start_frame_index, window.end_frame_index_exclusive)
        }
        frames_by_clip: dict[str, list[ObjectFrameInput]] = {}
        for bundle in preprocessing.iter_recorded_session(
            session_path,
            verify_media_hashes=False,
            clip_ids=set(final_results_by_clip),
        ):
            key = (bundle.sequence_id, bundle.frame_index)
            if key not in window_keys:
                continue
            frames_by_clip.setdefault(bundle.sequence_id, []).append(
                ObjectFrameInput(
                    clip_id=bundle.sequence_id,
                    frame_index=bundle.frame_index,
                    session_time_ns=bundle.session_time_ns,
                    image_bgr=bundle.image_bgr,
                    intrinsics=np.asarray(
                        bundle.calibration.rectified_camera_matrix,
                        dtype=np.float64,
                    ),
                    hands=final_hands.get(key, ()),
                )
            )
        if not frames_by_clip:
            raise ObjectTrackingError("object windows contain no decoded frames")
        if vio_run_info.trajectory is None:
            raise ObjectTrackingError("object tracking requires a parsed VIO trajectory")
        pipeline = OfflineObjectProcessing(config)
        result = pipeline.run(
            output_directory.name,
            profile,
            windows,
            {
                key: tuple(sorted(value, key=lambda frame: frame.frame_index))
                for key, value in frames_by_clip.items()
            },
            vio_run_info.trajectory,
            np.asarray(vio_run_info.transform_camera_to_imu, dtype=np.float64),
            output_directory / "objects",
            is_canceled=is_canceled,
        )
        return {
            "state": "completed",
            "task_profile_id": task_profile_id,
            "configuration": config.model_dump(mode="json"),
            "configuration_sha256": _json_sha256(config.model_dump(mode="json")),
            "mask_count": len(result.masks),
            "track_count": len(result.tracks),
            "triangulation_count": len(result.triangulations),
            "pose_count": len(result.poses),
            "artifact": "objects/object-result.json",
        }

    @staticmethod
    def _write_log(path: Path, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(f"{timestamp} {message}\n")
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _write_manifest(
        path: Path,
        job: ProcessingJob,
        *,
        state: str,
        started_at_unix_ns: int,
        completed_at_unix_ns: int | None = None,
        error: str | None = None,
        input_frame_count: int = 0,
        inferred_frame_count: int = 0,
        detected_hand_count: int = 0,
        raw_detected_hand_count: int = 0,
        vio_run_id: str | None = None,
        vio_error: str | None = None,
        temporal_stats: TemporalProcessingStats | None = None,
        phase_analysis: dict[str, object] | None = None,
        object_tracking: dict[str, object] | None = None,
    ) -> None:
        payload = {
            "schema_version": "1.0",
            "contract_id": "video-processing-run-v1",
            "run_id": path.parent.name,
            "job_id": job.job_id,
            "session_id": job.session_id,
            "clip_id": job.clip_id,
            "state": state,
            "preset": {
                "preset_id": job.preset.preset_id,
                "display_name": job.preset.display_name,
                "inference_stride_frames": job.preset.inference_stride_frames,
            },
            "configuration": {
                "revision": job.configuration_revision,
                "sha256_by_file": dict(job.configuration_sha256_by_file),
                "snapshot": json.loads(job.configuration_snapshot_json),
            },
            "started_at_unix_ns": started_at_unix_ns,
            "completed_at_unix_ns": completed_at_unix_ns,
            "input_frame_count": input_frame_count,
            "inferred_frame_count": inferred_frame_count,
            "detected_hand_count": detected_hand_count,
            "raw_detected_hand_count": raw_detected_hand_count,
            "vio_run_id": vio_run_id,
            "vio_error": vio_error,
            "task_profile_id": job.task_profile_id,
            "task_profile": json.loads(job.task_profile_snapshot_json),
            "phase_analysis": phase_analysis,
            "object_tracking": object_tracking,
            "temporal_processing": (
                _temporal_stats_payload(temporal_stats) if temporal_stats is not None else None
            ),
            "stages": {
                "capture_validation": {
                    "state": "completed" if input_frame_count else "pending",
                    "input_frame_count": input_frame_count,
                },
                "sensor_preprocessing": {
                    "state": "completed" if input_frame_count else "pending",
                    "output_frame_count": input_frame_count,
                },
                "basalt_vio": {
                    "state": "failed" if vio_error else ("completed" if vio_run_id else "pending"),
                    "run_id": vio_run_id,
                    "error": vio_error,
                },
                "hand_inference": {
                    "state": "completed" if inferred_frame_count else "pending",
                    "input_frame_count": input_frame_count,
                    "output_frame_count": inferred_frame_count,
                },
                "hand_temporal_processing": {
                    "state": "completed" if temporal_stats is not None else "pending",
                    "output_hand_count": detected_hand_count,
                },
                "phase_analysis": phase_analysis,
                "object_tracking": object_tracking,
                "result_index": {
                    "state": "completed" if state in {"completed", "partial"} else state,
                    "artifact": "results.sqlite",
                },
            },
            "error": error,
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)


def _processing_configuration(
    job: ProcessingJob,
) -> tuple[SensorPreprocessingConfig, SensorCalibration, HandTrackingConfig] | None:
    payload = json.loads(job.configuration_snapshot_json)
    if not payload:
        return None
    if not isinstance(payload, dict):
        raise ValueError("processing configuration snapshot must be an object")
    sensor_payload = payload.get("sensor_preprocessing")
    calibration_payload = payload.get("sensor_calibration")
    hand_payload = payload.get("offline_hand_tracking")
    if hand_payload is None:
        hand_payload = payload.get("hand_tracking")
    if not all(
        isinstance(value, dict) for value in (sensor_payload, calibration_payload, hand_payload)
    ):
        raise ValueError("processing configuration snapshot is incomplete")
    assert isinstance(sensor_payload, dict)
    assert isinstance(calibration_payload, dict)
    assert isinstance(hand_payload, dict)
    sensor_values = dict(sensor_payload)
    sensor_values["calibration_file"] = Path(str(sensor_values["calibration_file"]))
    hand_values = dict(hand_payload)
    hand_values["model_directory"] = Path(str(hand_values["model_directory"]))
    return (
        SensorPreprocessingConfig.model_validate(sensor_values),
        SensorCalibration.model_validate(_tupleize(calibration_payload)),
        HandTrackingConfig.model_validate(hand_values),
    )


def _tupleize(value: object) -> object:
    if isinstance(value, dict):
        return {key: _tupleize(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(_tupleize(item) for item in value)
    return value


def _aggregate_temporal_stats(
    stats: list[TemporalProcessingStats],
) -> TemporalProcessingStats:
    fields = (
        "raw_hand_frames",
        "confidence_rejected",
        "interpolated_frames",
        "suppressed_frames",
        "final_hand_frames",
        "grasp_transitions_before",
        "grasp_transitions_after",
        "grasp_states_changed",
        "vio_matched_frames",
        "world_optimized_frames",
        "temporal_processing_duration_ns",
        "confidence_filter_duration_ns",
        "interpolation_duration_ns",
        "segment_suppression_duration_ns",
        "grasp_smoothing_duration_ns",
        "world_mapping_duration_ns",
        "kinematic_optimization_duration_ns",
    )
    if not stats:
        return TemporalProcessingStats(**dict.fromkeys(fields, 0))
    return TemporalProcessingStats(
        **{field: sum(getattr(item, field) for item in stats) for field in fields}
    )


def _temporal_stats_payload(stats: TemporalProcessingStats) -> dict[str, object]:
    return {
        "raw_hand_frames": stats.raw_hand_frames,
        "confidence_rejected": stats.confidence_rejected,
        "interpolated_frames": stats.interpolated_frames,
        "suppressed_frames": stats.suppressed_frames,
        "final_hand_frames": stats.final_hand_frames,
        "grasp_transitions_before": stats.grasp_transitions_before,
        "grasp_transitions_after": stats.grasp_transitions_after,
        "grasp_states_changed": stats.grasp_states_changed,
        "vio_matched_frames": stats.vio_matched_frames,
        "world_optimized_frames": stats.world_optimized_frames,
        "vio_coverage_ratio": stats.vio_coverage_ratio,
        "duration_ns": stats.temporal_processing_duration_ns,
        "stage_durations_ns": {
            "confidence_filter": stats.confidence_filter_duration_ns,
            "interpolation": stats.interpolation_duration_ns,
            "segment_suppression": stats.segment_suppression_duration_ns,
            "grasp_smoothing": stats.grasp_smoothing_duration_ns,
            "world_mapping": stats.world_mapping_duration_ns,
            "kinematic_optimization": stats.kinematic_optimization_duration_ns,
        },
    }


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
