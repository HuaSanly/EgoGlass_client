from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path

from hand_tracking import (
    HandTrackingConfig,
    HandTrackingResult,
    HumanEgoHandTrackingPipeline,
    OfflineHandTemporalConfig,
    OfflineHandTemporalProcessor,
    TemporalProcessingStats,
    release_pipeline_resources,
)
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
        tracker_factory: Callable[[], HumanEgoHandTrackingPipeline] | None = None,
    ) -> None:
        self.sensor_config_path = Path(sensor_config_path).resolve()
        self.offline_hand_tracking_config_path = Path(
            offline_hand_tracking_config_path
        ).resolve()
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
            frame
            for clip in reader.session.clips
            for frame in reader.iter_frames(clip.clip_id)
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
        temporal_config = (
            effective_hand_config.temporal_processing or OfflineHandTemporalConfig()
        )
        temporal_processor = OfflineHandTemporalProcessor(
            temporal_config,
            grasp_ratio_threshold=effective_hand_config.grasp_ratio_threshold,
        )
        tracker = self._tracker_for_worker(effective_hand_config)
        raw_results_by_clip: dict[str, list[HandTrackingResult]] = {}
        temporal_stats: list[TemporalProcessingStats] = []
        partial = vio_error is not None or vio_run_info is None or not vio_run_info.is_viewable
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
                        vio_run_info.transform_camera_to_imu
                        if vio_run_info is not None
                        else None
                    ),
                )
                partial = partial or output.partial_world_coverage
                temporal_stats.append(output.stats)
                for final_result in output.final_results:
                    result_store.put_final(final_result.to_json_dict())
                    detected_count += len(final_result.hands)
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
            "temporal_processing": (
                _temporal_stats_payload(temporal_stats)
                if temporal_stats is not None
                else None
            ),
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
        isinstance(value, dict)
        for value in (sensor_payload, calibration_payload, hand_payload)
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
        **{
            field: sum(getattr(item, field) for item in stats)
            for field in fields
        }
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
