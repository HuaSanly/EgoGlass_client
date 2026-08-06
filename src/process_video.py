"""Run one offline capture-session processing pass without the Qt client.

This is deliberately a thin command-line entry point. The UI workflow owns
jobs, retries, SQLite indexing, and replay; this module only executes one
session and writes a small JSONL result stream.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from hand_tracking import (
    HandTrackingConfig,
    HandTrackingResult,
    HumanEgoHandTrackingPipeline,
    OfflineHandTemporalConfig,
    OfflineHandTemporalProcessor,
    TemporalProcessingStats,
    release_pipeline_resources,
)
from run_vio import run_session as run_vio_session
from sensor_preprocessing import (
    CaptureSessionReader,
    SensorPreprocessingPipeline,
    derive_recorded_clock_mapping,
)
from slam_vio import parse_euroc_trajectory


def process_session(
    session_path: str | Path,
    output_directory: str | Path,
    *,
    sensor_config_path: str | Path = "config/sensor-preprocessing.yaml",
    hand_tracking_config_path: str | Path = "config/offline-hand-tracking.yaml",
    basalt_config_path: str | Path = "config/basalt-vio.yaml",
    clip_id: str | None = None,
) -> dict[str, Any]:
    """Process every frame in one complete capture session.

    The canonical input is a validated capture-session directory. A raw MP4
    is rejected because it does not contain the clock and IMU evidence needed
    by the sensor-preprocessing contract.
    """

    session_directory = Path(session_path).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    reader = CaptureSessionReader.open(session_directory)
    selected_clip_ids = None if clip_id is None else {clip_id}
    selected_clips = tuple(
        clip
        for clip in reader.session.clips
        if selected_clip_ids is None or clip.clip_id in selected_clip_ids
    )
    if not selected_clips:
        raise ValueError(f"unknown clip_id: {clip_id}")

    frame_evidence = tuple(
        frame
        for clip in selected_clips
        for frame in reader.iter_frames(clip.clip_id)
    )
    imu_evidence = tuple(reader.iter_imu_samples())
    mapping = derive_recorded_clock_mapping(
        reader.session.session_id,
        frame_evidence,
        imu_evidence,
    )
    preprocessing = SensorPreprocessingPipeline.from_config_file(
        sensor_config_path,
        mapping.mapper,
    )
    hand_config = HandTrackingConfig.load(hand_tracking_config_path)
    temporal_processor = OfflineHandTemporalProcessor(
        hand_config.temporal_processing or OfflineHandTemporalConfig(),
        grasp_ratio_threshold=hand_config.grasp_ratio_threshold,
    )

    run_id = output.name or uuid.uuid4().hex
    manifest_path = output / "run.json"
    results_path = output / "results.jsonl"
    raw_results_path = output / "raw_results.jsonl"
    log_path = output / "run.log"
    started_at_ns = time.time_ns()
    _write_log(log_path, f"run preparing session={reader.session.session_id}")
    _write_manifest(
        manifest_path,
        run_id=run_id,
        session_id=reader.session.session_id,
        clip_id=clip_id,
        state="running",
        started_at_ns=started_at_ns,
        sensor_config_path=sensor_config_path,
        hand_tracking_config_path=hand_tracking_config_path,
        basalt_config_path=basalt_config_path,
    )

    total = sum(clip.frame_count or 0 for clip in selected_clips)
    input_frame_count = 0
    raw_detected_hand_count = 0
    detected_hand_count = 0
    raw_results_by_clip: dict[str, list[HandTrackingResult]] = {}
    temporal_stats: list[TemporalProcessingStats] = []
    vio_error: str | None = None
    trajectory = None
    transform_camera_to_imu: object | None = None
    tracker: HumanEgoHandTrackingPipeline | None = None
    try:
        try:
            vio_output = output / "vio"
            vio_summary = run_vio_session(
                session_directory,
                vio_output,
                sensor_config_path=sensor_config_path,
                basalt_config_path=basalt_config_path,
                clip_id=clip_id,
                allow_unverified_calibration=True,
            )
            trajectory = parse_euroc_trajectory(vio_output / "trajectory.csv")
            transform_camera_to_imu = preprocessing.calibration.transform_camera_to_body
        except Exception as error:
            vio_error = str(error)
            vio_summary = None
            _write_log(log_path, f"VIO unavailable: {vio_error}")

        tracker = HumanEgoHandTrackingPipeline.from_config(hand_config)
        with raw_results_path.open("w", encoding="utf-8", newline="\n") as stream:
            for bundle in preprocessing.iter_recorded_session(
                session_directory,
                clip_ids=selected_clip_ids,
            ):
                result = tracker.process_frame(bundle)
                stream.write(json.dumps(result.to_json_dict(), ensure_ascii=False) + "\n")
                stream.flush()
                raw_results_by_clip.setdefault(result.sequence_id, []).append(result)
                input_frame_count += 1
                raw_detected_hand_count += len(result.hands)
        partial = vio_error is not None
        with results_path.open("w", encoding="utf-8", newline="\n") as stream:
            for clip_results in raw_results_by_clip.values():
                temporal_output = temporal_processor.process_clip(
                    clip_results,
                    trajectory=trajectory,
                    transform_camera_to_imu=transform_camera_to_imu,
                )
                temporal_stats.append(temporal_output.stats)
                partial = partial or temporal_output.partial_world_coverage
                for result in temporal_output.final_results:
                    stream.write(json.dumps(result.to_json_dict(), ensure_ascii=False) + "\n")
                    detected_hand_count += len(result.hands)
            stream.flush()
        completed_at_ns = time.time_ns()
        state = "partial" if partial else "completed"
        _write_log(log_path, f"run {state}")
        summary = {
            "run_id": run_id,
            "session_id": reader.session.session_id,
            "clip_id": clip_id,
            "state": state,
            "input_frame_count": input_frame_count,
            "expected_frame_count": total,
            "inferred_frame_count": input_frame_count,
            "detected_hand_count": detected_hand_count,
            "raw_detected_hand_count": raw_detected_hand_count,
            "started_at_unix_ns": started_at_ns,
            "completed_at_unix_ns": completed_at_ns,
            "results_path": str(results_path),
            "raw_results_path": str(raw_results_path),
            "vio_run": vio_summary,
            "vio_error": vio_error,
            "temporal_processing": _temporal_stats_payload(temporal_stats),
        }
        _write_manifest(
            manifest_path,
            **summary,
            sensor_config_path=sensor_config_path,
            hand_tracking_config_path=hand_tracking_config_path,
            basalt_config_path=basalt_config_path,
        )
        return summary
    except BaseException as error:
        _write_log(log_path, f"run failed: {error}")
        _write_manifest(
            manifest_path,
            run_id=run_id,
            session_id=reader.session.session_id,
            clip_id=clip_id,
            state="failed",
            input_frame_count=input_frame_count,
            expected_frame_count=total,
            inferred_frame_count=input_frame_count,
            detected_hand_count=detected_hand_count,
            raw_detected_hand_count=raw_detected_hand_count,
            started_at_unix_ns=started_at_ns,
            completed_at_unix_ns=time.time_ns(),
            error=str(error),
            sensor_config_path=sensor_config_path,
            hand_tracking_config_path=hand_tracking_config_path,
            basalt_config_path=basalt_config_path,
        )
        raise
    finally:
        release_pipeline_resources(tracker)


def _write_manifest(path: Path, **payload: Any) -> None:
    manifest = {
        "schema_version": "1.0",
        "contract_id": "headless-video-processing-run-v1",
        **payload,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_log(path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{timestamp} {message}\n")


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported manifest value: {type(value).__name__}")


def _temporal_stats_payload(stats: list[TemporalProcessingStats]) -> dict[str, object]:
    names = (
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
    totals = {name: sum(int(getattr(item, name)) for item in stats) for name in names}
    final_count = totals["final_hand_frames"]
    totals["vio_coverage_ratio"] = (
        totals["vio_matched_frames"] / final_count if final_count else 0.0
    )
    totals["stage_durations_ns"] = {
        "confidence_filter": totals.pop("confidence_filter_duration_ns"),
        "interpolation": totals.pop("interpolation_duration_ns"),
        "segment_suppression": totals.pop("segment_suppression_duration_ns"),
        "grasp_smoothing": totals.pop("grasp_smoothing_duration_ns"),
        "world_mapping": totals.pop("world_mapping_duration_ns"),
        "kinematic_optimization": totals.pop("kinematic_optimization_duration_ns"),
    }
    return totals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process one EgoGlass capture session without Qt")
    parser.add_argument(
        "--session",
        type=Path,
        required=True,
        help="complete capture-session directory",
    )
    parser.add_argument(
        "--basalt-config",
        type=Path,
        default=Path("config/basalt-vio.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--clip-id", help="process only one clip")
    parser.add_argument(
        "--sensor-config",
        type=Path,
        default=Path("config/sensor-preprocessing.yaml"),
    )
    parser.add_argument(
        "--hand-tracking-config",
        type=Path,
        default=Path("config/offline-hand-tracking.yaml"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = process_session(
            args.session,
            args.output,
            sensor_config_path=args.sensor_config,
            hand_tracking_config_path=args.hand_tracking_config,
            basalt_config_path=args.basalt_config,
            clip_id=args.clip_id,
        )
    except Exception as error:
        print(f"EgoGlass offline processing failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
