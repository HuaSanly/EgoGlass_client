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

from hand_tracking import HumanEgoHandTrackingPipeline, release_pipeline_resources
from sensor_preprocessing import (
    CaptureSessionReader,
    SensorPreprocessingPipeline,
    derive_recorded_clock_mapping,
)


def process_session(
    session_path: str | Path,
    output_directory: str | Path,
    *,
    sensor_config_path: str | Path = "config/sensor-preprocessing.yaml",
    hand_tracking_config_path: str | Path = "config/offline-hand-tracking.yaml",
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
    tracker = HumanEgoHandTrackingPipeline.from_config_file(
        str(Path(hand_tracking_config_path).expanduser().resolve())
    )

    run_id = output.name or uuid.uuid4().hex
    manifest_path = output / "run.json"
    results_path = output / "results.jsonl"
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
    )

    total = sum(clip.frame_count or 0 for clip in selected_clips)
    input_frame_count = 0
    detected_hand_count = 0
    try:
        with results_path.open("w", encoding="utf-8", newline="\n") as stream:
            for bundle in preprocessing.iter_recorded_session(
                session_directory,
                clip_ids=selected_clip_ids,
            ):
                result = tracker.process_frame(bundle)
                stream.write(json.dumps(result.to_json_dict(), ensure_ascii=False) + "\n")
                stream.flush()
                input_frame_count += 1
                detected_hand_count += len(result.hands)
        completed_at_ns = time.time_ns()
        _write_log(log_path, "run completed")
        summary = {
            "run_id": run_id,
            "session_id": reader.session.session_id,
            "clip_id": clip_id,
            "state": "completed",
            "input_frame_count": input_frame_count,
            "expected_frame_count": total,
            "inferred_frame_count": input_frame_count,
            "detected_hand_count": detected_hand_count,
            "started_at_unix_ns": started_at_ns,
            "completed_at_unix_ns": completed_at_ns,
            "results_path": str(results_path),
        }
        _write_manifest(
            manifest_path,
            **summary,
            sensor_config_path=sensor_config_path,
            hand_tracking_config_path=hand_tracking_config_path,
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
            started_at_unix_ns=started_at_ns,
            completed_at_unix_ns=time.time_ns(),
            error=str(error),
            sensor_config_path=sensor_config_path,
            hand_tracking_config_path=hand_tracking_config_path,
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
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_log(path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{timestamp} {message}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process one EgoGlass capture session without Qt")
    parser.add_argument(
        "--session",
        type=Path,
        required=True,
        help="complete capture-session directory",
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
            clip_id=args.clip_id,
        )
    except Exception as error:
        print(f"EgoGlass offline processing failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
