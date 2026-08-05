"""Run one EgoGlass capture session through Basalt without Qt or the gateway."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from sensor_preprocessing import (
    CaptureSessionReader,
    SensorPreprocessingPipeline,
    derive_recorded_clock_mapping,
)
from slam_vio import BasaltVioConfig, BasaltVioRunner


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone VIO command-line interface."""

    parser = argparse.ArgumentParser(
        description="Run Basalt VIO for one complete EgoGlass capture session",
    )
    parser.add_argument("--session", type=Path, required=True, help="capture-session directory")
    parser.add_argument("--output", type=Path, required=True, help="new VIO run directory")
    parser.add_argument("--clip-id", help="process only one clip")
    parser.add_argument(
        "--sensor-config",
        type=Path,
        default=Path("config/sensor-preprocessing.yaml"),
    )
    parser.add_argument(
        "--basalt-config",
        type=Path,
        default=Path("config/basalt-vio.yaml"),
    )
    parser.add_argument(
        "--allow-unverified-calibration",
        action="store_true",
        help="allow sample/unmeasured calibration for integration tests",
    )
    return parser


def _sha256(path: Path) -> str:
    """Hash a configuration artifact recorded in the run manifest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_session(
    session_directory: str | Path,
    output_directory: str | Path,
    *,
    sensor_config_path: str | Path,
    basalt_config_path: str | Path,
    clip_id: str | None = None,
    allow_unverified_calibration: bool = False,
) -> dict[str, object]:
    """Prepare a session and return a JSON-serializable run summary."""

    session_path = Path(session_directory).resolve()
    output_path = Path(output_directory).resolve()
    reader = CaptureSessionReader.open(session_path)
    started_at_ns = time.time_ns()
    frame_evidence = tuple(
        frame for clip in reader.session.clips for frame in reader.iter_frames(clip.clip_id)
    )
    imu_evidence = tuple(reader.iter_imu_samples())
    mapping = derive_recorded_clock_mapping(
        reader.session.session_id,
        frame_evidence,
        imu_evidence,
    )
    preprocessing = SensorPreprocessingPipeline.from_config_file(sensor_config_path, mapping.mapper)
    config = BasaltVioConfig.load(basalt_config_path)
    if allow_unverified_calibration:
        config = config.model_copy(update={"allow_unverified_calibration": True})
    selected_clip_ids = None if clip_id is None else {clip_id}
    bundles = preprocessing.iter_recorded_session(session_path, clip_ids=selected_clip_ids)
    runner = BasaltVioRunner(config)
    result = runner.run(
        bundles,
        output_path,
        calibration=preprocessing.calibration,
    )
    summary = {
        "schema_version": "1.0",
        "contract_id": "basalt-vio-run-v1",
        "run_id": output_path.name,
        "session_id": reader.session.session_id,
        "clip_id": clip_id,
        "state": "completed",
        "started_at_unix_ns": started_at_ns,
        "completed_at_unix_ns": time.time_ns(),
        "frame_count": result.dataset.frame_count,
        "imu_count": result.dataset.imu_count,
        "interpolated_imu_count": result.dataset.interpolated_imu_count,
        "skipped_imu_timestamps": result.dataset.skipped_imu_timestamps,
        "trajectory_pose_count": len(result.trajectory.poses),
        "trajectory_path": str(result.trajectory_path),
        "dataset_path": str(result.dataset.root),
        "command": list(result.command),
        "calibration_profile_id": preprocessing.calibration.calibration_profile_id,
        "sensor_config_path": str(Path(sensor_config_path).resolve()),
        "sensor_config_sha256": _sha256(Path(sensor_config_path).resolve()),
        "basalt_config_path": str(Path(basalt_config_path).resolve()),
        "basalt_config_sha256": _sha256(Path(basalt_config_path).resolve()),
    }
    manifest_path = output_path / "run.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point returning a process exit status."""

    args = build_parser().parse_args(argv)
    try:
        summary = run_session(
            args.session,
            args.output,
            sensor_config_path=args.sensor_config,
            basalt_config_path=args.basalt_config,
            clip_id=args.clip_id,
            allow_unverified_calibration=args.allow_unverified_calibration,
        )
    except Exception as error:
        print(f"EgoGlass Basalt VIO failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
