"""Entry points for offline spatial-perception replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime import render_recorded_hand_tracking_replay
from .spatial_perception.hand_tracking import HumanEgoHandTrackingPipeline


def main() -> None:
    """Run a stored-session hand-tracking replay from the shared Conda environment."""

    parser = argparse.ArgumentParser(description="EgoGlass perception tools")
    subcommands = parser.add_subparsers(dest="command", required=True)
    replay = subcommands.add_parser("hand-replay", help="render an annotated recorded session")
    replay.add_argument("--session-directory", type=Path, required=True)
    replay.add_argument("--output-directory", type=Path, required=True)
    replay.add_argument(
        "--sensor-config",
        type=Path,
        default=Path("config/sensor-preprocessing.yaml"),
    )
    replay.add_argument("--hand-config", type=Path, default=Path("config/hand-tracking.yaml"))
    replay.add_argument("--inference-stride-frames", type=int, default=5)
    args = parser.parse_args()

    if args.command == "hand-replay":
        tracker = HumanEgoHandTrackingPipeline.from_config_file(str(args.hand_config))
        report = render_recorded_hand_tracking_replay(
            args.session_directory,
            args.output_directory,
            sensor_config_path=args.sensor_config,
            tracker=tracker,
            inference_stride_frames=args.inference_stride_frames,
        )
        print(
            json.dumps(
                report.to_json_dict(args.session_directory.parent.resolve()),
                ensure_ascii=False,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
