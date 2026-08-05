"""Download pinned hand-tracking models without executing HumanEgo code."""

from __future__ import annotations

import argparse
from pathlib import Path

from hand_tracking import HandTrackingConfig
from hand_tracking.weights import (
    ensure_hand_tracking_weights,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        help="Hand-tracking YAML to prepare; repeat for multiple profiles",
    )
    args = parser.parse_args()
    paths = args.config or (
        Path("config/live-hand-tracking.yaml"),
        Path("config/offline-hand-tracking.yaml"),
    )
    for path in paths:
        config = HandTrackingConfig.load(path)
        weights = ensure_hand_tracking_weights(config)
        print(f"{path}: {weights.manifest}")


if __name__ == "__main__":
    main()
