"""Download pinned hand-tracking models without executing HumanEgo code."""

from __future__ import annotations

import argparse
from pathlib import Path

from perception.spatial_perception.hand_tracking import HandTrackingConfig
from perception.spatial_perception.hand_tracking.weights import (
    ensure_hand_tracking_weights,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/hand-tracking.yaml"),
    )
    args = parser.parse_args()
    config = HandTrackingConfig.load(args.config)
    weights = ensure_hand_tracking_weights(config)
    print(f"Hand-tracking model manifest: {weights.manifest}")


if __name__ == "__main__":
    main()
