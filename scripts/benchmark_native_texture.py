from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
CLIENT_ROOT = Path(__file__).resolve().parents[1]
if str(CLIENT_ROOT) not in sys.path:
    sys.path.insert(0, str(CLIENT_ROOT))

from PyQt6.QtWidgets import QApplication  # noqa: E402

from ui.gateway.live_frames import LiveFrame  # noqa: E402
from ui.widgets.video_canvas import VideoCanvas  # noqa: E402


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def run_benchmark(width: int, height: int, frame_count: int) -> dict[str, float]:
    application = QApplication.instance() or QApplication([])
    canvas = VideoCanvas()
    canvas.resize(960, 720)
    image = np.random.default_rng(7).integers(
        0,
        256,
        size=(height, width, 3),
        dtype=np.uint8,
    )
    image.setflags(write=False)
    submit_ms: list[float] = []
    frame_ms: list[float] = []
    for frame_index in range(frame_count):
        started_at_ns = time.perf_counter_ns()
        frame = LiveFrame(
            session_id="benchmark",
            connection_session_id="benchmark",
            frame_index=frame_index,
            received_at_client_monotonic_ns=started_at_ns,
            converted_at_client_monotonic_ns=started_at_ns,
            image_rgb=image,
        )
        canvas.set_frame(frame)
        submitted_at_ns = time.perf_counter_ns()
        canvas.grab()
        application.processEvents()
        finished_at_ns = time.perf_counter_ns()
        if frame_index >= 10:
            submit_ms.append((submitted_at_ns - started_at_ns) / 1_000_000)
            frame_ms.append((finished_at_ns - started_at_ns) / 1_000_000)
    return {
        "width": float(width),
        "height": float(height),
        "frames": float(len(frame_ms)),
        "submit_mean_ms": statistics.fmean(submit_ms),
        "submit_p95_ms": percentile(submit_ms, 0.95),
        "frame_mean_ms": statistics.fmean(frame_ms),
        "frame_p95_ms": percentile(frame_ms, 0.95),
        "effective_fps": 1000.0 / statistics.fmean(frame_ms),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark PyQt RGB canvas updates")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=310)
    args = parser.parse_args()
    if args.width < 1 or args.height < 1 or args.frames < 20:
        parser.error("width, height, and frame count must be positive; use at least 20 frames")
    report = run_benchmark(args.width, args.height, args.frames)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
