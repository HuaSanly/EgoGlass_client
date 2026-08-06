"""Measure replay frame continuity without starting Qt or writing derived data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

CLIENT_ROOT = Path(__file__).resolve().parents[1]
if str(CLIENT_ROOT) not in sys.path:
    sys.path.insert(0, str(CLIENT_ROOT))

from ui.replay.player import ReplayPlayer, ReplayState  # noqa: E402


def run(session: Path, sensor_config: Path, duration_seconds: float) -> dict[str, object]:
    player = ReplayPlayer(sensor_config)
    player.open_session(session)
    deadline = time.perf_counter() + duration_seconds
    while time.perf_counter() < deadline:
        snapshot = player.snapshot()
        if snapshot.state in {ReplayState.PAUSED, ReplayState.PLAYING, ReplayState.ENDED}:
            break
        if snapshot.state is ReplayState.ERROR:
            player.close()
            return {"error": snapshot.error or "replay failed before playback"}
        time.sleep(0.01)

    player.play()
    first_session_time_ns: int | None = None
    last_session_time_ns: int | None = None
    last_frame_key: tuple[str, int, int] | None = None
    frame_keys: set[tuple[str, int, int]] = set()
    presented_frames = 0
    error: str | None = None
    rewound = False
    start = time.perf_counter()
    while time.perf_counter() - start < duration_seconds:
        snapshot = player.snapshot()
        if snapshot.state is ReplayState.ERROR:
            error = snapshot.error or "replay failed"
            break
        frame = snapshot.frame
        if frame is not None:
            if first_session_time_ns is None:
                first_session_time_ns = frame.session_time_ns
            if rewound:
                rewound = False
                last_session_time_ns = None
            if last_session_time_ns is not None and frame.session_time_ns < last_session_time_ns:
                error = "session time moved backwards"
                break
            last_session_time_ns = frame.session_time_ns
            frame_key = (frame.clip_id, frame.frame_index, frame.session_time_ns)
            if frame_key != last_frame_key:
                presented_frames += 1
                last_frame_key = frame_key
                frame_keys.add(frame_key)
        if snapshot.state is ReplayState.ENDED:
            rewound = True
            player.seek(0.0)
            player.play()
        time.sleep(0.005)
    elapsed = max(1e-9, time.perf_counter() - start)
    player.close()
    return {
        "session": str(session.resolve()),
        "duration_seconds": round(elapsed, 3),
        "unique_frames": len(frame_keys),
        "displayed_frames": presented_frames,
        "display_fps": round(presented_frames / elapsed, 3),
        "first_session_time_ns": first_session_time_ns,
        "last_session_time_ns": last_session_time_ns,
        "error": error,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark a capture-session replay timeline")
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument(
        "--sensor-config",
        type=Path,
        default=Path("config/sensor-preprocessing.yaml"),
    )
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    print(json.dumps(run(args.session, args.sensor_config, args.seconds), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
