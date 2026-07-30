from __future__ import annotations

import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from ui.replay.player import ReplayPlayer, ReplayState, frame_time_seconds


def _write_video(path: Path) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 10)
        for index in range(5):
            image = np.full((48, 64, 3), index * 30, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _wait_for(player: ReplayPlayer, predicate: object) -> None:
    check = predicate
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if callable(check) and check(player.snapshot()):
            return
        time.sleep(0.005)
    raise AssertionError(f"replay state did not converge: {player.snapshot()}")


def test_replay_player_decodes_first_frame_and_runs_to_end(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    _write_video(path)
    player = ReplayPlayer()
    try:
        player.open(path)
        _wait_for(
            player,
            lambda snapshot: snapshot.state is ReplayState.PAUSED and snapshot.frame is not None,
        )
        first = player.snapshot()
        assert first.path == path
        assert first.frame is not None
        assert first.frame.image_rgb.shape == (48, 64, 3)
        assert not first.frame.image_rgb.flags.writeable
        player.play()
        _wait_for(player, lambda snapshot: snapshot.state is ReplayState.ENDED)
        assert player.snapshot().position_seconds >= 0.3
    finally:
        player.close()


def test_frame_time_requires_pts_and_time_base() -> None:
    frame = av.VideoFrame(width=2, height=2, format="rgb24")
    frame.pts = 17
    frame.time_base = Fraction(1, 100)

    assert frame_time_seconds(frame) == 0.17


def test_replay_worker_recovers_after_invalid_file(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    _write_video(path)
    player = ReplayPlayer()
    try:
        player.open(tmp_path / "missing.mp4")
        _wait_for(player, lambda snapshot: snapshot.state is ReplayState.ERROR)

        player.open(path)
        _wait_for(
            player,
            lambda snapshot: snapshot.state is ReplayState.PAUSED and snapshot.frame is not None,
        )

        assert player.snapshot().path == path
        assert player.snapshot().error is None
    finally:
        player.close()
