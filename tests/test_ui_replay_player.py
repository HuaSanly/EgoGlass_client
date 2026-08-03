from __future__ import annotations

import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from ui.replay.player import (
    PlaybackFrame,
    ReplayPlayer,
    ReplayState,
    _clip_spans,
    _index_standalone_file,
    _IndexedClip,
    _IndexedFrame,
    _SessionDecoder,
    _SessionIndex,
    frame_time_seconds,
)


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
        assert isinstance(first.frame, PlaybackFrame)
        assert first.frame.session_id == "standalone"
        assert first.frame.clip_id == "clip"
        assert first.frame.pts_ns == first.frame.session_time_ns == 0
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


def test_session_decoder_crosses_clips_on_one_session_timeline(tmp_path: Path) -> None:
    first_path = tmp_path / "first.mp4"
    second_path = tmp_path / "second.mp4"
    _write_video(first_path)
    _write_video(second_path)
    references = _index_standalone_file(first_path).clips[0].frames
    second_references = tuple(
        _IndexedFrame(
            frame.frame_index,
            frame.pts,
            frame.time_base,
            1_000_000_000 + frame.session_time_ns,
        )
        for frame in _index_standalone_file(second_path).clips[0].frames
    )
    index = _SessionIndex(
        "session",
        tmp_path,
        (
            _IndexedClip("first", first_path, references),
            _IndexedClip("second", second_path, second_references),
        ),
        0,
        1_400_000_000,
    )
    decoder = _SessionDecoder(index)
    try:
        frames = [decoder.next() for _ in range(10)]
    finally:
        decoder.close()

    assert [frame.clip_id for frame in frames] == ["first"] * 5 + ["second"] * 5
    assert [frame.session_time_ns for frame in frames] == sorted(
        frame.session_time_ns for frame in frames
    )


def test_session_seek_opens_only_the_target_clip(tmp_path: Path) -> None:
    second_path = tmp_path / "second.mp4"
    _write_video(second_path)
    references = _index_standalone_file(second_path).clips[0].frames
    shifted = tuple(
        _IndexedFrame(
            frame.frame_index,
            frame.pts,
            frame.time_base,
            1_000_000_000 + frame.session_time_ns,
        )
        for frame in references
    )
    index = _SessionIndex(
        "session",
        tmp_path,
        (
            _IndexedClip("first", tmp_path / "missing-first.mp4", references),
            _IndexedClip("second", second_path, shifted),
        ),
        0,
        1_400_000_000,
    )
    decoder = _SessionDecoder(index)
    try:
        frame = decoder.seek(1_200_000_000)
    finally:
        decoder.close()

    assert frame.clip_id == "second"
    assert frame.session_time_ns == 1_200_000_000


def test_clip_spans_are_relative_to_the_complete_session_timeline(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    _write_video(path)
    frames = _index_standalone_file(path).clips[0].frames
    shifted = tuple(
        _IndexedFrame(
            frame.frame_index,
            frame.pts,
            frame.time_base,
            frame.session_time_ns + 1_000_000_000,
        )
        for frame in frames
    )
    index = _SessionIndex(
        "session",
        tmp_path,
        (_IndexedClip("first", path, frames), _IndexedClip("second", path, shifted)),
        0,
        1_400_000_000,
    )

    spans = _clip_spans(index)

    values = [
        (span.clip_id, span.start_seconds, span.end_seconds, span.frame_count)
        for span in spans
    ]
    assert values == [
        ("first", 0.0, 0.4, 5),
        ("second", 1.0, 1.4, 5),
    ]


def test_replay_opens_complete_session_at_requested_clip_and_unloads(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first_path = tmp_path / "first.mp4"
    second_path = tmp_path / "second.mp4"
    _write_video(first_path)
    _write_video(second_path)
    first = _index_standalone_file(first_path).clips[0].frames
    second = tuple(
        _IndexedFrame(
            frame.frame_index,
            frame.pts,
            frame.time_base,
            frame.session_time_ns + 1_000_000_000,
        )
        for frame in _index_standalone_file(second_path).clips[0].frames
    )
    index = _SessionIndex(
        "session",
        tmp_path,
        (_IndexedClip("first", first_path, first), _IndexedClip("second", second_path, second)),
        0,
        1_400_000_000,
    )
    monkeypatch.setattr("ui.replay.player._index_capture_session", lambda _path, _clip: index)
    player = ReplayPlayer()
    try:
        player.open_session(tmp_path, "second")
        _wait_for(
            player,
            lambda snapshot: snapshot.state is ReplayState.PAUSED
            and snapshot.frame is not None
            and snapshot.frame.clip_id == "second",
        )
        opened = player.snapshot()
        assert [span.clip_id for span in opened.clips] == ["first", "second"]
        assert opened.position_seconds == 1.0

        player.unload()
        _wait_for(player, lambda snapshot: snapshot.state is ReplayState.EMPTY)
        unloaded = player.snapshot()
        assert unloaded.path is None
        assert unloaded.frame is None
        assert unloaded.clips == ()

        player.open(first_path)
        _wait_for(
            player,
            lambda snapshot: snapshot.state is ReplayState.PAUSED and snapshot.frame is not None,
        )
    finally:
        player.close()
