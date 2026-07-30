from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from pathlib import Path

import av

from ingest_gateway.live_frames import LiveFrame


class ReplayState(StrEnum):
    EMPTY = "empty"
    LOADING = "loading"
    PAUSED = "paused"
    PLAYING = "playing"
    ENDED = "ended"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ReplaySnapshot:
    revision: int = 0
    state: ReplayState = ReplayState.EMPTY
    path: Path | None = None
    duration_seconds: float = 0.0
    position_seconds: float = 0.0
    playback_rate: float = 1.0
    frame: LiveFrame | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _Command:
    name: str
    value: object | None = None


class ReplayPlayer:
    """Decode one local MP4 on a worker and schedule frames from their PTS."""

    def __init__(self) -> None:
        self._commands: queue.Queue[_Command] = queue.Queue()
        self._lock = threading.Lock()
        self._snapshot = ReplaySnapshot()
        self._thread = threading.Thread(target=self._run, name="replay-player", daemon=False)
        self._thread.start()

    def open(self, path: Path) -> None:
        self._commands.put(_Command("open", path.expanduser().resolve()))

    def play(self) -> None:
        self._commands.put(_Command("play"))

    def pause(self) -> None:
        self._commands.put(_Command("pause"))

    def step(self) -> None:
        self._commands.put(_Command("step"))

    def seek(self, seconds: float) -> None:
        self._commands.put(_Command("seek", max(0.0, float(seconds))))

    def set_playback_rate(self, rate: float) -> None:
        if rate not in {0.25, 0.5, 1.0, 1.5, 2.0}:
            raise ValueError("unsupported playback rate")
        self._commands.put(_Command("rate", rate))

    def snapshot(self) -> ReplaySnapshot:
        with self._lock:
            return self._snapshot

    def close(self, timeout_seconds: float = 5.0) -> None:
        if not self._thread.is_alive():
            return
        self._commands.put(_Command("close"))
        self._thread.join(timeout_seconds)
        if self._thread.is_alive():
            raise TimeoutError("replay worker did not stop")

    def _run(self) -> None:
        container: av.InputContainer | None = None
        stream: av.VideoStream | None = None
        decoder = None
        playing = False
        step_requested = False
        wall_anchor_ns: int | None = None
        media_anchor_seconds = 0.0
        try:
            while True:
                if container is None or decoder is None or (not playing and not step_requested):
                    command = self._commands.get()
                    if command.name == "close":
                        return
                    try:
                        container, stream, decoder, playing, step_requested = (
                            self._apply_command(
                                command,
                                container,
                                stream,
                                decoder,
                                playing,
                                step_requested,
                            )
                        )
                    except Exception as error:
                        if container is not None:
                            container.close()
                        container = stream = decoder = None
                        playing = step_requested = False
                        self._update(state=ReplayState.ERROR, error=str(error))
                    wall_anchor_ns = None
                    continue
                try:
                    frame = next(decoder)
                except StopIteration:
                    playing = False
                    step_requested = False
                    self._update(state=ReplayState.ENDED)
                    continue
                except Exception as error:
                    container.close()
                    container = stream = decoder = None
                    playing = step_requested = False
                    self._update(state=ReplayState.ERROR, error=str(error))
                    continue
                position_seconds = frame_time_seconds(frame)
                if playing:
                    snapshot = self.snapshot()
                    if wall_anchor_ns is None:
                        wall_anchor_ns = time.perf_counter_ns()
                        media_anchor_seconds = position_seconds
                    target_ns = wall_anchor_ns + round(
                        (position_seconds - media_anchor_seconds)
                        * 1_000_000_000
                        / snapshot.playback_rate
                    )
                    interrupted = self._wait_until(target_ns)
                    if interrupted is not None:
                        if interrupted.name == "close":
                            return
                        try:
                            container, stream, decoder, playing, step_requested = (
                                self._apply_command(
                                    interrupted,
                                    container,
                                    stream,
                                    decoder,
                                    playing,
                                    step_requested,
                                )
                            )
                        except Exception as error:
                            container.close()
                            container = stream = decoder = None
                            playing = step_requested = False
                            self._update(state=ReplayState.ERROR, error=str(error))
                        wall_anchor_ns = None
                        continue
                self._publish_frame(frame, position_seconds)
                if step_requested:
                    step_requested = False
                    self._update(state=ReplayState.PAUSED)
        except Exception as error:
            self._update(state=ReplayState.ERROR, error=str(error))
        finally:
            if container is not None:
                container.close()

    def _apply_command(
        self,
        command: _Command,
        container: av.InputContainer | None,
        stream: av.VideoStream | None,
        decoder: object,
        playing: bool,
        step_requested: bool,
    ) -> tuple[av.InputContainer | None, av.VideoStream | None, object, bool, bool]:
        if command.name == "open":
            if container is not None:
                container.close()
            path = command.value
            if not isinstance(path, Path) or not path.is_file():
                raise FileNotFoundError(path)
            self._update(
                state=ReplayState.LOADING,
                path=path,
                position_seconds=0.0,
                frame=None,
                error=None,
            )
            container = av.open(str(path))
            stream = container.streams.video[0]
            duration = stream_duration_seconds(container, stream)
            decoder = container.decode(stream)
            self._update(
                state=ReplayState.PAUSED,
                duration_seconds=duration,
            )
            return container, stream, decoder, False, True
        if command.name == "play" and container is not None:
            if self.snapshot().state is ReplayState.ENDED:
                assert stream is not None
                container.seek(0, stream=stream, backward=True)
                decoder = container.decode(stream)
            self._update(state=ReplayState.PLAYING)
            return container, stream, decoder, True, False
        if command.name == "pause":
            self._update(state=ReplayState.PAUSED)
            return container, stream, decoder, False, False
        if command.name == "step" and container is not None:
            return container, stream, decoder, False, True
        if command.name == "rate":
            rate = float(command.value)
            self._update(playback_rate=rate)
            return container, stream, decoder, playing, step_requested
        if command.name == "seek" and container is not None and stream is not None:
            target = min(float(command.value), self.snapshot().duration_seconds)
            offset = round(target / float(stream.time_base))
            container.seek(offset, stream=stream, backward=True)
            decoder = container.decode(stream)
            for candidate in decoder:
                if frame_time_seconds(candidate) + 1e-9 >= target:
                    self._publish_frame(candidate, frame_time_seconds(candidate))
                    break
            self._update(state=ReplayState.PLAYING if playing else ReplayState.PAUSED)
            return container, stream, decoder, playing, False
        return container, stream, decoder, playing, step_requested

    def _wait_until(self, target_ns: int) -> _Command | None:
        while True:
            remaining_seconds = (target_ns - time.perf_counter_ns()) / 1_000_000_000
            if remaining_seconds <= 0:
                return None
            try:
                return self._commands.get(timeout=remaining_seconds)
            except queue.Empty:
                return None

    def _publish_frame(self, frame: av.VideoFrame, position_seconds: float) -> None:
        image_rgb = frame.to_ndarray(format="rgb24")
        image_rgb.setflags(write=False)
        snapshot = self.snapshot()
        index = 0 if snapshot.frame is None else snapshot.frame.frame_index + 1
        now_ns = time.perf_counter_ns()
        display_frame = LiveFrame(
            session_id="replay",
            connection_session_id=str(snapshot.path),
            frame_index=index,
            received_at_client_monotonic_ns=now_ns,
            converted_at_client_monotonic_ns=now_ns,
            image_rgb=image_rgb,
        )
        self._update(position_seconds=position_seconds, frame=display_frame)

    def _update(self, **changes: object) -> None:
        with self._lock:
            self._snapshot = ReplaySnapshot(
                **{
                    **{
                        field: getattr(self._snapshot, field)
                        for field in ReplaySnapshot.__dataclass_fields__
                    },
                    **changes,
                    "revision": self._snapshot.revision + 1,
                }
            )


def frame_time_seconds(frame: av.VideoFrame) -> float:
    if frame.pts is None or frame.time_base is None:
        raise ValueError("replay frame has no presentation timestamp")
    return float(Fraction(frame.pts) * Fraction(frame.time_base))


def stream_duration_seconds(container: av.InputContainer, stream: av.VideoStream) -> float:
    if stream.duration is not None and stream.time_base is not None:
        return float(Fraction(stream.duration) * Fraction(stream.time_base))
    if container.duration is not None:
        return container.duration / av.time_base
    return 0.0
