from __future__ import annotations

import queue
import threading
import time
from bisect import bisect_left
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray

from perception.sensor_preprocessing import (
    AlignmentStatus,
    CaptureSessionReader,
    RawFrameRef,
    RecordedImuPose,
    RecordedImuPoseTimeline,
    build_recorded_imu_pose_timeline,
    derive_recorded_clock_mapping,
    frame_presentation_observation,
)


class ReplayState(StrEnum):
    EMPTY = "empty"
    LOADING = "loading"
    PAUSED = "paused"
    PLAYING = "playing"
    ENDED = "ended"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class PlaybackFrame:
    session_id: str
    clip_id: str
    frame_index: int
    pts_ns: int
    session_time_ns: int
    image_rgb: NDArray[np.uint8]

    @property
    def width(self) -> int:
        return int(self.image_rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.image_rgb.shape[0])


@dataclass(frozen=True, slots=True)
class PlaybackClipSpan:
    clip_id: str
    start_seconds: float
    end_seconds: float
    frame_count: int


@dataclass(frozen=True, slots=True)
class ReplaySnapshot:
    revision: int = 0
    state: ReplayState = ReplayState.EMPTY
    path: Path | None = None
    session_id: str | None = None
    clip_id: str | None = None
    duration_seconds: float = 0.0
    position_seconds: float = 0.0
    session_time_ns: int | None = None
    playback_rate: float = 1.0
    clips: tuple[PlaybackClipSpan, ...] = ()
    frame: PlaybackFrame | None = None
    imu_pose: RecordedImuPose | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _IndexedFrame:
    frame_index: int
    pts: int
    time_base: Fraction
    session_time_ns: int


@dataclass(frozen=True, slots=True)
class _IndexedClip:
    clip_id: str
    media_path: Path
    frames: tuple[_IndexedFrame, ...]


@dataclass(frozen=True, slots=True)
class _SessionIndex:
    session_id: str
    root_path: Path
    clips: tuple[_IndexedClip, ...]
    origin_session_time_ns: int
    duration_ns: int
    imu_timeline: RecordedImuPoseTimeline | None = None


@dataclass(frozen=True, slots=True)
class _Command:
    name: str
    value: object | None = None


class _SessionDecoder:
    def __init__(self, index: _SessionIndex) -> None:
        self.index = index
        self.clip_cursor = -1
        self.frame_cursor = 0
        self.container: av.InputContainer | None = None
        self.frames = None

    def close(self) -> None:
        if self.container is not None:
            self.container.close()
        self.container = None
        self.frames = None

    def reset(self) -> None:
        self.close()
        self.clip_cursor = -1
        self.frame_cursor = 0

    def next(self) -> PlaybackFrame:
        while True:
            if self.container is None or self.frames is None:
                self._open_next_clip()
            clip = self.index.clips[self.clip_cursor]
            try:
                decoded = next(self.frames)
            except StopIteration:
                self.close()
                continue
            if self.frame_cursor >= len(clip.frames):
                raise ValueError("decoded frame count exceeds capture index")
            reference = clip.frames[self.frame_cursor]
            self.frame_cursor += 1
            if decoded.pts != reference.pts or decoded.time_base != reference.time_base:
                raise ValueError("decoded frame PTS does not match capture index")
            return self._playback_frame(decoded, clip, reference)

    def seek(self, session_time_ns: int) -> PlaybackFrame:
        clip_index = next(
            (
                index
                for index, clip in enumerate(self.index.clips)
                if clip.frames[-1].session_time_ns >= session_time_ns
            ),
            len(self.index.clips) - 1,
        )
        clip = self.index.clips[clip_index]
        target_index = bisect_left(
            tuple(frame.session_time_ns for frame in clip.frames),
            session_time_ns,
        )
        target_index = min(target_index, len(clip.frames) - 1)
        target = clip.frames[target_index]
        self.close()
        self.clip_cursor = clip_index
        self.frame_cursor = 0
        self._open_clip(clip)
        assert self.container is not None
        stream = self.container.streams.video[0]
        self.container.seek(target.pts, stream=stream, backward=True)
        decoder = self.container.decode(stream)
        reference_by_pts = {frame.pts: index for index, frame in enumerate(clip.frames)}
        for decoded in decoder:
            reference_index = reference_by_pts.get(decoded.pts)
            if reference_index is None or reference_index < target_index:
                continue
            reference = clip.frames[reference_index]
            if decoded.time_base != reference.time_base:
                raise ValueError("decoded seek frame time base does not match capture index")
            self.frames = decoder
            self.frame_cursor = reference_index + 1
            return self._playback_frame(decoded, clip, reference)
        raise ValueError("decoder did not produce the requested capture frame")

    def _open_next_clip(self) -> None:
        self.clip_cursor += 1
        if self.clip_cursor >= len(self.index.clips):
            raise StopIteration
        clip = self.index.clips[self.clip_cursor]
        self._open_clip(clip)

    def _open_clip(self, clip: _IndexedClip) -> None:
        self.frame_cursor = 0
        self.container = av.open(str(clip.media_path), mode="r")
        stream = self.container.streams.video[0]
        self.frames = self.container.decode(stream)

    def _playback_frame(
        self,
        decoded: av.VideoFrame,
        clip: _IndexedClip,
        reference: _IndexedFrame,
    ) -> PlaybackFrame:
        image_rgb = np.ascontiguousarray(decoded.to_ndarray(format="rgb24"))
        image_rgb.setflags(write=False)
        return PlaybackFrame(
            session_id=self.index.session_id,
            clip_id=clip.clip_id,
            frame_index=reference.frame_index,
            pts_ns=_fraction_ns(Fraction(reference.pts) * reference.time_base),
            session_time_ns=reference.session_time_ns,
            image_rgb=image_rgb,
        )


class ReplayPlayer:
    """Decode a capture session on its PTS-derived session timeline."""

    def __init__(self) -> None:
        self._commands: queue.Queue[_Command] = queue.Queue()
        self._lock = threading.Lock()
        self._snapshot = ReplaySnapshot()
        self._thread = threading.Thread(target=self._run, name="replay-player", daemon=False)
        self._thread.start()

    def open(self, path: Path) -> None:
        """Open one standalone MP4 while still producing a typed PlaybackFrame."""

        self._commands.put(_Command("open-file", path.expanduser().resolve()))

    def open_session(
        self,
        session_directory: Path,
        initial_clip_id: str | None = None,
    ) -> None:
        self._commands.put(
            _Command(
                "open-session",
                (session_directory.expanduser().resolve(), initial_clip_id),
            )
        )

    def unload(self) -> None:
        """Release the active decoder while keeping the replay worker reusable."""

        self._commands.put(_Command("unload"))

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
        decoder: _SessionDecoder | None = None
        playing = False
        step_requested = False
        wall_anchor_ns: int | None = None
        media_anchor_ns = 0
        try:
            while True:
                if decoder is None or (not playing and not step_requested):
                    command = self._commands.get()
                    if command.name == "close":
                        return
                    decoder, playing, step_requested = self._apply_command(
                        command, decoder, playing, step_requested
                    )
                    wall_anchor_ns = None
                    continue
                try:
                    frame = decoder.next()
                except StopIteration:
                    playing = step_requested = False
                    self._update(state=ReplayState.ENDED)
                    continue
                if playing:
                    if wall_anchor_ns is None:
                        wall_anchor_ns = time.perf_counter_ns()
                        media_anchor_ns = frame.session_time_ns
                    target_ns = wall_anchor_ns + round(
                        (frame.session_time_ns - media_anchor_ns)
                        / self.snapshot().playback_rate
                    )
                    command = self._wait_until(target_ns)
                    if command is not None:
                        if command.name == "close":
                            return
                        decoder, playing, step_requested = self._apply_command(
                            command, decoder, playing, step_requested
                        )
                        wall_anchor_ns = None
                        continue
                self._publish_frame(frame, decoder.index)
                if step_requested:
                    step_requested = False
                    self._update(state=ReplayState.PAUSED)
        except Exception as error:
            self._update(state=ReplayState.ERROR, error=str(error))
        finally:
            if decoder is not None:
                decoder.close()

    def _apply_command(
        self,
        command: _Command,
        decoder: _SessionDecoder | None,
        playing: bool,
        step_requested: bool,
    ) -> tuple[_SessionDecoder | None, bool, bool]:
        try:
            if command.name == "unload":
                if decoder is not None:
                    decoder.close()
                self._reset_snapshot()
                return None, False, False
            if command.name in {"open-file", "open-session"}:
                if decoder is not None:
                    decoder.close()
                self._update(state=ReplayState.LOADING, frame=None, error=None)
                if command.name == "open-file":
                    assert isinstance(command.value, Path)
                    index = _index_standalone_file(command.value)
                else:
                    value = command.value
                    if not isinstance(value, tuple) or len(value) != 2:
                        raise TypeError("invalid session-open command")
                    path, initial_clip_id = value
                    if not isinstance(path, Path) or (
                        initial_clip_id is not None and not isinstance(initial_clip_id, str)
                    ):
                        raise TypeError("invalid session-open command")
                    index = _index_capture_session(path, None)
                decoder = _SessionDecoder(index)
                self._update(
                    state=ReplayState.PAUSED,
                    path=index.root_path,
                    session_id=index.session_id,
                    clip_id=(index.clips[0].clip_id if len(index.clips) == 1 else None),
                    duration_seconds=index.duration_ns / 1_000_000_000,
                    position_seconds=0.0,
                    clips=_clip_spans(index),
                    imu_pose=None,
                )
                if command.name == "open-session" and initial_clip_id is not None:
                    target = next(
                        (clip for clip in index.clips if clip.clip_id == initial_clip_id),
                        None,
                    )
                    if target is None:
                        raise KeyError(f"unknown complete clip {initial_clip_id!r}")
                    frame = decoder.seek(target.frames[0].session_time_ns)
                    self._publish_frame(frame, index)
                    self._update(state=ReplayState.PAUSED)
                    return decoder, False, False
                return decoder, False, True
            if command.name == "play" and decoder is not None:
                if self.snapshot().state is ReplayState.ENDED:
                    decoder.reset()
                self._update(state=ReplayState.PLAYING)
                return decoder, True, False
            if command.name == "pause":
                self._update(state=ReplayState.PAUSED)
                return decoder, False, False
            if command.name == "step" and decoder is not None:
                return decoder, False, True
            if command.name == "rate":
                self._update(playback_rate=float(command.value))
                return decoder, playing, step_requested
            if command.name == "seek" and decoder is not None:
                seconds = min(float(command.value), self.snapshot().duration_seconds)
                target_ns = decoder.index.origin_session_time_ns + round(seconds * 1_000_000_000)
                frame = decoder.seek(target_ns)
                self._publish_frame(frame, decoder.index)
                self._update(state=ReplayState.PLAYING if playing else ReplayState.PAUSED)
                return decoder, playing, False
            return decoder, playing, step_requested
        except Exception as error:
            if decoder is not None:
                decoder.close()
            self._update(state=ReplayState.ERROR, error=str(error))
            return None, False, False

    def _wait_until(self, target_ns: int) -> _Command | None:
        remaining = (target_ns - time.perf_counter_ns()) / 1_000_000_000
        if remaining <= 0:
            return None
        try:
            return self._commands.get(timeout=remaining)
        except queue.Empty:
            return None

    def _publish_frame(self, frame: PlaybackFrame, index: _SessionIndex) -> None:
        imu_pose = (
            index.imu_timeline.pose_at(frame.session_time_ns)
            if index.imu_timeline is not None
            else None
        )
        self._update(
            position_seconds=(
                frame.session_time_ns - index.origin_session_time_ns
            )
            / 1_000_000_000,
            session_time_ns=frame.session_time_ns,
            clip_id=frame.clip_id,
            frame=frame,
            imu_pose=imu_pose,
        )

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

    def _reset_snapshot(self) -> None:
        with self._lock:
            self._snapshot = ReplaySnapshot(revision=self._snapshot.revision + 1)


def _clip_spans(index: _SessionIndex) -> tuple[PlaybackClipSpan, ...]:
    return tuple(
        PlaybackClipSpan(
            clip.clip_id,
            (clip.frames[0].session_time_ns - index.origin_session_time_ns) / 1_000_000_000,
            (clip.frames[-1].session_time_ns - index.origin_session_time_ns) / 1_000_000_000,
            len(clip.frames),
        )
        for clip in index.clips
    )


def _index_capture_session(session_path: Path, clip_id: str | None) -> _SessionIndex:
    reader = CaptureSessionReader.open(session_path, verify_media_hashes=False)
    all_frames = tuple(
        frame
        for clip in reader.session.clips
        for frame in reader.iter_frames(clip.clip_id)
    )
    if not all_frames:
        raise ValueError("capture session contains no indexed frames")
    imu_samples = tuple(reader.iter_imu_samples())
    mapper = None
    if any(
        item.stored_alignment.status is AlignmentStatus.PENDING
        for item in (*all_frames, *imu_samples)
    ):
        mapper = derive_recorded_clock_mapping(
            reader.session.session_id,
            all_frames,
            imu_samples,
        ).mapper
    selected_clips = tuple(
        clip for clip in reader.session.clips if clip_id is None or clip.clip_id == clip_id
    )
    if not selected_clips:
        raise KeyError(f"unknown complete clip {clip_id!r}")
    frames_by_clip: dict[str, list[RawFrameRef]] = {}
    for frame in all_frames:
        frames_by_clip.setdefault(frame.clip_id, []).append(frame)
    clips: list[_IndexedClip] = []
    for clip in selected_clips:
        indexed: list[_IndexedFrame] = []
        for frame in frames_by_clip[clip.clip_id]:
            session_time_ns = frame.stored_alignment.session_time_ns
            if session_time_ns is None:
                assert mapper is not None
                estimate = mapper.map(frame_presentation_observation(frame))
                if estimate.session_time_ns is None:
                    raise ValueError("capture frame has no session-time mapping")
                session_time_ns = estimate.session_time_ns
            indexed.append(
                _IndexedFrame(
                    frame.frame_index,
                    frame.mp4_timestamp.pts,
                    Fraction(
                        frame.mp4_timestamp.time_base_numerator,
                        frame.mp4_timestamp.time_base_denominator,
                    ),
                    session_time_ns,
                )
            )
        clips.append(_IndexedClip(clip.clip_id, clip.media_path, tuple(indexed)))
    clips.sort(key=lambda item: item.frames[0].session_time_ns)
    first_ns = clips[0].frames[0].session_time_ns
    last_ns = clips[-1].frames[-1].session_time_ns
    imu_timeline = build_recorded_imu_pose_timeline(
        reader.session.session_id,
        imu_samples,
        mapper,
    )
    return _SessionIndex(
        reader.session.session_id,
        session_path,
        tuple(clips),
        first_ns,
        last_ns - first_ns,
        imu_timeline,
    )


def _index_standalone_file(path: Path) -> _SessionIndex:
    if not path.is_file():
        raise FileNotFoundError(path)
    frames: list[_IndexedFrame] = []
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("replay frame has no presentation timestamp")
            pts_ns = _fraction_ns(Fraction(frame.pts) * Fraction(frame.time_base))
            frames.append(
                _IndexedFrame(index, frame.pts, Fraction(frame.time_base), pts_ns)
            )
    if not frames:
        raise ValueError("replay video contains no frames")
    clip = _IndexedClip(path.stem, path, tuple(frames))
    return _SessionIndex(
        "standalone",
        path,
        (clip,),
        frames[0].session_time_ns,
        frames[-1].session_time_ns - frames[0].session_time_ns,
    )


def frame_time_seconds(frame: av.VideoFrame) -> float:
    if frame.pts is None or frame.time_base is None:
        raise ValueError("replay frame has no presentation timestamp")
    return float(Fraction(frame.pts) * Fraction(frame.time_base))


def _fraction_ns(value: Fraction) -> int:
    return round(value * 1_000_000_000)
