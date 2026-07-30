from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

import numpy as np
from av import VideoFrame

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LiveFrame:
    """One immutable RGB frame ready for the native UI."""

    session_id: str
    connection_session_id: str
    frame_index: int
    received_at_client_monotonic_ns: int
    converted_at_client_monotonic_ns: int
    image_rgb: np.ndarray
    video_pts_ns: int | None = None

    @property
    def width(self) -> int:
        return int(self.image_rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.image_rgb.shape[0])


@dataclass(frozen=True, slots=True)
class LiveFrameStatus:
    """Bounded-buffer counters used by the UI diagnostics view."""

    frames_received: int
    frames_converted: int
    pending_frames_overwritten: int
    conversion_failures: int
    rgb_frames_forwarded: int
    rgb_sink_failures: int
    presentation_queue_depth: int
    presentation_frames_dropped: int
    presentation_starvations: int
    presentation_interval_ms: float
    display_poll_fps: float
    presentation_fps: float
    presentation_frames_presented: int
    conversion_gap_p95_ms: float
    conversion_gap_max_ms: float
    source_pts_gap_p95_ms: float
    source_pts_gap_max_ms: float
    latest_frame_index: int | None
    latest_width: int | None
    latest_height: int | None
    recent_fps: float
    latest_conversion_ms: float | None
    latest_frame_age_ms: float | None
    last_error: str | None


@dataclass(frozen=True, slots=True)
class _PendingFrame:
    session_id: str
    connection_session_id: str
    frame_index: int
    received_at_client_monotonic_ns: int
    decoded_frame: VideoFrame
    video_pts_ns: int | None


@dataclass(frozen=True, slots=True)
class LiveFramePacerStatus:
    queue_depth: int
    frames_dropped: int
    starvations: int
    interval_ms: float
    frames_presented: int
    recent_fps: float
    source_gap_p95_ms: float
    source_gap_max_ms: float


class LiveFramePacer:
    """Absorb short receive bursts with three frames of bounded presentation delay."""

    def __init__(
        self,
        *,
        maximum_queue_frames: int = 4,
        target_queue_frames: int = 4,
        startup_wait_ns: int = 110_000_000,
        default_interval_ns: int = 33_333_333,
    ) -> None:
        if maximum_queue_frames < 2:
            raise ValueError("maximum_queue_frames must be at least two")
        if target_queue_frames not in range(1, maximum_queue_frames + 1):
            raise ValueError("target_queue_frames must fit inside the queue")
        if startup_wait_ns < 0 or default_interval_ns <= 0:
            raise ValueError("presentation timing must be non-negative")
        self._maximum_queue_frames = maximum_queue_frames
        self._target_queue_frames = target_queue_frames
        self._startup_wait_ns = startup_wait_ns
        self._default_interval_ns = default_interval_ns
        self._queue: deque[LiveFrame] = deque()
        self._source_pts_ns: deque[int] = deque(maxlen=31)
        self._stream_key: tuple[str, str] | None = None
        self._last_enqueued_pts_ns: int | None = None
        self._presented: LiveFrame | None = None
        self._next_due_ns: int | None = None
        self._frames_dropped = 0
        self._starvations = 0
        self._starved = False
        self._frames_presented = 0
        self._presented_at_ns: deque[int] = deque(maxlen=240)

    def enqueue(self, frame: LiveFrame) -> None:
        stream_key = (frame.session_id, frame.connection_session_id)
        if stream_key != self._stream_key:
            self._reset_stream(stream_key)
        if frame.video_pts_ns is not None:
            if (
                self._last_enqueued_pts_ns is not None
                and (
                    frame.video_pts_ns < self._last_enqueued_pts_ns
                    or frame.video_pts_ns - self._last_enqueued_pts_ns > 1_000_000_000
                )
            ):
                self._source_pts_ns.clear()
            self._source_pts_ns.append(frame.video_pts_ns)
        self._last_enqueued_pts_ns = frame.video_pts_ns
        if len(self._queue) >= self._maximum_queue_frames:
            self._queue.popleft()
            self._frames_dropped += 1
        self._queue.append(frame)

    def next_frame(self, now_ns: int) -> LiveFrame | None:
        if self._presented is None:
            if not self._ready_to_start(now_ns):
                return None
            self._drop_excess_before_presentation()
            return self._present_next(now_ns, reset_deadline=True)
        if self._starved:
            if not self._ready_to_start(now_ns):
                return self._presented
            self._drop_excess_before_presentation()
            return self._present_next(now_ns, reset_deadline=True)
        if self._next_due_ns is not None and now_ns < self._next_due_ns:
            return self._presented
        if not self._queue:
            self._starvations += 1
            self._starved = True
            self._next_due_ns = None
            return self._presented
        self._drop_excess_before_presentation()
        return self._present_next(now_ns, reset_deadline=self._starved)

    def status(self, now_ns: int | None = None) -> LiveFramePacerStatus:
        measured_at_ns = time.perf_counter_ns() if now_ns is None else now_ns
        source_gap_p95_ms, source_gap_max_ms = _gap_stats_ms(tuple(self._source_pts_ns))
        return LiveFramePacerStatus(
            queue_depth=len(self._queue),
            frames_dropped=self._frames_dropped,
            starvations=self._starvations,
            interval_ms=round(self._interval_ns() / 1_000_000, 3),
            frames_presented=self._frames_presented,
            recent_fps=round(_recent_rate(self._presented_at_ns, measured_at_ns), 3),
            source_gap_p95_ms=source_gap_p95_ms,
            source_gap_max_ms=source_gap_max_ms,
        )

    def _ready_to_start(self, now_ns: int) -> bool:
        if len(self._queue) >= self._target_queue_frames:
            return True
        return bool(
            self._queue
            and now_ns - self._queue[0].converted_at_client_monotonic_ns
            >= self._startup_wait_ns
        )

    def _drop_excess_before_presentation(self) -> None:
        # Presenting removes one frame, so allow target + 1 before the pop.
        while len(self._queue) > self._target_queue_frames + 1:
            self._queue.popleft()
            self._frames_dropped += 1

    def _present_next(self, now_ns: int, *, reset_deadline: bool) -> LiveFrame:
        self._presented = self._queue.popleft()
        interval_ns = self._interval_ns()
        if reset_deadline or self._next_due_ns is None:
            self._next_due_ns = now_ns + interval_ns
        else:
            self._next_due_ns = max(self._next_due_ns + interval_ns, now_ns + 1)
        self._starved = False
        self._frames_presented += 1
        self._presented_at_ns.append(now_ns)
        return self._presented

    def _interval_ns(self) -> int:
        if len(self._source_pts_ns) < 2:
            return self._default_interval_ns
        span_ns = self._source_pts_ns[-1] - self._source_pts_ns[0]
        if span_ns <= 0:
            return self._default_interval_ns
        estimated_ns = round(span_ns / (len(self._source_pts_ns) - 1))
        return min(100_000_000, max(16_666_667, estimated_ns))

    def _reset_stream(self, stream_key: tuple[str, str]) -> None:
        self._queue.clear()
        self._source_pts_ns.clear()
        self._stream_key = stream_key
        self._last_enqueued_pts_ns = None
        self._presented = None
        self._next_due_ns = None
        self._starved = False


class LiveRgbFrameSink(Protocol):
    """A consumer of the one canonical RGB conversion produced for each live frame."""

    async def submit_rgb_frame(
        self,
        *,
        session_id: str,
        connection_session_id: str,
        frame_index: int,
        received_at_client_monotonic_ns: int,
        image_rgb: np.ndarray,
    ) -> None: ...


class LiveFrameBuffer:
    """Convert WebRTC frames off the media callback and retain only the newest RGB frame."""

    def __init__(
        self,
        *,
        perf_clock: Callable[[], int] = time.perf_counter_ns,
        converter: Callable[[VideoFrame], np.ndarray] | None = None,
        rgb_frame_sink: LiveRgbFrameSink | None = None,
    ) -> None:
        self._perf_clock = perf_clock
        self._converter = converter or _to_rgb24
        self._rgb_frame_sink = rgb_frame_sink
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ui-frame")
        self._pending_lock = asyncio.Lock()
        self._state_lock = threading.Lock()
        self._pending: _PendingFrame | None = None
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self._latest: LiveFrame | None = None
        self._pacer = LiveFramePacer()
        self._frames_received = 0
        self._frames_converted = 0
        self._pending_frames_overwritten = 0
        self._conversion_failures = 0
        self._rgb_frames_forwarded = 0
        self._rgb_sink_failures = 0
        self._latest_conversion_ms: float | None = None
        self._last_error: str | None = None
        self._published_at_ns: deque[int] = deque(maxlen=240)
        self._display_polled_at_ns: deque[int] = deque(maxlen=240)

    def set_rgb_frame_sink(self, sink: LiveRgbFrameSink | None) -> None:
        """Route immutable RGB frames downstream after display conversion completes."""

        self._rgb_frame_sink = sink

    async def submit_gateway_frame(
        self,
        *,
        session_id: str,
        connection_session_id: str,
        frame_index: int,
        received_at_client_monotonic_ns: int,
        decoded_frame: VideoFrame,
    ) -> None:
        pending = _PendingFrame(
            session_id=session_id,
            connection_session_id=connection_session_id,
            frame_index=frame_index,
            received_at_client_monotonic_ns=received_at_client_monotonic_ns,
            decoded_frame=decoded_frame,
            video_pts_ns=_video_pts_ns(decoded_frame),
        )
        async with self._pending_lock:
            if self._closed:
                return
            with self._state_lock:
                self._frames_received += 1
                if self._pending is not None:
                    self._pending_frames_overwritten += 1
            self._pending = pending
            if self._worker is None or self._worker.done():
                self._worker = asyncio.create_task(self._run_converter())

    def latest(self) -> LiveFrame | None:
        with self._state_lock:
            return self._latest

    def next_for_display(self) -> LiveFrame | None:
        now_ns = self._perf_clock()
        with self._state_lock:
            self._display_polled_at_ns.append(now_ns)
            return self._pacer.next_frame(now_ns)

    def status(self) -> LiveFrameStatus:
        now_ns = self._perf_clock()
        with self._state_lock:
            published = tuple(self._published_at_ns)
            recent_cutoff_ns = now_ns - 2_000_000_000
            recent = tuple(value for value in published if value >= recent_cutoff_ns)
            recent_fps = 0.0
            if len(recent) > 1 and recent[-1] > recent[0]:
                recent_fps = (len(recent) - 1) * 1_000_000_000 / (recent[-1] - recent[0])
            latest = self._latest
            presentation = self._pacer.status(now_ns)
            conversion_gap_p95_ms, conversion_gap_max_ms = _gap_stats_ms(recent)
            latest_age_ms = None
            if latest is not None:
                latest_age_ms = max(
                    0.0,
                    (now_ns - latest.converted_at_client_monotonic_ns) / 1_000_000,
                )
            return LiveFrameStatus(
                frames_received=self._frames_received,
                frames_converted=self._frames_converted,
                pending_frames_overwritten=self._pending_frames_overwritten,
                conversion_failures=self._conversion_failures,
                rgb_frames_forwarded=self._rgb_frames_forwarded,
                rgb_sink_failures=self._rgb_sink_failures,
                presentation_queue_depth=presentation.queue_depth,
                presentation_frames_dropped=presentation.frames_dropped,
                presentation_starvations=presentation.starvations,
                presentation_interval_ms=presentation.interval_ms,
                display_poll_fps=round(
                    _recent_rate(self._display_polled_at_ns, now_ns),
                    3,
                ),
                presentation_fps=presentation.recent_fps,
                presentation_frames_presented=presentation.frames_presented,
                conversion_gap_p95_ms=conversion_gap_p95_ms,
                conversion_gap_max_ms=conversion_gap_max_ms,
                source_pts_gap_p95_ms=presentation.source_gap_p95_ms,
                source_pts_gap_max_ms=presentation.source_gap_max_ms,
                latest_frame_index=latest.frame_index if latest is not None else None,
                latest_width=latest.width if latest is not None else None,
                latest_height=latest.height if latest is not None else None,
                recent_fps=round(recent_fps, 3),
                latest_conversion_ms=self._latest_conversion_ms,
                latest_frame_age_ms=(
                    round(latest_age_ms, 3) if latest_age_ms is not None else None
                ),
                last_error=self._last_error,
            )

    async def close(self) -> None:
        async with self._pending_lock:
            self._closed = True
            self._pending = None
            worker = self._worker
        if worker is not None:
            await worker
        self._executor.shutdown(wait=True, cancel_futures=True)

    async def _run_converter(self) -> None:
        while True:
            async with self._pending_lock:
                pending = self._pending
                self._pending = None
            if pending is None:
                return
            started_at_ns = self._perf_clock()
            try:
                image_rgb = await asyncio.get_running_loop().run_in_executor(
                    self._executor,
                    self._converter,
                    pending.decoded_frame,
                )
                if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
                    raise ValueError("live frame converter must return HxWx3 uint8 RGB")
                if not image_rgb.flags.c_contiguous:
                    image_rgb = np.ascontiguousarray(image_rgb)
                image_rgb.setflags(write=False)
            except Exception as error:
                LOGGER.exception("failed to convert decoded frame for native UI")
                with self._state_lock:
                    self._conversion_failures += 1
                    self._last_error = str(error)
                continue
            converted_at_ns = self._perf_clock()
            live_frame = LiveFrame(
                session_id=pending.session_id,
                connection_session_id=pending.connection_session_id,
                frame_index=pending.frame_index,
                received_at_client_monotonic_ns=pending.received_at_client_monotonic_ns,
                converted_at_client_monotonic_ns=converted_at_ns,
                image_rgb=image_rgb,
                video_pts_ns=pending.video_pts_ns,
            )
            with self._state_lock:
                self._latest = live_frame
                self._pacer.enqueue(live_frame)
                self._frames_converted += 1
                self._latest_conversion_ms = round(
                    (converted_at_ns - started_at_ns) / 1_000_000,
                    3,
                )
                self._published_at_ns.append(converted_at_ns)
                self._last_error = None
            sink = self._rgb_frame_sink
            if sink is not None:
                try:
                    await sink.submit_rgb_frame(
                        session_id=live_frame.session_id,
                        connection_session_id=live_frame.connection_session_id,
                        frame_index=live_frame.frame_index,
                        received_at_client_monotonic_ns=(
                            live_frame.received_at_client_monotonic_ns
                        ),
                        image_rgb=live_frame.image_rgb,
                    )
                except Exception as error:
                    LOGGER.exception("RGB frame sink rejected canonical live frame")
                    with self._state_lock:
                        self._rgb_sink_failures += 1
                        self._last_error = str(error)
                else:
                    with self._state_lock:
                        self._rgb_frames_forwarded += 1


def _to_rgb24(frame: VideoFrame) -> np.ndarray:
    return frame.to_ndarray(format="rgb24")


def _video_pts_ns(frame: VideoFrame) -> int | None:
    if frame.pts is None or frame.time_base is None:
        return None
    return round(Fraction(frame.pts) * Fraction(frame.time_base) * 1_000_000_000)


def _recent_rate(timestamps_ns: deque[int], now_ns: int) -> float:
    recent_cutoff_ns = now_ns - 2_000_000_000
    recent = tuple(value for value in timestamps_ns if value >= recent_cutoff_ns)
    if len(recent) < 2 or recent[-1] <= recent[0]:
        return 0.0
    return (len(recent) - 1) * 1_000_000_000 / (recent[-1] - recent[0])


def _gap_stats_ms(timestamps_ns: tuple[int, ...]) -> tuple[float, float]:
    gaps_ns = sorted(
        current - previous
        for previous, current in zip(timestamps_ns, timestamps_ns[1:], strict=False)
        if current > previous
    )
    if not gaps_ns:
        return 0.0, 0.0
    percentile_index = round((len(gaps_ns) - 1) * 0.95)
    return (
        round(gaps_ns[percentile_index] / 1_000_000, 3),
        round(gaps_ns[-1] / 1_000_000, 3),
    )
