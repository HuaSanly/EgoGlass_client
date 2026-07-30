from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

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


class LiveFrameBuffer:
    """Convert WebRTC frames off the media callback and retain only the newest RGB frame."""

    def __init__(
        self,
        *,
        perf_clock: Callable[[], int] = time.perf_counter_ns,
        converter: Callable[[VideoFrame], np.ndarray] | None = None,
    ) -> None:
        self._perf_clock = perf_clock
        self._converter = converter or _to_rgb24
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ui-frame")
        self._pending_lock = asyncio.Lock()
        self._state_lock = threading.Lock()
        self._pending: _PendingFrame | None = None
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self._latest: LiveFrame | None = None
        self._frames_received = 0
        self._frames_converted = 0
        self._pending_frames_overwritten = 0
        self._conversion_failures = 0
        self._latest_conversion_ms: float | None = None
        self._last_error: str | None = None
        self._published_at_ns: deque[int] = deque(maxlen=240)

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
            )
            with self._state_lock:
                self._latest = live_frame
                self._frames_converted += 1
                self._latest_conversion_ms = round(
                    (converted_at_ns - started_at_ns) / 1_000_000,
                    3,
                )
                self._published_at_ns.append(converted_at_ns)
                self._last_error = None


def _to_rgb24(frame: VideoFrame) -> np.ndarray:
    return frame.to_ndarray(format="rgb24")
