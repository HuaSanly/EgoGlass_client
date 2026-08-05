from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
from aiortc import MediaStreamError
from av import VideoFrame

_ENCODER_TIME_BASE = Fraction(1, 90_000)
_NS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class RecordedVideoFrame:
    frame_index: int
    source_frame_pts: int | None
    source_frame_time_base_num: int | None
    source_frame_time_base_den: int | None
    mp4_pts: int
    mp4_time_base_num: int
    mp4_time_base_den: int
    received_at_client_perf_counter_ns: int


class PyAvH264Mp4Recorder:
    """Encode one decoded WebRTC video track into a finalized H.264 MP4."""

    def __init__(
        self,
        path: Path,
        track: object,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        perf_clock: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self._path = path
        self._track = track
        self._width = width
        self._height = height
        self._fps = fps
        self._perf_clock = perf_clock
        self._container: av.container.OutputContainer | None = None
        self._stream: av.video.stream.VideoStream | None = None
        self._task: asyncio.Task[None] | None = None
        self._frames_received = 0
        self._source_frame_records: list[tuple[int | None, int | None, int | None, int]] = []
        self._frame_records: list[RecordedVideoFrame] = []
        self._source_segment_anchor: Fraction | None = None
        self._output_segment_anchor_pts = 0
        self._last_source_time: Fraction | None = None
        self._last_output_pts: int | None = None
        self._last_received_at_ns: int | None = None

    @property
    def frames_received(self) -> int:
        return self._frames_received

    @property
    def frame_records(self) -> tuple[RecordedVideoFrame, ...]:
        return tuple(self._frame_records)

    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("recorder is already started")
        self._container = av.open(
            str(self._path),
            mode="w",
            format="mp4",
            options={"movflags": "+faststart"},
        )
        stream = self._container.add_stream(
            "libx264",
            rate=self._fps,
            options={"crf": "18", "preset": "veryfast"},
        )
        stream.width = self._width
        stream.height = self._height
        stream.pix_fmt = "yuv420p"
        stream.time_base = _ENCODER_TIME_BASE
        stream.codec_context.time_base = _ENCODER_TIME_BASE
        self._stream = stream
        self._task = asyncio.create_task(self._consume())

    async def wait(self) -> None:
        if self._task is None:
            raise RuntimeError("recorder is not started")
        await asyncio.shield(self._task)

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass
            finally:
                self._close_container()
        else:
            self._close_container()

    async def _consume(self) -> None:
        try:
            while True:
                frame = await self._track.recv()  # type: ignore[attr-defined]
                received_at_client_perf_counter_ns = self._perf_clock()
                if not isinstance(frame, VideoFrame):
                    raise TypeError("recording track returned a non-video frame")
                if (frame.width, frame.height) != (self._width, self._height):
                    raise ValueError(
                        "source dimensions changed during recording: "
                        f"{frame.width}x{frame.height}"
                    )
                converted_frame = frame.reformat(
                    width=self._width,
                    height=self._height,
                    format="yuv420p",
                )
                output_frame = VideoFrame(self._width, self._height, "yuv420p")
                for source_plane, output_plane in zip(
                    converted_frame.planes,
                    output_frame.planes,
                    strict=True,
                ):
                    output_plane.update(bytes(source_plane))
                output_frame.pts = self._next_output_pts(
                    frame,
                    received_at_client_perf_counter_ns,
                )
                output_frame.time_base = _ENCODER_TIME_BASE
                container = self._container
                stream = self._stream
                if container is None or stream is None:
                    raise RuntimeError("recorder container is closed")
                for packet in stream.encode(output_frame):
                    container.mux(packet)
                source_time_base = frame.time_base
                self._source_frame_records.append(
                    (
                        frame.pts,
                        source_time_base.numerator if source_time_base is not None else None,
                        source_time_base.denominator if source_time_base is not None else None,
                        received_at_client_perf_counter_ns,
                    )
                )
                self._frames_received += 1
        except MediaStreamError:
            return

    def _next_output_pts(
        self,
        frame: VideoFrame,
        received_at_client_perf_counter_ns: int,
    ) -> int:
        """Preserve source presentation time and bridge only real discontinuities."""

        source_time = (
            Fraction(frame.pts) * Fraction(frame.time_base)
            if frame.pts is not None and frame.time_base is not None
            else None
        )
        if self._last_output_pts is None:
            output_pts = 0
            if source_time is not None:
                self._source_segment_anchor = source_time
                self._output_segment_anchor_pts = output_pts
        else:
            output_pts = self._source_output_candidate(source_time)
            if output_pts is None or output_pts <= self._last_output_pts:
                assert self._last_received_at_ns is not None
                receipt_delta_ns = max(
                    0,
                    received_at_client_perf_counter_ns - self._last_received_at_ns,
                )
                output_pts = self._last_output_pts + max(
                    1,
                    _round_fraction(
                        Fraction(receipt_delta_ns * _ENCODER_TIME_BASE.denominator, _NS_PER_SECOND)
                    ),
                )
                if source_time is not None:
                    self._source_segment_anchor = source_time
                    self._output_segment_anchor_pts = output_pts

        self._last_source_time = source_time
        self._last_output_pts = output_pts
        self._last_received_at_ns = received_at_client_perf_counter_ns
        return output_pts

    def _source_output_candidate(self, source_time: Fraction | None) -> int | None:
        if (
            source_time is None
            or self._last_source_time is None
            or source_time <= self._last_source_time
            or self._source_segment_anchor is None
        ):
            return None
        source_delta = source_time - self._source_segment_anchor
        return self._output_segment_anchor_pts + _round_fraction(
            source_delta * _ENCODER_TIME_BASE.denominator
        )

    def _close_container(self) -> None:
        container, self._container = self._container, None
        stream, self._stream = self._stream, None
        if container is None:
            return
        if stream is not None:
            for packet in stream.encode(None):
                container.mux(packet)
        container.close()
        self._load_muxed_frame_timing()

    def _load_muxed_frame_timing(self) -> None:
        muxed_frames: list[tuple[int, int, int]] = []
        with av.open(str(self._path), mode="r") as container:
            for frame in container.decode(video=0):
                if frame.pts is None or frame.time_base is None:
                    raise RuntimeError("finalized MP4 frame is missing exact PTS")
                muxed_frames.append(
                    (frame.pts, frame.time_base.numerator, frame.time_base.denominator)
                )
        if len(muxed_frames) != len(self._source_frame_records):
            raise RuntimeError("finalized MP4 frame index does not match encoded frames")
        self._frame_records = [
            RecordedVideoFrame(
                frame_index=frame_index,
                source_frame_pts=source[0],
                source_frame_time_base_num=source[1],
                source_frame_time_base_den=source[2],
                mp4_pts=muxed[0],
                mp4_time_base_num=muxed[1],
                mp4_time_base_den=muxed[2],
                received_at_client_perf_counter_ns=source[3],
            )
            for frame_index, (source, muxed) in enumerate(
                zip(self._source_frame_records, muxed_frames, strict=True)
            )
        ]


def _round_fraction(value: Fraction) -> int:
    quotient, remainder = divmod(value.numerator, value.denominator)
    return quotient + int(remainder * 2 >= value.denominator)
