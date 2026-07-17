from __future__ import annotations

import asyncio
from fractions import Fraction
from pathlib import Path

import av
from aiortc import MediaStreamError
from av import VideoFrame


class PyAvH264Mp4Recorder:
    """Encode one decoded WebRTC video track into a finalized H.264 MP4."""

    def __init__(
        self,
        path: Path,
        track: object,
        *,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
    ) -> None:
        self._path = path
        self._track = track
        self._width = width
        self._height = height
        self._fps = fps
        self._container: av.container.OutputContainer | None = None
        self._stream: av.video.stream.VideoStream | None = None
        self._task: asyncio.Task[None] | None = None
        self._frames_received = 0

    @property
    def frames_received(self) -> int:
        return self._frames_received

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
                output_frame.pts = self._frames_received
                output_frame.time_base = Fraction(1, self._fps)
                container = self._container
                stream = self._stream
                if container is None or stream is None:
                    raise RuntimeError("recorder container is closed")
                for packet in stream.encode(output_frame):
                    container.mux(packet)
                self._frames_received += 1
        except MediaStreamError:
            return

    def _close_container(self) -> None:
        container, self._container = self._container, None
        stream, self._stream = self._stream, None
        if container is None:
            return
        if stream is not None:
            for packet in stream.encode(None):
                container.mux(packet)
        container.close()
