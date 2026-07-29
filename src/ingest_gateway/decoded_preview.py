from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from enum import StrEnum

import cv2
from av import VideoFrame
from pydantic import BaseModel, ConfigDict, Field

MJPEG_BOUNDARY = "frame"


class DecodedPreviewState(StrEnum):
    WAITING = "waiting"
    STREAMING = "streaming"
    ERROR = "error"
    CLOSED = "closed"


class DecodedPreviewStatus(BaseModel):
    """Observable state for the gateway-owned decoded preview stream."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    state: DecodedPreviewState
    frames_received: int = Field(ge=0)
    frames_encoded: int = Field(ge=0)
    frames_dropped: int = Field(ge=0)
    clients_connected: int = Field(ge=0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    output_fps: float | None = Field(default=None, ge=0)
    latest_frame_index: int | None = Field(default=None, ge=0)
    latest_received_at_client_monotonic_ns: int | None = Field(default=None, ge=0)
    latest_encoded_at_perf_counter_ns: int | None = Field(default=None, ge=0)
    last_frame_age_ms: float | None = Field(default=None, ge=0)
    last_error: str | None = None


@dataclass(frozen=True, slots=True)
class _PendingFrame:
    frame_index: int
    received_at_client_monotonic_ns: int
    decoded_frame: VideoFrame


JpegEncoder = Callable[[VideoFrame], bytes]


class DecodedPreviewRuntime:
    """Encode the newest gateway-decoded frame once and broadcast it as MJPEG."""

    def __init__(
        self,
        *,
        jpeg_quality: int = 82,
        encoder: JpegEncoder | None = None,
        perf_clock: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        self._jpeg_quality = jpeg_quality
        self._encoder = encoder or self._encode_jpeg
        self._perf_clock = perf_clock
        self._condition = asyncio.Condition()
        self._pending_frame: _PendingFrame | None = None
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self._state = DecodedPreviewState.WAITING
        self._frames_received = 0
        self._frames_encoded = 0
        self._frames_dropped = 0
        self._clients_connected = 0
        self._width: int | None = None
        self._height: int | None = None
        self._latest_jpeg: bytes | None = None
        self._latest_frame_index: int | None = None
        self._latest_received_at_ns: int | None = None
        self._latest_encoded_at_ns: int | None = None
        self._first_encoded_at_ns: int | None = None
        self._last_error: str | None = None

    async def submit_gateway_frame(
        self,
        *,
        session_id: str,
        connection_session_id: str,
        frame_index: int,
        received_at_client_monotonic_ns: int,
        decoded_frame: VideoFrame,
    ) -> None:
        """Keep only the newest frame while the sole JPEG encoder is busy."""

        del session_id, connection_session_id
        async with self._condition:
            if self._closed:
                return
            self._frames_received += 1
            if self._pending_frame is not None:
                self._frames_dropped += 1
            self._pending_frame = _PendingFrame(
                frame_index=frame_index,
                received_at_client_monotonic_ns=received_at_client_monotonic_ns,
                decoded_frame=decoded_frame,
            )
            if self._worker is None or self._worker.done():
                self._worker = asyncio.create_task(self._run_encoder())

    async def status(self) -> DecodedPreviewStatus:
        async with self._condition:
            output_fps = None
            if (
                self._frames_encoded > 1
                and self._first_encoded_at_ns is not None
                and self._latest_encoded_at_ns is not None
            ):
                duration_ns = self._latest_encoded_at_ns - self._first_encoded_at_ns
                if duration_ns > 0:
                    output_fps = round(
                        (self._frames_encoded - 1) * 1_000_000_000 / duration_ns,
                        3,
                    )
            last_frame_age_ms = None
            if self._latest_encoded_at_ns is not None:
                last_frame_age_ms = max(
                    0.0,
                    round((self._perf_clock() - self._latest_encoded_at_ns) / 1_000_000, 3),
                )
            return DecodedPreviewStatus(
                state=self._state,
                frames_received=self._frames_received,
                frames_encoded=self._frames_encoded,
                frames_dropped=self._frames_dropped,
                clients_connected=self._clients_connected,
                width=self._width,
                height=self._height,
                output_fps=output_fps,
                latest_frame_index=self._latest_frame_index,
                latest_received_at_client_monotonic_ns=self._latest_received_at_ns,
                latest_encoded_at_perf_counter_ns=self._latest_encoded_at_ns,
                last_frame_age_ms=last_frame_age_ms,
                last_error=self._last_error,
            )

    async def stream(self) -> AsyncIterator[bytes]:
        """Yield the current frame immediately, then each newly encoded frame once."""

        last_frame_index: int | None = None
        async with self._condition:
            self._clients_connected += 1
        try:
            while True:
                async with self._condition:
                    while (
                        not self._closed
                        and (
                            self._latest_jpeg is None
                            or self._latest_frame_index == last_frame_index
                        )
                    ):
                        await self._condition.wait()
                    if self._closed:
                        return
                    jpeg = self._latest_jpeg
                    frame_index = self._latest_frame_index
                if jpeg is None or frame_index is None:
                    continue
                last_frame_index = frame_index
                yield self._mjpeg_part(jpeg, frame_index)
        finally:
            async with self._condition:
                self._clients_connected -= 1

    async def close(self) -> None:
        async with self._condition:
            if self._closed:
                return
            self._closed = True
            self._state = DecodedPreviewState.CLOSED
            self._pending_frame = None
            worker = self._worker
            self._condition.notify_all()
        if worker is not None:
            await worker

    async def _run_encoder(self) -> None:
        while True:
            async with self._condition:
                if self._closed:
                    return
                pending = self._pending_frame
                self._pending_frame = None
                if pending is None:
                    return
            try:
                jpeg = await asyncio.to_thread(self._encoder, pending.decoded_frame)
                encoded_at_ns = self._perf_clock()
            except Exception as error:
                async with self._condition:
                    self._state = DecodedPreviewState.ERROR
                    self._last_error = f"decoded preview encode failed: {type(error).__name__}"
                continue
            async with self._condition:
                if self._closed:
                    return
                self._frames_encoded += 1
                self._state = DecodedPreviewState.STREAMING
                self._width = pending.decoded_frame.width
                self._height = pending.decoded_frame.height
                self._latest_jpeg = jpeg
                self._latest_frame_index = pending.frame_index
                self._latest_received_at_ns = pending.received_at_client_monotonic_ns
                self._latest_encoded_at_ns = encoded_at_ns
                if self._first_encoded_at_ns is None:
                    self._first_encoded_at_ns = encoded_at_ns
                self._last_error = None
                self._condition.notify_all()

    def _encode_jpeg(self, frame: VideoFrame) -> bytes:
        image_bgr = frame.to_ndarray(format="bgr24")
        success, encoded = cv2.imencode(
            ".jpg",
            image_bgr,
            (cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality),
        )
        if not success:
            raise RuntimeError("OpenCV could not encode decoded preview frame")
        return encoded.tobytes()

    @staticmethod
    def _mjpeg_part(jpeg: bytes, frame_index: int) -> bytes:
        headers = (
            f"--{MJPEG_BOUNDARY}\r\n"
            "Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(jpeg)}\r\n"
            f"X-Frame-Index: {frame_index}\r\n\r\n"
        ).encode("ascii")
        return headers + jpeg + b"\r\n"
