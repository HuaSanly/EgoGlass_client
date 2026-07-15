from __future__ import annotations

from typing import Any

import av


class MjpegPreviewEncoder:
    """Encode decoded video frames as browser-compatible JPEG snapshots."""

    def __init__(self) -> None:
        self._codec: Any | None = None
        self._size: tuple[int, int] | None = None

    def encode(self, frame: Any) -> bytes:
        size = (int(frame.width), int(frame.height))
        if self._codec is None or self._size != size:
            codec = av.CodecContext.create("mjpeg", "w")
            codec.width, codec.height = size
            codec.pix_fmt = "yuvj420p"
            self._codec = codec
            self._size = size

        preview_frame = frame.reformat(width=size[0], height=size[1], format="yuvj420p")
        payload = b"".join(bytes(packet) for packet in self._codec.encode(preview_frame))
        if not payload:
            raise RuntimeError("MJPEG encoder produced no preview frame")
        return payload
