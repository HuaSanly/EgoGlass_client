"""Immutable RGB frame boundary shared by gateway, replay, and algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

RgbFrame = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class FramePacket:
    """One decoded RGB frame and its source timing identity."""

    session_id: str
    stream_id: str
    frame_index: int
    captured_at_ns: int | None
    received_at_ns: int
    pts_ns: int | None
    image_rgb: RgbFrame

    def __post_init__(self) -> None:
        image = self.image_rgb
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise TypeError("image_rgb must be an uint8 HxWx3 array")
        if not image.flags.c_contiguous:
            raise ValueError("image_rgb must be C-contiguous")
        if image.flags.writeable:
            raise ValueError("image_rgb must be read-only")
        if self.frame_index < 0 or self.received_at_ns < 0:
            raise ValueError("frame index and receive time must be non-negative")

    @property
    def width(self) -> int:
        return int(self.image_rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.image_rgb.shape[0])
