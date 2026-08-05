"""Playback frame shared by the video canvas and spatial presentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PlaybackFrame:
    session_id: str
    clip_id: str
    frame_index: int
    pts_ns: int
    session_time_ns: int
    image_rgb: NDArray[np.uint8]

    def __post_init__(self) -> None:
        if self.image_rgb.dtype != np.uint8 or self.image_rgb.ndim != 3:
            raise TypeError("image_rgb must be an uint8 image")
        if self.image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must have three channels")

    @property
    def width(self) -> int:
        return int(self.image_rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.image_rgb.shape[0])
