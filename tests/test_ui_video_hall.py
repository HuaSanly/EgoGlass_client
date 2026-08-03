from __future__ import annotations

import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from ui.video_processing.thumbnails import VideoThumbnailService


def test_thumbnail_service_decodes_exact_media_first_frame(tmp_path: Path) -> None:
    session = tmp_path / "session"
    media = session / "media"
    media.mkdir(parents=True)
    clip_id = "clip-a"
    _write_clip(media / f"{clip_id}.mp4")
    service = VideoThumbnailService(workers=1)
    try:
        service.request("session", clip_id, session)
        result = _wait_for_thumbnail(service)

        assert result.error is None
        assert result.image is not None
        assert result.image.width() == 320
        assert result.image.height() == 240
        assert result.image.pixelColor(160, 120).red() > 100
    finally:
        service.close()


def test_thumbnail_service_does_not_recursively_scan_for_media(tmp_path: Path) -> None:
    session = tmp_path / "session"
    nested = session / "unrelated" / "nested"
    nested.mkdir(parents=True)
    _write_clip(nested / "clip-a.mp4")
    service = VideoThumbnailService(workers=1)
    try:
        service.request("session", "clip-a", session)
        result = _wait_for_thumbnail(service)

        assert result.image is None
        assert result.error == "找不到视频文件"
    finally:
        service.close()


def _wait_for_thumbnail(service: VideoThumbnailService):
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        completed = service.take_completed()
        if completed:
            return completed[0]
        time.sleep(0.005)
    raise AssertionError("thumbnail decoder did not complete")


def _write_clip(path: Path) -> None:
    with av.open(str(path), mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 30)
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        image[:, :, 0] = 200
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = 0
        frame.time_base = Fraction(1, 30)
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
