from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from schemas import PlaybackFrame
from ui.widgets.video_canvas import VideoCanvas


def test_video_canvas_renders_object_mask_and_visible_track_points(
    qt_application: object,
    tmp_path: Path,
) -> None:
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:200, 100:200] = 255
    mask_path = tmp_path / "mask.png"
    encoded, payload = cv2.imencode(".png", mask)
    assert encoded
    mask_path.write_bytes(payload.tobytes())
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image.setflags(write=False)
    canvas = VideoCanvas()
    canvas.resize(640, 480)
    canvas.set_frame(PlaybackFrame("session", "clip", 3, 100, 100, image))
    canvas.set_overlay(
        {
            "session_id": "session",
            "sequence_id": "clip",
            "frame_index": 3,
            "session_time_ns": 100,
            "source_image_width_px": 640,
            "source_image_height_px": 480,
            "hands": [],
            "object_overlays": [
                {
                    "object_id": "obj1",
                    "mask_path": str(mask_path),
                    "track": {
                        "points_xy_px": [[250.0, 250.0], [300.0, 300.0]],
                        "visibility": [1.0, 0.0],
                    },
                }
            ],
        }
    )

    rendered = canvas.grab().toImage()
    qt_application.processEvents()  # type: ignore[attr-defined]

    assert rendered.pixelColor(150, 150) != rendered.pixelColor(50, 50)
    assert rendered.pixelColor(250, 250) != rendered.pixelColor(300, 300)
    assert canvas.status().overlay_visible
    canvas.close()
