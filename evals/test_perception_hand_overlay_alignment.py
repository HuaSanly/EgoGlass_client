from pathlib import Path

import numpy as np

from perception.spatial_perception.hand_tracking import (
    rotated_image_bbox_to_source,
    rotated_image_points_to_source,
    source_image_dimensions,
)


def test_portrait_hamer_coordinates_align_with_landscape_decoded_preview() -> None:
    rotated_points = np.asarray(((719.0, 0.0), (0.0, 1279.0)), dtype=np.float32)

    source_points = rotated_image_points_to_source(
        rotated_points,
        image_width_px=720,
        image_height_px=1280,
        rotation_degrees=90,
    )
    source_bbox = rotated_image_bbox_to_source(
        (100.0, 200.0, 300.0, 500.0),
        image_width_px=720,
        image_height_px=1280,
        rotation_degrees=90,
    )

    assert source_image_dimensions(720, 1280, 90) == (1280, 720)
    np.testing.assert_allclose(source_points, ((0.0, 0.0), (1279.0, 719.0)))
    assert source_bbox == (200.0, 420.0, 500.0, 620.0)


def test_operator_overlay_consumes_source_coordinate_contract() -> None:
    repository = Path(__file__).parents[1]
    script = (repository / "src/operator_console/static/app.js").read_text(encoding="utf-8")

    assert "hand.source_keypoints_2d_px.map(mapPoint)" in script
    assert "hand.source_bbox_xyxy_px.slice(0, 2)" in script
    assert "result?.source_image_width_px" in script
    assert "result?.source_image_height_px" in script
