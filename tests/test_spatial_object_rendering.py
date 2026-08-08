from __future__ import annotations

import numpy as np

from schemas import VioPose
from ui.presentation.spatial_scene import (
    SpatialReferenceFrame,
    build_spatial_scene_state,
)
from ui.widgets.spatial_sync_canvas import SpatialSyncCanvas


def _transform(tx: float) -> list[float]:
    return [
        1.0,
        0.0,
        0.0,
        tx,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def _overlay(current_tx: float = 4.0) -> list[dict[str, object]]:
    return [
        {
            "object_id": "cup",
            "triangulation": {
                "points_world_m": [[3.5, 0.0, 0.0], [3.0, 0.1, 0.0], [3.0, 0.0, 0.1]],
                "transform_object_to_world": _transform(3.0),
            },
            "pose": {"transform_object_to_world": _transform(current_tx)},
        }
    ]


def test_world_object_cloud_uses_first_vio_origin_and_current_pose() -> None:
    state = build_spatial_scene_state(
        SpatialReferenceFrame.WORLD,
        hand_result={"frame_index": 12, "object_overlays": _overlay()},
        vio_first_pose=VioPose(1_000, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        vio_pose=VioPose(2_000, (2.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        allow_imu_world_fallback=False,
    )

    np.testing.assert_allclose(state.object_points_m[0], (3.5, 0.0, 0.0))
    np.testing.assert_allclose(state.object_axes_m[0][0], (3.0, 0.0, 0.0))
    assert state.latest_frame_index == 12
    assert state.show_ground


def test_camera_reference_hides_object_world_geometry() -> None:
    state = build_spatial_scene_state(
        SpatialReferenceFrame.CAMERA,
        vio_first_pose=VioPose(1_000, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        vio_pose=VioPose(2_000, (2.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        object_overlays=_overlay(),
    )

    assert state.object_points_m == ()
    assert state.object_axes_m == ()
    assert not state.show_ground


def test_malformed_object_overlay_is_ignored() -> None:
    state = build_spatial_scene_state(
        SpatialReferenceFrame.WORLD,
        vio_first_pose=VioPose(1_000, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        vio_pose=VioPose(2_000, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        object_overlays=[{"triangulation": {"points_world_m": [[float("nan"), 0, 0]]}}],
        allow_imu_world_fallback=False,
    )

    assert state.object_points_m == ()
    assert state.object_axes_m == ()


def test_canvas_renders_object_geometry_without_moving_camera(qt_application: object) -> None:
    canvas = SpatialSyncCanvas()
    canvas.set_reference_frame(SpatialReferenceFrame.WORLD)
    first_pose = VioPose(1_000, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    current_pose = VioPose(2_000, (2.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    first_state = build_spatial_scene_state(
        SpatialReferenceFrame.WORLD,
        vio_first_pose=first_pose,
        vio_pose=current_pose,
        object_overlays=_overlay(),
        allow_imu_world_fallback=False,
    )
    canvas.set_scene_state(first_state)
    center = canvas.view.opts["center"]
    camera_before = (
        center.x(),
        center.y(),
        center.z(),
        canvas.view.opts["distance"],
        canvas.view.opts["elevation"],
        canvas.view.opts["azimuth"],
        canvas.view.opts["fov"],
    )

    second_state = build_spatial_scene_state(
        SpatialReferenceFrame.WORLD,
        vio_first_pose=first_pose,
        vio_pose=current_pose,
        object_overlays=_overlay(current_tx=7.0),
        allow_imu_world_fallback=False,
    )
    canvas.set_scene_state(second_state)
    center = canvas.view.opts["center"]
    camera_after = (
        center.x(),
        center.y(),
        center.z(),
        canvas.view.opts["distance"],
        canvas.view.opts["elevation"],
        canvas.view.opts["azimuth"],
        canvas.view.opts["fov"],
    )

    assert canvas._object_points.pos.shape == (3, 3)
    assert canvas._object_axes.pos.shape == (6, 3)
    assert camera_after == camera_before
    canvas.close()
