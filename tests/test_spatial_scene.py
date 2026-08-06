from __future__ import annotations

import numpy as np

from schemas import VioPose
from ui.presentation.spatial_scene import (
    SpatialReferenceFrame,
    build_spatial_scene_state,
)


def _hands() -> dict[str, object]:
    points = [[1.0, 2.0, 3.0] for _ in range(21)]
    return {
        "frame_index": 7,
        "hands": [{"handedness": "left", "keypoints_3d_camera_m": points}],
    }


def test_world_scene_uses_first_vio_pose_as_origin_and_camera_extrinsic() -> None:
    first = VioPose(1_000, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    current = VioPose(2_000, (2.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    state = build_spatial_scene_state(
        SpatialReferenceFrame.WORLD,
        hand_result=_hands(),
        vio_first_pose=first,
        vio_pose=current,
        transform_camera_to_imu=(
            (1.0, 0.0, 0.0, 0.1),
            (0.0, 1.0, 0.0, 0.2),
            (0.0, 0.0, 1.0, 0.3),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )

    np.testing.assert_allclose(state.left_hand_points_m[0], (2.1, 2.2, 3.3))
    np.testing.assert_allclose(state.head_axes_m[0], (1.0, 0.0, 0.0))
    assert state.head_pose_source == "Basalt VIO"
    assert state.latest_frame_index == 7
    assert state.show_ground


def test_camera_scene_hides_head_and_leaves_hand_in_camera_coordinates() -> None:
    state = build_spatial_scene_state(
        SpatialReferenceFrame.CAMERA,
        hand_result=_hands(),
        vio_pose=VioPose(2_000, (9.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        vio_first_pose=VioPose(1_000, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    )

    assert state.left_hand_points_m[0] == (1.0, 2.0, 3.0)
    assert not state.head_axes_m
    assert not state.show_ground
    assert state.camera_axes_m[0] == (0.0, 0.0, 0.0)


def test_world_scene_falls_back_to_imu_orientation_without_translation() -> None:
    state = build_spatial_scene_state(
        SpatialReferenceFrame.WORLD,
        hand_result=_hands(),
        imu_pose=type(
            "Pose", (), {"samples_received": 1, "quaternion_wxyz": (1.0, 0.0, 0.0, 0.0)}
        )(),
    )

    assert state.head_pose_source == "IMU 朝向，无平移"
    assert state.has_imu_pose
    assert state.head_axes_m[0] == (0.0, 0.0, 0.0)


def test_offline_world_scene_does_not_fabricate_pose_from_imu_or_identity() -> None:
    state = build_spatial_scene_state(
        SpatialReferenceFrame.WORLD,
        hand_result=_hands(),
        imu_pose=type(
            "Pose", (), {"samples_received": 1, "quaternion_wxyz": (1.0, 0.0, 0.0, 0.0)}
        )(),
        allow_imu_world_fallback=False,
    )

    assert state.head_pose_source == "世界坐标不可用"
    assert state.left_hand_points_m == ()
    assert state.head_axes_m == ()
    assert state.show_ground is False
