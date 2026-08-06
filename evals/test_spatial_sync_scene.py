"""Periodic contract checks for reference-frame and pose freshness behavior."""

from pathlib import Path

from schemas import VioPose, VioTrajectory
from ui.presentation.spatial_scene import SpatialReferenceFrame, build_spatial_scene_state
from ui.processing.vio import VioRunInfo, VioRunState


def test_vio_pose_older_than_100ms_is_not_viewable() -> None:
    info = VioRunInfo(
        run_id="run",
        session_id="session",
        clip_id=None,
        state=VioRunState.COMPLETED,
        output_directory=Path("."),
        started_at_unix_ns=0,
        completed_at_unix_ns=1,
        trajectory=VioTrajectory((VioPose(0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),)),
    )

    assert info.pose_at(100_000_001) is None


def test_reference_frame_switch_does_not_change_hand_payload() -> None:
    result = {
        "frame_index": 1,
        "hands": [{"handedness": "right", "keypoints_3d_camera_m": [[0.0, 0.0, 1.0]] * 21}],
    }
    camera = build_spatial_scene_state(SpatialReferenceFrame.CAMERA, hand_result=result)
    world = build_spatial_scene_state(SpatialReferenceFrame.WORLD, hand_result=result)
    assert camera.right_hand_points_m[0] == (0.0, 0.0, 1.0)
    assert world.right_hand_points_m[0] == (0.0, 0.0, 1.0)
