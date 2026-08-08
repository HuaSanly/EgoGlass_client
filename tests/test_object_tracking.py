from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from hand_tracking.models import Handedness
from object_tracking import (
    ContourKeypointSelector,
    MultiViewTriangulator,
    ObjectFrameInput,
    ObjectPoseLatcher,
    OfflineObjectProcessing,
    SegmentationPrediction,
    TaskProfile,
)
from object_tracking.config import ObjectTrackingConfig, ObjectTrackingError
from object_tracking.dino_sam import _merge_sam_masks
from object_tracking.triangulator import CameraObservation
from schemas import BoundingBox, ObjectCentricWindow, ObjectTriangulation, VioTrajectory


def test_contour_keypoints_are_inside_a_cleaned_mask() -> None:
    mask = np.zeros((100, 120), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (100, 80), 255, -1)
    points = ContourKeypointSelector(ObjectTrackingConfig(keypoint_count=20)).select(mask)
    assert points.shape == (20, 2)
    assert all(mask[int(point[1]), int(point[0])] > 0 for point in points)


def test_sam2_mask_batches_are_merged_across_supported_versions() -> None:
    masks = np.zeros((2, 6, 8), dtype=np.uint8)
    masks[0, 1:3, 2:4] = 1
    masks[1, 4:6, 5:7] = 1

    expected = _merge_sam_masks(masks)

    assert expected.shape == (6, 8)
    assert np.array_equal(_merge_sam_masks(masks[:, None]), expected)
    assert set(np.unique(expected)) == {0, 255}
    with pytest.raises(ObjectTrackingError, match="unsupported mask shape"):
        _merge_sam_masks(np.zeros((6, 8), dtype=np.uint8))


def test_offline_object_pipeline_writes_replayable_stage_artifacts(tmp_path: Path) -> None:
    config = ObjectTrackingConfig(
        device="cpu",
        require_cuda=False,
        keypoint_count=3,
        morphology_close_kernel=1,
        morphology_erode_kernel=1,
        inner_edge_kernel=1,
    )
    pipeline = OfflineObjectProcessing(
        config,
        segmenter=_FakeSegmenter(),
        tracker=_FakePointTracker(),
        triangulator=_FakeTriangulator(),  # type: ignore[arg-type]
    )
    profile = TaskProfile(
        profile_id="test-object",
        display_name="Test object",
        object_prompts={"obj1": "test object ."},
    )
    frames = tuple(
        ObjectFrameInput(
            clip_id="clip",
            frame_index=index,
            session_time_ns=index + 1,
            image_bgr=np.zeros((32, 32, 3), dtype=np.uint8),
            intrinsics=np.eye(3, dtype=np.float64),
            hands=(),
        )
        for index in range(2)
    )
    window = ObjectCentricWindow(
        clip_id="clip",
        start_frame_index=0,
        reference_frame_index=0,
        end_frame_index_exclusive=2,
        start_session_time_ns=1,
        end_session_time_ns=2,
        evidence=("test",),
    )

    result = pipeline.run(
        "run",
        profile,
        (window,),
        {"clip": frames},
        VioTrajectory(()),
        np.eye(4, dtype=np.float64),
        tmp_path,
    )

    assert len(result.masks) == 2
    assert len(result.tracks) == 1
    assert len(result.triangulations) == 1
    assert len(result.poses) == 2
    for relative_path in (
        "object-result.json",
        "selected-keypoints.jsonl",
        "tracks.json",
        "triangulation.json",
        "object-qa.json",
        "masks/obj1-000000.png",
        "masks/obj1-000001.png",
    ):
        assert (tmp_path / relative_path).is_file()
    rejecting_pipeline = OfflineObjectProcessing(
        config.model_copy(update={"mask_min_area_ratio": 0.5}),
        segmenter=_FakeSegmenter(),
        tracker=_FakePointTracker(),
        triangulator=_FakeTriangulator(),  # type: ignore[arg-type]
    )
    with pytest.raises(ObjectTrackingError, match="mask area"):
        rejecting_pipeline.run(
            "run-rejected",
            profile,
            (window,),
            {"clip": frames},
            VioTrajectory(()),
            np.eye(4, dtype=np.float64),
            tmp_path / "rejected",
        )


class _FakeSegmenter:
    def segment(self, image_bgr: np.ndarray, _prompt: str) -> SegmentationPrediction:
        mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        mask[8:24, 8:24] = 255
        return SegmentationPrediction(
            mask,
            (BoundingBox(x1=8.0, y1=8.0, x2=24.0, y2=24.0, confidence=0.9),),
        )


class _FakePointTracker:
    def track(
        self,
        images_bgr: list[np.ndarray],
        initial_points_xy_px: np.ndarray,
        _reference_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        tracks = np.repeat(initial_points_xy_px[None], len(images_bgr), axis=0)
        visibility = np.ones(tracks.shape[:2], dtype=np.float32)
        return tracks.astype(np.float32), visibility


class _FakeTriangulator:
    def camera_observations(self, *_args: object) -> tuple[object, ...]:
        return (object(), object())

    def triangulate(self, object_id: str, *_args: object) -> ObjectTriangulation:
        identity = tuple(float(value) for value in np.eye(4).reshape(-1))
        return ObjectTriangulation(
            object_id=object_id,
            points_world_m=((0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.0, 0.1, 1.0)),
            transform_object_to_world=identity,
            transform_object_to_camera=identity,
            mean_reprojection_error_px=0.1,
            contributing_frame_count=2,
            valid_point_count=3,
            orientation_method="pca1",
        )


def test_triangulator_recovers_synthetic_world_points() -> None:
    intrinsics = np.array(((500.0, 0.0, 320.0), (0.0, 500.0, 240.0), (0.0, 0.0, 1.0)))
    points = np.array(
        ((-0.2, -0.1, 2.0), (0.2, -0.1, 2.0), (0.2, 0.2, 2.0), (-0.2, 0.2, 2.0)),
        dtype=np.float64,
    )
    observations: list[CameraObservation] = []
    tracks: list[np.ndarray] = []
    for index, x in enumerate((0.0, 0.1, 0.2, 0.3)):
        camera_to_world = np.eye(4)
        camera_to_world[0, 3] = x
        observations.append(
            CameraObservation(index, index * 100_000_000, camera_to_world, intrinsics)
        )
        camera_points = points - camera_to_world[:3, 3]
        pixels = (intrinsics @ camera_points.T).T
        tracks.append(pixels[:, :2] / pixels[:, 2:3])
    result = MultiViewTriangulator(ObjectTrackingConfig(triangulation_frame_stride=1)).triangulate(
        "obj1",
        tuple(observations),
        np.asarray(tracks, dtype=np.float32),
        np.ones((4, 4), dtype=np.float32),
        0,
        "pca1",
    )
    assert result.valid_point_count == 4
    assert result.mean_reprojection_error_px < 1e-3
    assert np.allclose(np.asarray(result.points_world_m), points, atol=1e-3)


def test_latcher_attaches_only_during_grasp() -> None:
    initial = np.eye(4)
    hand_pose = np.eye(4)
    hand_pose[0, 3] = 1.0
    hand = SimpleNamespace(
        handedness=Handedness.RIGHT,
        confidence=0.9,
        is_grasping=True,
        kinematics=SimpleNamespace(midpoint_pose_optimized_world=hand_pose),
    )
    latcher = ObjectPoseLatcher("obj1", initial)
    pose = latcher.update("clip", 0, 0, (hand,))
    hand_pose_next = hand_pose.copy()
    hand_pose_next[0, 3] = 2.0
    next_hand = SimpleNamespace(
        handedness=Handedness.RIGHT,
        confidence=0.9,
        is_grasping=True,
        kinematics=SimpleNamespace(midpoint_pose_optimized_world=hand_pose_next),
    )
    pose = latcher.update("clip", 1, 1, (next_hand,))
    assert pose.dynamic
    assert pose.source == "hand_latched"
    expected = hand_pose_next @ np.linalg.inv(hand_pose)
    assert np.allclose(np.asarray(pose.transform_object_to_world).reshape(4, 4), expected)

    released = SimpleNamespace(
        handedness=Handedness.RIGHT,
        confidence=0.9,
        is_grasping=False,
        kinematics=SimpleNamespace(midpoint_pose_optimized_world=hand_pose_next),
    )
    held = latcher.update("clip", 2, 2, (released,))
    assert not held.dynamic
    assert held.source == "hand_latched_hold"
    assert held.grasped_by == Handedness.RIGHT.value
    assert held.transform_object_to_world == pose.transform_object_to_world


def test_latcher_rejects_a_grasp_far_from_the_object() -> None:
    initial = np.eye(4)
    hand_pose = np.eye(4)
    hand_pose[0, 3] = 1.0
    hand = SimpleNamespace(
        handedness=Handedness.RIGHT,
        confidence=0.9,
        is_grasping=True,
        kinematics=SimpleNamespace(midpoint_pose_optimized_world=hand_pose),
    )
    latcher = ObjectPoseLatcher("obj1", initial, maximum_latch_distance_m=0.2)

    pose = latcher.update("clip", 0, 0, (hand,))

    assert pose.source == "static_triangulation"
    assert pose.grasped_by is None
