from __future__ import annotations

import numpy as np

from object_tracking import MultiViewTriangulator
from object_tracking.config import ObjectTrackingConfig
from object_tracking.triangulator import CameraObservation
from phase_analysis import PhaseAnalysisConfig, PhaseAnalysisService
from schemas import MotionPhase
from tests.test_phase_analysis import _frames


def test_phase_proposal_matches_fixed_manipulation_golden_interval() -> None:
    result = PhaseAnalysisService(
        PhaseAnalysisConfig(
            minimum_segment_frames=3,
            precontact_window_frames=5,
            minimum_object_window_frames=10,
            finished_trailing_frames=5,
        )
    ).analyze("eval", _frames())
    predicted = {
        frame.frame_index
        for frame in result.frames
        if frame.phase is MotionPhase.MANIPULATION
    }
    golden = set(range(30, 60))
    intersection = len(predicted & golden)
    union = len(predicted | golden)

    assert intersection / union >= 0.95
    assert result.object_centric_windows[0].reference_frame_index < min(golden)


def test_object_triangulation_meets_reprojection_and_position_thresholds() -> None:
    intrinsics = np.asarray(
        ((520.0, 0.0, 320.0), (0.0, 520.0, 240.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    expected = np.asarray(
        ((-0.15, -0.1, 1.8), (0.15, -0.1, 1.8), (0.15, 0.15, 1.8)),
        dtype=np.float64,
    )
    cameras: list[CameraObservation] = []
    tracks: list[np.ndarray] = []
    for index, x_offset in enumerate((0.0, 0.08, 0.16, 0.24, 0.32)):
        camera_to_world = np.eye(4, dtype=np.float64)
        camera_to_world[0, 3] = x_offset
        cameras.append(
            CameraObservation(index, index * 50_000_000, camera_to_world, intrinsics)
        )
        camera_points = expected - camera_to_world[:3, 3]
        projected = (intrinsics @ camera_points.T).T
        tracks.append(projected[:, :2] / projected[:, 2:3])
    result = MultiViewTriangulator(
        ObjectTrackingConfig(triangulation_frame_stride=1)
    ).triangulate(
        "obj1",
        tuple(cameras),
        np.asarray(tracks, dtype=np.float32),
        np.ones((len(cameras), len(expected)), dtype=np.float32),
        0,
        "pca1",
    )
    reconstructed = np.asarray(result.points_world_m)
    position_rmse_m = float(np.sqrt(np.mean((reconstructed - expected) ** 2)))

    assert result.mean_reprojection_error_px < 0.5
    assert position_rmse_m < 0.01
