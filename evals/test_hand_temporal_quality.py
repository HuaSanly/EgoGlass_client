from __future__ import annotations

import numpy as np

from schemas import VioPose, VioTrajectory
from tests.test_hand_temporal_processing import _hand, _processor, _result


def _jitter(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.diff(values, n=2, axis=0) ** 2)))


def test_temporal_pipeline_reduces_fixed_noise_and_grasp_flicker() -> None:
    random = np.random.default_rng(20260806)
    frame_count = 41
    smooth_x = np.linspace(0.0, 0.4, frame_count)
    noisy_x = smooth_x + random.normal(0.0, 0.015, frame_count)
    grasp_states = [False] * frame_count
    for index in (10, 20, 21, 30):
        grasp_states[index] = True
    raw = tuple(
        _result(index, _hand(offset_x=float(noisy_x[index]), grasping=grasp_states[index]))
        for index in range(frame_count)
    )
    trajectory = VioTrajectory(
        tuple(
            VioPose(
                result.session_time_ns,
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
            )
            for result in raw
        )
    )

    output = _processor(
        interpolation_max_gap_frames=20,
        minimum_segment_frames=10,
        grasp_smoothing_window_frames=5,
        grasp_flicker_max_frames=5,
        sg_window_frames=21,
        minimum_smoothing_frames=6,
    ).process_clip(raw, trajectory=trajectory, transform_camera_to_imu=np.eye(4))

    raw_positions = np.stack(
        [result.hands[0].kinematics.wrist_pose_raw_world[:3, 3] for result in output.final_results]
    )
    optimized_positions = np.stack(
        [
            result.hands[0].kinematics.wrist_pose_optimized_world[:3, 3]
            for result in output.final_results
        ]
    )
    optimized_grasp = [result.hands[0].is_grasping for result in output.final_results]

    assert _jitter(optimized_positions) < _jitter(raw_positions) * 0.35
    assert sum(
        first != second
        for first, second in zip(optimized_grasp, optimized_grasp[1:], strict=False)
    ) == 0
    assert output.stats.grasp_transitions_after < output.stats.grasp_transitions_before


def test_humanego_golden_cleanup_order_and_edge_flicker_fix() -> None:
    raw = (
        _result(0, _hand(grasping=True)),
        _result(1),
        _result(2, _hand(grasping=False)),
        _result(3),
        _result(4, _hand(grasping=True)),
        _result(5),
        _result(6),
        _result(7, _hand(grasping=False)),
    )

    output = _processor(
        interpolation_max_gap_frames=1,
        minimum_segment_frames=3,
        grasp_smoothing_window_frames=1,
        grasp_flicker_max_frames=3,
    ).process_clip(raw, trajectory=None, transform_camera_to_imu=None)

    assert [len(result.hands) for result in output.final_results] == [1, 1, 1, 1, 1, 0, 0, 0]
    assert [
        result.hands[0].temporal.source.value if result.hands else None
        for result in output.final_results
    ] == [
        "observed",
        "interpolated",
        "observed",
        "interpolated",
        "observed",
        None,
        None,
        None,
    ]
    assert [result.hands[0].is_grasping for result in output.final_results[:5]] == [
        True,
        True,
        True,
        True,
        True,
    ]
    assert output.stats.interpolated_frames == 2
    assert output.stats.suppressed_frames == 1


def test_short_vio_dropout_keeps_smoothing_quality_without_inventing_world_pose() -> None:
    random = np.random.default_rng(260806)
    frame_count = 31
    expected_x = np.linspace(0.0, 0.3, frame_count)
    noisy_x = expected_x + random.normal(0.0, 0.012, frame_count)
    raw = tuple(
        _result(index, _hand(offset_x=float(noisy_x[index])))
        for index in range(frame_count)
    )
    missing = {10, 20}
    trajectory = VioTrajectory(
        tuple(
            VioPose(
                result.session_time_ns,
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
            )
            for index, result in enumerate(raw)
            if index not in missing
        )
    )

    output = _processor(
        maximum_vio_pose_gap_ms=1,
        smoothing_fill_max_gap_frames=1,
        minimum_smoothing_frames=6,
        sg_window_frames=21,
    ).process_clip(raw, trajectory=trajectory, transform_camera_to_imu=np.eye(4))

    available = [index for index in range(frame_count) if index not in missing]
    raw_positions = np.stack(
        [output.final_results[index].hands[0].kinematics.wrist_pose_raw_world[:3, 3]
         for index in available]
    )
    optimized_positions = np.stack(
        [output.final_results[index].hands[0].kinematics.wrist_pose_optimized_world[:3, 3]
         for index in available]
    )
    assert _jitter(optimized_positions) < _jitter(raw_positions) * 0.4
    assert output.stats.world_optimized_frames == frame_count - len(missing)
    assert output.partial_world_coverage is True
    assert all(output.final_results[index].hands[0].kinematics is None for index in missing)
