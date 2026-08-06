from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hand_tracking import (
    Handedness,
    HandTrackingResult,
    MetricDepthStatus,
    OfflineHandTemporalConfig,
    OfflineHandTemporalProcessor,
    ReconstructionBackend,
    TemporalSource,
    TrackedHand,
)
from hand_tracking.models import readonly_float_array
from schemas import VioPose, VioTrajectory


def _config(**changes: object) -> OfflineHandTemporalConfig:
    return OfflineHandTemporalConfig.model_validate(
        {
            "confidence_threshold": 0.3,
            "interpolation_max_gap_frames": 2,
            "minimum_segment_frames": 1,
            "grasp_smoothing_window_frames": 1,
            "grasp_flicker_max_frames": 1,
            "sg_window_frames": 5,
            "sg_polyorder": 2,
            "orientation_ema_alpha": 0.15,
            "minimum_smoothing_frames": 3,
            "smoothing_fill_max_gap_frames": 1,
            "maximum_vio_pose_gap_ms": 100,
            **changes,
        }
    )


def _hand(
    side: Handedness = Handedness.RIGHT,
    *,
    confidence: float = 0.9,
    offset_x: float = 0.0,
    grasping: bool = False,
) -> TrackedHand:
    points = np.zeros((21, 3), dtype=np.float32)
    points[5] = (offset_x, 0.0, 1.0)
    points[6] = (offset_x - 0.03, 0.05, 1.0)
    points[8] = (offset_x + 0.03, 0.05, 1.0)
    points[11] = (offset_x, 0.07, 1.0)
    points[20] = (offset_x, 0.03, 1.0)
    points[0] = (offset_x - 0.02, 0.12, 1.0)
    points[1] = (offset_x + 0.02, 0.12, 1.0)
    return TrackedHand(
        handedness=side,
        confidence=confidence,
        detector_confidence=confidence,
        reconstruction_quality=confidence,
        depth_score=1.0,
        coverage_score=0.9,
        compactness_score=0.9,
        reconstruction_backend=ReconstructionBackend.HAMER,
        metric_depth_status=MetricDepthStatus.MODEL_ESTIMATED,
        bbox_xyxy_px=(10.0 + offset_x, 20.0, 30.0 + offset_x, 40.0),
        keypoints_2d_px=readonly_float_array(np.zeros((21, 2)), (21, 2)),
        keypoints_3d_camera_m=readonly_float_array(points, (21, 3)),
        joint_angles_degrees={},
        grasp_ratio=0.5 if grasping else 1.5,
        is_grasping=grasping,
    )


def _result(
    frame_index: int,
    *hands: TrackedHand,
    sequence_id: str = "clip-1",
    session_time_ns: int | None = None,
) -> HandTrackingResult:
    return HandTrackingResult(
        schema_version="1.0",
        session_id="session-1",
        sequence_id=sequence_id,
        frame_index=frame_index,
        session_time_ns=(frame_index + 1) * 100_000_000
        if session_time_ns is None
        else session_time_ns,
        timestamp_uncertainty_ns=0,
        image_width_px=640,
        image_height_px=480,
        source_rotation_degrees=0,
        detector_backend="test",
        requested_device="cpu",
        execution_device="cpu",
        hamer_loaded=True,
        inference_duration_ns=0,
        hands=tuple(hands),
    )


def _processor(**changes: object) -> OfflineHandTemporalProcessor:
    return OfflineHandTemporalProcessor(_config(**changes), grasp_ratio_threshold=1.0)


def test_filters_confidence_interpolates_short_gap_and_keeps_raw_immutable() -> None:
    raw = (
        _result(0, _hand(offset_x=0.0), _hand(Handedness.LEFT, confidence=0.2)),
        _result(1),
        _result(2, _hand(offset_x=0.2)),
    )

    output = _processor().process_clip(raw, trajectory=None, transform_camera_to_imu=None)

    assert output.raw_results == raw
    assert len(output.raw_results[1].hands) == 0
    assert output.stats.confidence_rejected == 1
    assert output.stats.interpolated_frames == 1
    interpolated = output.final_results[1].hands[0]
    assert interpolated.temporal is not None
    assert interpolated.temporal.source is TemporalSource.INTERPOLATED
    assert interpolated.temporal.interpolation_source_frames == (0, 2)
    assert interpolated.keypoints_3d_camera_m[5, 0] == pytest.approx(0.1)


def test_short_segments_include_sequence_edges_and_hands_are_independent() -> None:
    raw = (
        _result(0, _hand(Handedness.LEFT)),
        _result(1),
        _result(2, _hand(Handedness.RIGHT)),
        _result(3, _hand(Handedness.RIGHT)),
    )

    output = _processor(
        interpolation_max_gap_frames=0,
        minimum_segment_frames=2,
    ).process_clip(raw, trajectory=None, transform_camera_to_imu=None)

    assert output.final_results[0].hands == ()
    assert [len(result.hands) for result in output.final_results] == [0, 0, 1, 1]
    assert output.stats.suppressed_frames == 1


def test_processor_rejects_cross_clip_input() -> None:
    with pytest.raises(ValueError, match="clip boundaries"):
        _processor().process_clip(
            (_result(0), _result(1, sequence_id="clip-2")),
            trajectory=None,
            transform_camera_to_imu=None,
        )


def test_flicker_suppression_does_not_flip_unbracketed_edge_run() -> None:
    raw = tuple(
        _result(index, _hand(grasping=state))
        for index, state in enumerate((True, False, False, True, False))
    )

    output = _processor(grasp_flicker_max_frames=2).process_clip(
        raw, trajectory=None, transform_camera_to_imu=None
    )

    states = [result.hands[0].is_grasping for result in output.final_results]
    assert states == [True, True, True, True, False]


def test_grasp_smoothing_uses_ratio_and_preserves_constant_segment_edges() -> None:
    stale_state = replace(_hand(grasping=True), is_grasping=False)
    raw = tuple(_result(index, stale_state) for index in range(6))

    output = _processor(grasp_smoothing_window_frames=5).process_clip(
        raw,
        trajectory=None,
        transform_camera_to_imu=None,
    )

    assert all(result.hands[0].is_grasping for result in output.final_results)


def test_world_transform_and_velocity_use_real_session_time_delta() -> None:
    raw = (
        _result(0, _hand(), session_time_ns=1_000_000_000),
        _result(1, _hand(), session_time_ns=1_500_000_000),
    )
    trajectory = VioTrajectory(
        (
            VioPose(1_000_000_000, (2.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            VioPose(1_500_000_000, (3.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        )
    )
    transform_camera_to_imu = np.eye(4)
    transform_camera_to_imu[0, 3] = 1.0

    output = _processor(minimum_smoothing_frames=3).process_clip(
        raw,
        trajectory=trajectory,
        transform_camera_to_imu=transform_camera_to_imu,
    )

    first = output.final_results[0].hands[0]
    second = output.final_results[1].hands[0]
    assert first.kinematics is not None
    assert second.kinematics is not None
    assert first.kinematics.wrist_pose_raw_world[0, 3] == pytest.approx(3.0)
    assert second.kinematics.wrist_linear_velocity_raw_m_s[0] == pytest.approx(2.0)
    assert second.temporal is not None
    assert second.temporal.world_kinematics_available is True
    assert second.temporal.kinematics_optimized is False
    assert output.partial_world_coverage is False


def test_vio_gap_keeps_camera_result_without_fake_world_coordinates() -> None:
    raw = (_result(0, _hand()), _result(1, _hand()))
    trajectory = VioTrajectory(
        (VioPose(9_000_000_000, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),)
    )

    output = _processor().process_clip(
        raw,
        trajectory=trajectory,
        transform_camera_to_imu=np.eye(4),
    )

    assert output.partial_world_coverage is True
    assert output.stats.vio_matched_frames == 0
    for result in output.final_results:
        hand = result.hands[0]
        assert hand.kinematics is None
        assert hand.temporal is not None
        assert hand.temporal.world_kinematics_available is False


def test_savgol_and_ema_return_orthonormal_smoothed_rotations() -> None:
    raw = tuple(
        _result(index, _hand(offset_x=0.05 * index)) for index in range(5)
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

    output = _processor().process_clip(
        raw,
        trajectory=trajectory,
        transform_camera_to_imu=np.eye(4),
    )

    assert output.stats.world_optimized_frames == 5
    for result in output.final_results:
        hand = result.hands[0]
        assert hand.kinematics is not None
        rotation = hand.kinematics.midpoint_pose_optimized_world[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-5)


def test_savgol_fills_only_bounded_internal_vio_gaps_without_faking_world_data() -> None:
    raw = tuple(
        _result(index, _hand(offset_x=0.05 * index + (0.02 if index % 2 else -0.02)))
        for index in range(8)
    )
    trajectory = VioTrajectory(
        tuple(
            VioPose(
                result.session_time_ns,
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
            )
            for index, result in enumerate(raw)
            if index != 3
        )
    )

    without_fill = _processor(
        maximum_vio_pose_gap_ms=1,
        minimum_smoothing_frames=6,
        smoothing_fill_max_gap_frames=0,
    ).process_clip(raw, trajectory=trajectory, transform_camera_to_imu=np.eye(4))
    with_fill = _processor(
        maximum_vio_pose_gap_ms=1,
        minimum_smoothing_frames=6,
        smoothing_fill_max_gap_frames=1,
    ).process_clip(raw, trajectory=trajectory, transform_camera_to_imu=np.eye(4))

    assert without_fill.stats.world_optimized_frames == 0
    assert with_fill.stats.world_optimized_frames == 7
    assert with_fill.partial_world_coverage is True
    missing = with_fill.final_results[3].hands[0]
    assert missing.kinematics is None
    assert missing.temporal is not None
    assert missing.temporal.world_kinematics_available is False
    assert all(
        result.hands[0].temporal is not None
        and result.hands[0].temporal.kinematics_optimized
        for index, result in enumerate(with_fill.final_results)
        if index != 3
    )


def test_temporal_stats_record_each_stage_duration() -> None:
    output = _processor().process_clip(
        (_result(0, _hand()),),
        trajectory=None,
        transform_camera_to_imu=None,
    )

    stage_total = sum(
        getattr(output.stats, name)
        for name in (
            "confidence_filter_duration_ns",
            "interpolation_duration_ns",
            "segment_suppression_duration_ns",
            "grasp_smoothing_duration_ns",
            "world_mapping_duration_ns",
            "kinematic_optimization_duration_ns",
        )
    )
    assert stage_total >= 0
    assert output.stats.temporal_processing_duration_ns >= stage_total


def test_json_contains_temporal_provenance_and_kinematics() -> None:
    result = _result(0, _hand())
    trajectory = VioTrajectory(
        (
            VioPose(
                result.session_time_ns,
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
            ),
        )
    )
    final = _processor().process_clip(
        (result,), trajectory=trajectory, transform_camera_to_imu=np.eye(4)
    ).final_results[0]

    payload = final.to_json_dict()["hands"][0]
    assert payload["temporal"]["temporal_source"] == "observed"
    assert payload["temporal"]["world_kinematics_available"] is True
    assert len(payload["kinematics"]["keypoints_3d_world_m"]) == 21


def test_raw_arrays_are_not_mutated_by_final_processing() -> None:
    raw_hand = _hand(offset_x=0.2)
    original = raw_hand.keypoints_3d_camera_m.copy()
    raw = (_result(0, raw_hand),)

    _processor().process_clip(raw, trajectory=None, transform_camera_to_imu=None)

    np.testing.assert_array_equal(raw_hand.keypoints_3d_camera_m, original)
    assert raw_hand.temporal is None
    assert raw_hand.kinematics is None


def test_vio_translation_does_not_modify_camera_keypoints() -> None:
    raw = _result(0, _hand())
    trajectory = VioTrajectory(
        (
            VioPose(
                raw.session_time_ns,
                (5.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
            ),
        )
    )
    final = _processor().process_clip(
        (raw,), trajectory=trajectory, transform_camera_to_imu=np.eye(4)
    ).final_results[0]

    np.testing.assert_array_equal(
        final.hands[0].keypoints_3d_camera_m,
        raw.hands[0].keypoints_3d_camera_m,
    )
    assert final.hands[0] is not raw.hands[0]
