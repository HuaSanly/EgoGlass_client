from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from perception.sensor_preprocessing import (
    CalibrationProvenance,
    ImuCalibration,
    PreparedFrameBundle,
    SensorCalibration,
    TimestampSemantic,
    TimeStatus,
)
from perception.spatial_perception.hand_tracking import (
    DetectedHand,
    Handedness,
    HandReconstruction,
    HandTrackingConfig,
    HandTrackingError,
    HumanEgoHandTrackingPipeline,
    MetricDepthStatus,
    ReconstructionBackend,
    remap_hamer_to_humanego_aria,
    rotated_image_bbox_to_source,
    rotated_image_points_to_source,
    source_image_dimensions,
)
from perception.spatial_perception.hand_tracking.models import (
    detector_bbox_has_valid_geometry,
    readonly_float_array,
)
from perception.spatial_perception.hand_tracking.weights import (
    ensure_hand_tracking_weights,
)

REPOSITORY = Path(__file__).parents[1]


class FakeDetector:
    name = "fake-vitpose"

    def __init__(self, detections: tuple[DetectedHand, ...]) -> None:
        self._detections = detections

    def detect(self, image_rgb: np.ndarray) -> tuple[DetectedHand, ...]:
        assert image_rgb.shape == (48, 64, 3)
        return self._detections


def test_hand_tracking_runtime_does_not_reference_humanego_checkout() -> None:
    runtime_files = list(
        (REPOSITORY / "src" / "perception" / "spatial_perception").rglob("*.py")
    )
    runtime_files.extend(
        (
            REPOSITORY / "scripts" / "download_hand_tracking_models.py",
            REPOSITORY / "scripts" / "setup_client.ps1",
        )
    )

    offenders = [
        path.relative_to(REPOSITORY).as_posix()
        for path in runtime_files
        if "reference_code" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


class FakeReconstructor:
    name = "hamer"
    execution_device = "cuda:0"
    is_available = True

    def __init__(self, reconstruction: HandReconstruction | None) -> None:
        self._reconstruction = reconstruction

    def predict_from_crop(
        self,
        image_rgb: np.ndarray,
        detection: DetectedHand,
    ) -> HandReconstruction | None:
        return self._reconstruction


def _config(tmp_path: Path, **updates: object) -> HandTrackingConfig:
    values: dict[str, object] = {
        "schema_version": "1.0",
        "model_directory": tmp_path,
        "device": "cuda",
        "require_cuda": True,
        "require_hamer": True,
        "download_models": False,
    }
    values.update(updates)
    return HandTrackingConfig.model_validate(values)


def _bundle() -> PreparedFrameBundle:
    calibration = SensorCalibration(
        schema_version="1.0",
        profile_name="tiny-hand-test",
        capture_config_id="tiny",
        source_width=64,
        source_height=48,
        rotation_degrees=0,
        calibrated_width=64,
        calibrated_height=48,
        distortion_model="opencv_radtan",
        camera_matrix=((100.0, 0.0, 32.0), (0.0, 100.0, 24.0), (0.0, 0.0, 1.0)),
        distortion_coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),
        rectified_camera_matrix=(
            (100.0, 0.0, 32.0),
            (0.0, 100.0, 24.0),
            (0.0, 0.0, 1.0),
        ),
        transform_camera_to_imu=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        imu=ImuCalibration(
            nominal_rate_hz=100.0,
            accelerometer_noise_density_m_s2_sqrt_hz=0.02,
            accelerometer_random_walk_m_s3_sqrt_hz=0.002,
            gyroscope_noise_density_rad_s_sqrt_hz=0.002,
            gyroscope_random_walk_rad_s2_sqrt_hz=0.0002,
        ),
        provenance=CalibrationProvenance(
            source="synthetic",
            tool="pytest",
            tool_version="1",
            evidence_id="hand-test",
        ),
    )
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image.setflags(write=False)
    return PreparedFrameBundle(
        session_id="session-1",
        sequence_id="clip-1",
        frame_index=7,
        session_time_ns=123_000_000,
        timestamp_uncertainty_ns=1_000_000,
        timestamp_status=TimeStatus.ESTIMATED,
        timestamp_semantic=TimestampSemantic.MEDIA_PRESENTATION,
        clock_mapping_id="mapping-1",
        image_bgr=image,
        imu_samples=(),
        calibration=calibration,
    )


def _keypoints_2d() -> np.ndarray:
    points = np.asarray(
        [[12.0 + index * 1.5, 10.0 + (index % 5) * 4.0] for index in range(21)],
        dtype=np.float32,
    )
    points[0] = (20.0, 24.0)
    points[9] = (40.0, 24.0)
    return points


def _keypoints_3d(depth: float = 0.6) -> np.ndarray:
    points = np.asarray(
        [[index * 0.004, (index % 5) * 0.006, depth] for index in range(21)],
        dtype=np.float32,
    )
    points[0] = (0.0, 0.0, depth)
    points[9] = (0.085, 0.0, depth)
    return points


def _detection(
    *,
    with_relative_3d: bool = False,
    confidence: float = 0.9,
    egocentric_fallback: bool = False,
) -> DetectedHand:
    relative = _keypoints_3d(0.0) if with_relative_3d else None
    return DetectedHand(
        bbox_xyxy_px=(8.0, 6.0, 56.0, 42.0),
        handedness=Handedness.RIGHT,
        confidence=confidence,
        keypoints_2d_px=readonly_float_array(_keypoints_2d(), (21, 2)),
        relative_keypoints_3d_m=(
            readonly_float_array(relative, (21, 3)) if relative is not None else None
        ),
        egocentric_fallback=egocentric_fallback,
    )


def _reconstruction(depth: float = 0.6) -> HandReconstruction:
    return HandReconstruction(
        keypoints_3d_m=readonly_float_array(_keypoints_3d(depth), (21, 3)),
        keypoints_2d_px=readonly_float_array(_keypoints_2d(), (21, 2)),
        confidence=0.8,
    )


def test_repository_config_is_pinned_and_resolves_local_model_directory() -> None:
    repository = Path(__file__).parents[1]
    config = HandTrackingConfig.load(repository / "config" / "hand-tracking.yaml")

    assert config.require_cuda is True
    assert config.require_hamer is True
    assert config.detector == "vitpose"
    assert config.model_directory == (
        repository / "local-data" / "models" / "hand-tracking"
    ).resolve()
    assert config.sources.hamer_code_revision == (
        "3a01849f4148352e9260b69bf28b65d1671a4905"
    )


def test_hamer_result_uses_humanego_joint_order_and_records_backend(tmp_path: Path) -> None:
    pipeline = HumanEgoHandTrackingPipeline(
        _config(tmp_path),
        FakeDetector((_detection(),)),
        FakeReconstructor(_reconstruction()),
    )

    result = pipeline.process_frame(_bundle())

    assert result.hamer_loaded is True
    assert result.execution_device == "cuda:0"
    assert result.detector_backend == "fake-vitpose"
    assert len(result.hands) == 1
    hand = result.hands[0]
    assert hand.reconstruction_backend is ReconstructionBackend.HAMER
    assert hand.metric_depth_status is MetricDepthStatus.MODEL_ESTIMATED
    assert hand.confidence == pytest.approx(0.72)
    np.testing.assert_allclose(hand.keypoints_3d_camera_m[0], _keypoints_3d()[4])
    np.testing.assert_allclose(hand.keypoints_3d_camera_m[5], _keypoints_3d()[0])
    assert hand.keypoints_3d_camera_m.flags.writeable is False
    assert len(hand.joint_angles_degrees) == 20


def test_invalid_hamer_depth_uses_humanego_physical_size_recovery(tmp_path: Path) -> None:
    pipeline = HumanEgoHandTrackingPipeline(
        _config(tmp_path),
        FakeDetector((_detection(),)),
        FakeReconstructor(_reconstruction(depth=4.0)),
    )

    hand = pipeline.process_frame(_bundle()).hands[0]

    assert hand.reconstruction_backend is ReconstructionBackend.HAMER
    assert hand.metric_depth_status is MetricDepthStatus.PHYSICAL_SIZE_ESTIMATED
    assert hand.keypoints_3d_camera_m[5, 2] == pytest.approx(0.425)
    assert hand.confidence == pytest.approx(0.9)


def test_depth_recovery_does_not_promote_low_confidence_false_positive(
    tmp_path: Path,
) -> None:
    pipeline = HumanEgoHandTrackingPipeline(
        _config(tmp_path),
        FakeDetector((_detection(confidence=0.1634),)),
        FakeReconstructor(_reconstruction(depth=4.0)),
    )

    result = pipeline.process_frame(_bundle())

    assert result.hands == ()


def test_egocentric_fallback_does_not_promote_detector_confidence(
    tmp_path: Path,
) -> None:
    pipeline = HumanEgoHandTrackingPipeline(
        _config(tmp_path),
        FakeDetector(
            (_detection(confidence=0.1634, egocentric_fallback=True),)
        ),
        FakeReconstructor(_reconstruction()),
    )

    result = pipeline.process_frame(_bundle())

    assert result.hands == ()


def test_mediapipe_3d_is_used_only_as_explicit_crop_fallback(tmp_path: Path) -> None:
    pipeline = HumanEgoHandTrackingPipeline(
        _config(tmp_path),
        FakeDetector((_detection(with_relative_3d=True),)),
        FakeReconstructor(None),
    )

    hand = pipeline.process_frame(_bundle()).hands[0]

    assert hand.reconstruction_backend is ReconstructionBackend.MEDIAPIPE
    assert hand.metric_depth_status is MetricDepthStatus.PHYSICAL_SIZE_ESTIMATED


def test_pipeline_rejects_missing_required_hamer(tmp_path: Path) -> None:
    with pytest.raises(HandTrackingError, match="requires HaMeR"):
        HumanEgoHandTrackingPipeline(
            _config(tmp_path),
            FakeDetector((_detection(with_relative_3d=True),)),
            None,
        )


def test_result_payload_is_json_serializable_and_keeps_camera_frame_name(
    tmp_path: Path,
) -> None:
    pipeline = HumanEgoHandTrackingPipeline(
        _config(tmp_path),
        FakeDetector((_detection(),)),
        FakeReconstructor(_reconstruction()),
    )

    payload = pipeline.process_frame(_bundle()).to_json_dict()
    encoded = json.dumps(payload)

    assert "keypoints_3d_camera_m" in encoded
    assert "keypoints_3d_world" not in encoded
    assert payload["joint_order"][5] == "wrist"
    assert payload["source_image_width_px"] == 64
    assert payload["source_image_height_px"] == 48
    assert payload["hands"][0]["source_keypoints_2d_px"] == payload["hands"][0][
        "keypoints_2d_px"
    ]
    assert payload["hands"][0]["source_bbox_xyxy_px"] == payload["hands"][0][
        "bbox_xyxy_px"
    ]


@pytest.mark.parametrize(
    ("rotation", "point", "expected_point", "expected_bbox"),
    [
        (0, (1.0, 1.0), (1.0, 1.0), (0.25, 0.5, 2.25, 2.0)),
        (90, (1.0, 2.0), (2.0, 1.0), (0.5, 0.75, 2.0, 2.75)),
        (180, (1.0, 1.0), (2.0, 1.0), (1.75, 1.0, 3.75, 2.5)),
        (270, (1.0, 2.0), (1.0, 1.0), (2.0, 0.25, 3.5, 2.25)),
    ],
)
def test_rotated_result_coordinates_map_back_to_decoded_preview(
    rotation: int,
    point: tuple[float, float],
    expected_point: tuple[float, float],
    expected_bbox: tuple[float, float, float, float],
) -> None:
    image_width, image_height = (3, 4) if rotation in {90, 270} else (4, 3)

    mapped = rotated_image_points_to_source(
        np.asarray([point], dtype=np.float32),
        image_width_px=image_width,
        image_height_px=image_height,
        rotation_degrees=rotation,
    )
    bbox = rotated_image_bbox_to_source(
        (0.25, 0.5, 2.25, 2.0),
        image_width_px=image_width,
        image_height_px=image_height,
        rotation_degrees=rotation,
    )

    assert source_image_dimensions(image_width, image_height, rotation) == (4, 3)
    np.testing.assert_allclose(mapped[0], expected_point)
    assert bbox == pytest.approx(expected_bbox)


def test_remap_rejects_non_21_joint_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        remap_hamer_to_humanego_aria(np.zeros((20, 3), dtype=np.float32))


def test_readonly_array_factory_freezes_normal_writable_input() -> None:
    source = np.zeros((21, 2), dtype=np.float32)

    result = readonly_float_array(source, (21, 2))

    assert result.flags.writeable is False
    assert source.flags.writeable is True


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        ((282.7, 632.6, 354.2, 681.7), False),
        ((18.2, 164.9, 720.0, 744.0), False),
        ((105.2, 338.9, 450.0, 621.8), True),
    ],
)
def test_bbox_geometry_rejects_observed_glass3_false_positives(
    tmp_path: Path,
    bbox: tuple[float, float, float, float],
    expected: bool,
) -> None:
    assert detector_bbox_has_valid_geometry(
        bbox,
        image_width=720,
        image_height=1280,
        config=_config(tmp_path),
    ) is expected


def test_weight_resolver_fails_without_network_when_download_is_disabled(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, download_models=False)

    with pytest.raises(HandTrackingError, match="model files are missing"):
        ensure_hand_tracking_weights(config)
