from __future__ import annotations

import hashlib
import importlib
import os
import urllib.request
from pathlib import Path

import av
import cv2
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
    HandTrackingResult,
    HumanEgoHandTrackingPipeline,
    ReconstructionBackend,
)

RUN_MODEL_EVAL = os.environ.get("EGOGLASS_RUN_HAND_MODEL_EVAL") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_MODEL_EVAL,
    reason="set EGOGLASS_RUN_HAND_MODEL_EVAL=1 in the egoglass Conda environment",
)

REPOSITORY = Path(__file__).parents[1]
MODEL_CONFIG = REPOSITORY / "config" / "hand-tracking.yaml"
TEASER_URL = (
    "https://raw.githubusercontent.com/geopavlakos/hamer/"
    "3a01849f4148352e9260b69bf28b65d1671a4905/assets/teaser.jpg"
)
TEASER_SHA256 = "f04f2c0f9d49b88c01a25b1f20f0c30fc211c4fe9dfe4faa2d534d73aba99349"
RECORDING = (
    REPOSITORY
    / "local-data"
    / "recordings"
    / "19e082cc36ea490d98f2611ee736d0a9"
    / "media"
    / "4c3aaec3025547409d7d995d6ac9e5ce.mp4"
)


@pytest.fixture(scope="module")
def model_pipeline() -> HumanEgoHandTrackingPipeline:
    pipeline = HumanEgoHandTrackingPipeline.from_config_file(str(MODEL_CONFIG))
    runtime_module = importlib.import_module(
        "perception.spatial_perception.hand_tracking.humanego_hamer"
    )
    runtime_path = Path(runtime_module.__file__).resolve()
    assert runtime_path.is_relative_to(REPOSITORY / "src")
    assert "reference_code" not in runtime_path.parts
    return pipeline


def test_public_hand_sample_runs_vitpose_and_cuda_hamer(
    model_pipeline: HumanEgoHandTrackingPipeline,
) -> None:
    teaser = _load_public_teaser()
    crop = teaser[0:980, 2480:3700]
    result = model_pipeline.process_frame(_bundle(crop, frame_index=0))

    print(
        "hand_model_positive "
        f"hands={len(result.hands)} "
        f"latency_ms={result.inference_duration_ns / 1_000_000:.3f} "
        f"confidences={[round(hand.confidence, 4) for hand in result.hands]}"
    )

    assert result.hamer_loaded is True
    assert result.execution_device.startswith("cuda")
    assert result.detector_backend == "vitpose-h+yolov8s"
    assert result.inference_duration_ns < 30_000_000_000
    assert result.hands
    assert all(
        hand.reconstruction_backend is ReconstructionBackend.HAMER
        for hand in result.hands
    )
    assert all(0.0 <= hand.confidence <= 1.0 for hand in result.hands)


def test_glass3_no_hand_recording_has_low_false_positive_rate(
    model_pipeline: HumanEgoHandTrackingPipeline,
) -> None:
    if not RECORDING.is_file():
        pytest.skip("local Glass3 evaluation recording is unavailable")
    sampled_indices = {0, 134, 268, 402, 536}
    results: list[HandTrackingResult] = []
    with av.open(str(RECORDING)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index not in sampled_indices:
                continue
            image = frame.to_ndarray(format="bgr24")
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            results.append(model_pipeline.process_frame(_bundle(image, frame_index=index)))
            if len(results) == len(sampled_indices):
                break

    assert len(results) == len(sampled_indices)
    false_positive_frames = sum(bool(result.hands) for result in results)
    latencies_ms = [result.inference_duration_ns / 1_000_000 for result in results]
    frame_details = [
        (
            result.frame_index,
            [
                (
                    hand.handedness.value,
                    hand.reconstruction_backend.value,
                    round(hand.confidence, 4),
                    tuple(round(value, 1) for value in hand.bbox_xyxy_px),
                )
                for hand in result.hands
            ],
        )
        for result in results
    ]
    print(
        "hand_model_glass3_no_hand "
        f"false_positive_frames={false_positive_frames}/{len(results)} "
        f"mean_latency_ms={sum(latencies_ms) / len(latencies_ms):.3f} "
        f"max_latency_ms={max(latencies_ms):.3f} "
        f"frames={frame_details}"
    )
    assert false_positive_frames / len(results) <= 0.2
    assert max(result.inference_duration_ns for result in results) < 30_000_000_000


def _load_public_teaser() -> np.ndarray:
    cache_path = REPOSITORY / "local-data" / "evals" / "hamer-teaser.jpg"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.is_file():
        urllib.request.urlretrieve(TEASER_URL, cache_path)
    digest = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    if digest != TEASER_SHA256:
        raise AssertionError("HaMeR public teaser hash changed")
    image = cv2.imread(str(cache_path), cv2.IMREAD_COLOR)
    if image is None:
        raise AssertionError("HaMeR public teaser could not be decoded")
    return image


def _bundle(image_bgr: np.ndarray, *, frame_index: int) -> PreparedFrameBundle:
    height, width = image_bgr.shape[:2]
    focal = float(max(width, height))
    camera_matrix = (
        (focal, 0.0, width / 2.0),
        (0.0, focal, height / 2.0),
        (0.0, 0.0, 1.0),
    )
    calibration = SensorCalibration(
        schema_version="1.0",
        profile_name="hand-model-eval",
        capture_config_id="model-eval",
        source_width=width,
        source_height=height,
        rotation_degrees=0,
        calibrated_width=width,
        calibrated_height=height,
        distortion_model="opencv_radtan",
        camera_matrix=camera_matrix,
        distortion_coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),
        rectified_camera_matrix=camera_matrix,
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
            source="model evaluation fixture",
            tool="pytest",
            tool_version="1",
            evidence_id="hand-model-eval-v1",
        ),
    )
    immutable_image = np.ascontiguousarray(image_bgr, dtype=np.uint8)
    immutable_image.setflags(write=False)
    return PreparedFrameBundle(
        session_id="hand-model-eval",
        sequence_id="sample",
        frame_index=frame_index,
        session_time_ns=frame_index * 33_333_333,
        timestamp_uncertainty_ns=0,
        timestamp_status=TimeStatus.VERIFIED,
        timestamp_semantic=TimestampSemantic.MEDIA_PRESENTATION,
        clock_mapping_id="hand-model-eval",
        image_bgr=immutable_image,
        imu_samples=(),
        calibration=calibration,
    )
