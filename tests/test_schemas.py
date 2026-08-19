from pathlib import Path

import yaml

from schemas import (
    CalibrationSnapshot,
    CameraCalibration,
    CameraFrameRow,
    ImuCalibration,
    ImuSensorType,
    RecordingImuRow,
    RecordingLibrary,
    RecordingOutput,
    RecordingState,
    RecordingStatus,
    RecordingSummary,
)

CONFIG_ROOT = Path(__file__).parents[1] / "config"


def test_public_schema_package_exports_minimal_recording_contracts() -> None:
    for contract in (
        CalibrationSnapshot,
        CameraCalibration,
        CameraFrameRow,
        ImuCalibration,
        ImuSensorType,
        RecordingImuRow,
        RecordingLibrary,
        RecordingOutput,
        RecordingState,
        RecordingStatus,
        RecordingSummary,
    ):
        assert contract.__module__ == "schemas.recording"


def test_real_glass3_calibration_matches_protocol() -> None:
    payload = yaml.safe_load(
        (CONFIG_ROOT / "rokid-glass3-calibration.yaml").read_text(encoding="utf-8")
    )
    calibration = CalibrationSnapshot.model_validate(payload)

    assert calibration.camera.resolution == (640, 480)
    assert calibration.camera.intrinsics == (
        413.5336559113122,
        414.47606785606047,
        322.536987749118,
        241.1408173835428,
    )
    assert calibration.camera.distortion_coeffs == (
        -0.04381689910347928,
        0.038809576727509704,
        0.0025398843004178816,
        -0.001789994180868115,
    )
    assert calibration.T_cam_imu == (
        (
            0.9999938149586683,
            0.002239491076882919,
            0.0027119594622947154,
            -0.002707645684730107,
        ),
        (
            0.002197616594563916,
            -0.9998798213383645,
            0.015346444593567826,
            0.011123750134552288,
        ),
        (
            0.0027460017683653007,
            -0.015340389828055597,
            -0.9998785585830968,
            0.001363632624072346,
        ),
        (0.0, 0.0, 0.0, 1.0),
    )
    assert calibration.timeshift_cam_imu == -0.07100836969839247
    assert calibration.imu.gyro_noise_density == 9.141998445730967e-05
    assert calibration.imu.gyro_random_walk == 4.7278530101656415e-05
    assert calibration.imu.accel_noise_density == 0.0021079448412509427
    assert calibration.imu.accel_random_walk == 0.0011816409740879987
