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


def test_placeholder_calibration_matches_protocol() -> None:
    calibration = CalibrationSnapshot.placeholder(640, 480)

    assert calibration.camera.resolution == (640, 480)
    assert calibration.camera.intrinsics == (1.0, 1.0, 0.0, 0.0)
    assert calibration.camera.distortion_coeffs == (0.0, 0.0, 0.0, 0.0)
    assert calibration.T_cam_imu == (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    assert set(calibration.imu.model_dump().values()) == {None}
