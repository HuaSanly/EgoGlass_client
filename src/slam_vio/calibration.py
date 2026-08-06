"""Convert EgoGlass calibration into Basalt's cereal JSON format."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from sensor_preprocessing import SensorCalibration


def calibration_is_verified(calibration: SensorCalibration) -> bool:
    """Return whether provenance contains measured/validated evidence.

    The repository's sample files intentionally contain words such as
    ``sample`` and ``unmeasured``. They are useful for pipeline tests but are
    rejected by the VIO runner unless the caller explicitly opts in.
    """

    provenance = calibration.provenance
    text = " ".join(
        value.lower()
        for value in (
            provenance.source,
            provenance.tool,
            provenance.tool_version,
            provenance.evidence_id,
        )
    )
    return not any(marker in text for marker in ("sample", "unmeasured", "unverified", "synthetic"))


def _rotation_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert an SO(3) matrix to Basalt's ``qx,qy,qz,qw`` fields."""

    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / scale
        qx = 0.25 * scale
        qy = (rotation[0, 1] + rotation[1, 0]) / scale
        qz = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / scale
        qx = (rotation[0, 1] + rotation[1, 0]) / scale
        qy = 0.25 * scale
        qz = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / scale
        qx = (rotation[0, 2] + rotation[2, 0]) / scale
        qy = (rotation[1, 2] + rotation[2, 1]) / scale
        qz = 0.25 * scale
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    return (qx / norm, qy / norm, qz / norm, qw / norm)


def calibration_to_basalt_json(
    calibration: SensorCalibration,
    *,
    input_is_rectified: bool = True,
) -> dict[str, Any]:
    """Build a single-camera Basalt calibration payload.

    Prepared frames are rectified by default, so the exported camera uses the
    rectified pinhole matrix and has no distortion. If raw images are passed,
    the original OpenCV radial-tangential coefficients are padded to Basalt's
    eight-coefficient model instead.
    """

    matrix_name = "rectified_camera_matrix" if input_is_rectified else "camera_matrix"
    matrix = np.asarray(getattr(calibration, matrix_name), dtype=np.float64)
    if input_is_rectified:
        camera = {
            "camera_type": "pinhole",
            "intrinsics": {
                "fx": float(matrix[0, 0]),
                "fy": float(matrix[1, 1]),
                "cx": float(matrix[0, 2]),
                "cy": float(matrix[1, 2]),
            },
        }
    else:
        coeffs = (*calibration.distortion_coefficients, 0.0, 0.0, 0.0)
        camera = {
            "camera_type": "pinhole-radtan8",
            "intrinsics": {
                "fx": float(matrix[0, 0]),
                "fy": float(matrix[1, 1]),
                "cx": float(matrix[0, 2]),
                "cy": float(matrix[1, 2]),
                "k1": float(coeffs[0]),
                "k2": float(coeffs[1]),
                "p1": float(coeffs[2]),
                "p2": float(coeffs[3]),
                "k3": float(coeffs[4]),
                "k4": float(coeffs[5]),
                "k5": float(coeffs[6]),
                "k6": float(coeffs[7]),
                "rpmax": 1.0e10,
            },
        }

    # Prepared IMU rows are already mapped into calibrated body axes.
    transform = np.asarray(calibration.transform_camera_to_body, dtype=np.float64)
    qx, qy, qz, qw = _rotation_to_quaternion(transform[:3, :3])
    transform_payload = {
        "px": float(transform[0, 3]),
        "py": float(transform[1, 3]),
        "pz": float(transform[2, 3]),
        "qx": qx,
        "qy": qy,
        "qz": qz,
        "qw": qw,
    }
    noise = calibration.imu
    values = {
        "T_imu_cam": [transform_payload],
        "intrinsics": [camera],
        "resolution": [[calibration.calibrated_width, calibration.calibrated_height]],
        "calib_accel_bias": [0.0] * 9,
        "calib_gyro_bias": [0.0] * 12,
        "imu_update_rate": float(noise.nominal_rate_hz),
        "accel_noise_std": [float(noise.accelerometer_noise_density_m_s2_sqrt_hz)] * 3,
        "gyro_noise_std": [float(noise.gyroscope_noise_density_rad_s_sqrt_hz)] * 3,
        "accel_bias_std": [float(noise.accelerometer_random_walk_m_s3_sqrt_hz)] * 3,
        "gyro_bias_std": [float(noise.gyroscope_random_walk_rad_s2_sqrt_hz)] * 3,
        "cam_time_offset_ns": 0,
        # Basalt's cereal calibration schema always serializes one vignette
        # spline per camera. An empty list explicitly means no vignette model.
        "vignette": [],
    }
    return {"value0": values}
