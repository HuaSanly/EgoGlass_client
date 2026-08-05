"""Export prepared EgoGlass frames and IMU samples to Basalt's EuRoC layout."""

from __future__ import annotations

import csv
import json
from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2

from sensor_preprocessing import (
    ImuSensorType,
    PreparedFrameBundle,
    PreparedImuSample,
    SensorCalibration,
)

from .calibration import calibration_to_basalt_json
from .models import BasaltDataset, BasaltExportError


@dataclass(frozen=True, slots=True)
class _ImuRow:
    timestamp_ns: int
    gyro: tuple[float, float, float]
    accel: tuple[float, float, float]
    interpolated: bool


def _collapse_samples(
    samples: Iterable[PreparedImuSample],
    sensor_type: ImuSensorType,
) -> tuple[PreparedImuSample, ...]:
    """Sort one sensor stream and remove exact duplicate timestamps."""

    by_timestamp: dict[int, PreparedImuSample] = {}
    for sample in samples:
        if sample.sensor_type is sensor_type:
            by_timestamp[sample.session_time_ns] = sample
    return tuple(by_timestamp[timestamp] for timestamp in sorted(by_timestamp))


def _interpolate(
    samples: tuple[PreparedImuSample, ...],
    timestamp_ns: int,
) -> tuple[tuple[float, float, float] | None, bool]:
    """Interpolate a sensor at a union timestamp without extrapolation."""

    if not samples:
        return None, False
    times = [sample.session_time_ns for sample in samples]
    index = bisect_left(times, timestamp_ns)
    if index < len(samples) and times[index] == timestamp_ns:
        return samples[index].values, False
    if index == 0 or index == len(samples):
        return None, False
    before = samples[index - 1]
    after = samples[index]
    fraction = (timestamp_ns - before.session_time_ns) / (
        after.session_time_ns - before.session_time_ns
    )
    values = tuple(
        left + fraction * (right - left)
        for left, right in zip(before.values, after.values, strict=False)
    )
    return values, True


def synchronize_imu_samples(
    samples: Iterable[PreparedImuSample],
) -> tuple[tuple[_ImuRow, ...], int]:
    """Merge separate accelerometer/gyro streams using linear interpolation.

    Basalt's EuRoC reader requires one row containing both measurements. Union
    timestamps are retained when both channels can be evaluated; timestamps
    outside either channel's measured range are counted and omitted rather
    than extrapolated silently.
    """

    materialized = tuple(samples)
    accelerometer = _collapse_samples(materialized, ImuSensorType.ACCELEROMETER)
    gyroscope = _collapse_samples(materialized, ImuSensorType.GYROSCOPE)
    if not accelerometer or not gyroscope:
        raise BasaltExportError("Basalt EuRoC export requires accelerometer and gyro samples")
    timestamps = sorted(
        {sample.session_time_ns for sample in accelerometer}
        | {sample.session_time_ns for sample in gyroscope}
    )
    rows: list[_ImuRow] = []
    skipped = 0
    for timestamp_ns in timestamps:
        accel, accel_interpolated = _interpolate(accelerometer, timestamp_ns)
        gyro, gyro_interpolated = _interpolate(gyroscope, timestamp_ns)
        if accel is None or gyro is None:
            skipped += 1
            continue
        rows.append(
            _ImuRow(
                timestamp_ns=timestamp_ns,
                gyro=gyro,
                accel=accel,
                interpolated=accel_interpolated or gyro_interpolated,
            )
        )
    if not rows:
        raise BasaltExportError("accelerometer and gyro time ranges do not overlap")
    return tuple(rows), skipped


class BasaltEuRoCExporter:
    """Materialize immutable prepared bundles into a Basalt dataset."""

    def export(
        self,
        bundles: Iterable[PreparedFrameBundle],
        output_directory: str | Path,
        *,
        calibration: SensorCalibration,
        input_is_rectified: bool = True,
    ) -> BasaltDataset:
        """Write images, CSV indices, calibration, and an export manifest."""

        materialized = tuple(bundles)
        if not materialized:
            raise BasaltExportError("Basalt EuRoC export requires at least one frame")
        session_ids = {bundle.session_id for bundle in materialized}
        if len(session_ids) != 1:
            raise BasaltExportError("all prepared frames must belong to one session")
        sorted_bundles = tuple(
            sorted(
                materialized,
                key=lambda item: (
                    item.session_time_ns,
                    item.sequence_id,
                    item.frame_index,
                ),
            )
        )
        if any(
            current.session_time_ns <= previous.session_time_ns
            for previous, current in zip(sorted_bundles, sorted_bundles[1:], strict=False)
        ):
            raise BasaltExportError("prepared frame session times must strictly increase")

        root = Path(output_directory).resolve()
        cam_data = root / "mav0" / "cam0" / "data"
        imu_data = root / "mav0" / "imu0"
        cam_data.mkdir(parents=True, exist_ok=True)
        imu_data.mkdir(parents=True, exist_ok=True)
        camera_csv = cam_data.parent / "data.csv"
        imu_csv = imu_data / "data.csv"

        frame_imu: dict[int, PreparedImuSample] = {}
        for bundle in sorted_bundles:
            for sample in bundle.imu_samples:
                frame_imu[sample.sample_id] = sample

        with camera_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["#timestamp [ns]", "filename"])
            for output_index, bundle in enumerate(sorted_bundles):
                filename = f"{output_index:06d}.png"
                image_path = cam_data / filename
                encoded_ok, encoded = cv2.imencode(".png", bundle.image_bgr)
                if not encoded_ok:
                    raise BasaltExportError(f"failed to write image {image_path}")
                image_path.write_bytes(encoded.tobytes())
                writer.writerow([bundle.session_time_ns, f"data/{filename}"])

        imu_rows, skipped = synchronize_imu_samples(frame_imu.values())
        with imu_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(
                [
                    "#timestamp [ns]",
                    "w_x [rad s^-1]",
                    "w_y [rad s^-1]",
                    "w_z [rad s^-1]",
                    "a_x [m s^-2]",
                    "a_y [m s^-2]",
                    "a_z [m s^-2]",
                ]
            )
            for row in imu_rows:
                writer.writerow([row.timestamp_ns, *row.gyro, *row.accel])

        calibration_path = root / "calibration.json"
        calibration_path.write_text(
            json.dumps(
                calibration_to_basalt_json(
                    calibration,
                    input_is_rectified=input_is_rectified,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (root / "export.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "session_id": sorted_bundles[0].session_id,
                    "frame_count": len(sorted_bundles),
                    "imu_count": len(imu_rows),
                    "interpolated_imu_count": sum(row.interpolated for row in imu_rows),
                    "skipped_imu_timestamps": skipped,
                    "calibration_profile_id": calibration.calibration_profile_id,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return BasaltDataset(
            root=root,
            camera_data_csv=camera_csv,
            imu_data_csv=imu_csv,
            frame_count=len(sorted_bundles),
            imu_count=len(imu_rows),
            interpolated_imu_count=sum(row.interpolated for row in imu_rows),
            skipped_imu_timestamps=skipped,
        )
