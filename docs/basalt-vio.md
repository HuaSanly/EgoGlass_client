# Basalt VIO

`src/slam_vio` is a thin adapter around the native Basalt executable. It does
not copy Basalt's estimator into Python and it does not start a process for
each frame. The first implementation is offline only.

## Input boundary

`SensorPreprocessingPipeline.iter_recorded_session()` supplies decoded,
optionally undistorted BGR frames and IMU samples mapped to `session_time_ns`.
The adapter writes the frames and timestamps into a temporary EuRoC dataset:

```text
mav0/
  cam0/data.csv
  cam0/data/*.png
  imu0/data.csv
```

The glasses record accelerometer and gyroscope samples as separate rows. Basalt
requires both values on every IMU row, so the exporter takes the union of both
sensor timestamps and linearly interpolates the other sensor only when the
timestamp is bracketed by measured samples. It never extrapolates; skipped
boundary timestamps are recorded in `export.json`.

## Calibration

Basalt receives a generated `calibration.json` in its cereal format. The
EgoGlass `transform_camera_to_imu` matrix is passed as Basalt's `T_imu_cam`,
which maps a point from camera coordinates to IMU coordinates. Prepared frames
are rectified by default, so the calibration uses a pinhole model and the
rectified camera matrix to avoid applying distortion twice.

The repository sample calibration is deliberately unmeasured. The runner
refuses it unless `allow_unverified_calibration` is enabled in the Basalt YAML
or `--allow-unverified-calibration` is supplied for an integration test. A
measured camera-IMU calibration must be used for any trajectory interpretation.

## Running

Install `basalt_vio` separately and run:

```powershell
conda run -n egoglass python src\run_vio.py `
  --session local-data\recordings\<session-id> `
  --output local-data\vio-runs\<run-id>
```

The output contains the EuRoC dataset, generated calibration, `trajectory.csv`,
`run.json`, and captured Basalt stdout/stderr. `trajectory.csv` is parsed into
`schemas.VioTrajectory` and `schemas.VioPose` objects. No Qt, gateway, or task
queue is imported by this command.

The checked-in `config/basalt-euroc-config.json` follows Basalt's official
EuRoC offline defaults. `config/basalt-vio.yaml` controls the executable,
thread count, GUI flag, precision, frame limit, and calibration policy.
