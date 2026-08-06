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
which maps a point from camera coordinates to IMU coordinates. When a non-unit
`raw_imu_to_body_axes` calibration is supplied, prepared IMU rows and the
exported extrinsic use the corresponding camera-to-body transform. Prepared
frames are rectified by default, so the calibration uses a pinhole model and
the rectified camera matrix to avoid applying distortion twice.

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

## Windows native build

Basalt is kept outside this repository because it is a native third-party
dependency. The tested Windows build uses Basalt commit `0f3b2b5` in
`<workspace>/.tools/basalt-src`, Visual Studio 2022 x64 tools, and the bundled
vcpkg tree. Build the `relwithdebinfo` preset with `basalt_vio` as the target,
then add both the build directory and its vcpkg runtime directory to `PATH`:

```powershell
$basaltBuild = "F:\data\Project\EgoGlass\.tools\basalt-src\build\relwithdebinfo"
$env:PATH = "$basaltBuild;$basaltBuild\vcpkg_installed\x64-windows\bin;$env:PATH"
```

The Basalt EuRoC reader in this Windows build accepts a monocular dataset when
`mav0/cam1/data.csv` is absent. This is required for the current glasses
capture, which has one camera. The exporter writes the standard EuRoC CSV form
with bare image filenames; Basalt adds `cam0/data/` while reading them.

The smoke run used session `224a88e7342e4702acea0c7d2a2cb6db` at 640x480 and
produced 360 poses from 361 frames. The sample calibration is not measured, so
that run required `--allow-unverified-calibration`; its trajectory is only an
I/O integration check, not a calibrated motion result.

## Native UI route

The video-processing workbench has one `处理当前视频` action. That persistent
offline job runs hand tracking first and then Basalt VIO for the same selected
clip. There is no separate UI VIO task or live VIO route. The VIO stage starts
after the hand-tracking model is released, and the processing job is marked
complete only after both stages finish.

The VIO stage stores its run under the selected session:

```text
<session>/derived/vio/basalt/<run-id>/
```

When the processing job completes, reopening the session workbench discovers the
newest valid `trajectory.csv`. The spatial OpenGL view draws the complete
trajectory and selects the pose nearest the shared playback `session_time_ns`,
so video, hand results, and VIO remain on one timeline. The live collection view
never starts this service and online inference is unchanged.
