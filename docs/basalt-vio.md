# Basalt VIO

`src/slam_vio` is a thin offline adapter around Basalt. The Windows process
exports one prepared capture session to EuRoC, invokes one Basalt process in
WSL, and parses the resulting trajectory. Basalt is never started per frame and
is not part of the live inference route.

## Runtime boundary

Windows owns PyQt, recording, hand tracking, task state, and result playback.
The configured WSL2 distribution owns only the Basalt binaries and per-run
staging data:

```text
Windows                                  WSL ext4
capture session -> prepared EuRoC  ->    Basalt input staging
UI processing job                 ->    basalt_vio
run.json + trajectory.csv         <-    logs + trajectory.csv
```

The default installation uses:

```text
distribution: Nvidia_SDKM_Ubuntu_22.04_JetPack_7.2
source:       /home/nvidia/egoglass/tools/basalt-src
build:        /home/nvidia/egoglass/tools/basalt-build
staging:      /home/nvidia/.cache/egoglass/basalt
revision:     0f3b2b52c807f70ff4e2973ce253c73329eea7bc
```

There is no automatic fallback to the previous Windows-native Basalt build. A
missing WSL distribution or executable makes the VIO stage fail explicitly;
the surrounding video-processing job remains viewable as `partial` with its
camera-space hand results.

## Install

Run once from `EgoGlass_client`:

```powershell
.\scripts\setup-basalt-wsl.ps1
```

The setup script installs Linux build dependencies, clones the pinned Basalt
revision into WSL ext4, applies the repository's monocular EuRoC and
EuRoC-only dependency patches, then builds:

- `basalt_vio`
- `basalt_calibrate`
- `basalt_calibrate_imu`

The Linux dependency set includes the X11, Mesa GLES, and GLU development
headers required while compiling Pangolin, libepoxy, and GLEW. Basalt still
runs headless in the client workflow.

Ubuntu 22.04 provides CMake 3.22 while this Basalt revision requires 3.24. The
setup script therefore installs the official pinned CMake 3.31.10 archive under
`/home/nvidia/egoglass/tools` without replacing the distribution's system
CMake. The Basalt checkout stays shallow. vcpkg retains its history because its
version database references port tree objects outside the checked-out submodule
commit; setup verifies the exact registry baseline declared by Basalt's
`vcpkg-configuration.json` before configuration.

The vcpkg package cache is persistent. Setup retries dependency configuration
up to three times when a transient proxy or upstream download failure interrupts
the manifest install; completed packages are restored from cache on retry.
Build concurrency defaults to four jobs to stay within the WSL memory budget.
Set `EGOGLASS_BASALT_BUILD_JOBS` before setup only when the WSL VM has enough
memory for additional C++ template compiler processes.

The pinned Basalt revision contains unused diagnostic variables in template
instantiations. Setup preserves `-Wall` output but passes
`-Wno-error=unused-variable` through Basalt's captured `CMAKE_CXX_FLAGS`,
matching its existing treatment of other compiler-version-dependent warnings
instead of patching estimator logic.

It requires the configured WSL user to have passwordless `sudo`. Re-running the
script is supported when the source contains exactly the EgoGlass patch set;
unrelated source changes stop setup instead of being overwritten.

Verify the installed binary directly:

```powershell
wsl.exe -d Nvidia_SDKM_Ubuntu_22.04_JetPack_7.2 --exec `
  /home/nvidia/egoglass/tools/basalt-build/basalt_vio --help
```

## Input boundary

`SensorPreprocessingPipeline.iter_recorded_session()` supplies decoded,
optionally undistorted BGR frames and IMU samples mapped to `session_time_ns`.
The exporter writes:

```text
mav0/
  cam0/data.csv
  cam0/data/*.png
  imu0/data.csv
calibration.json
```

The glasses record accelerometer and gyroscope values as separate rows. Basalt
requires both on every IMU row, so the exporter takes the union of sensor
timestamps and linearly interpolates the other sensor only when bracketed by
measured samples. It never extrapolates. Skipped boundary timestamps are
recorded in `export.json`.

The complete EuRoC directory is copied from the Windows filesystem to a unique
`session_id/run_id` staging directory in WSL ext4 before Basalt starts. Command
arguments are passed as separate `wsl.exe --exec` arguments and are never
assembled into a shell command string.

On success, `trajectory.csv`, stdout, stderr, and the run log are copied back to
the Windows run directory and only that validated staging directory is removed.
On failure, staging is retained by default for diagnosis. `run.json` records
the backend, distribution, executable, pinned revision, exit code, staging path,
and staging state.

## Calibration

Basalt receives a generated cereal `calibration.json`. EgoGlass
`transform_camera_to_imu` is passed as Basalt `T_imu_cam`, mapping camera
coordinates into IMU coordinates. When `raw_imu_to_body_axes` is non-unit, the
prepared IMU rows and extrinsic both use the corresponding body frame.

Prepared frames are rectified by default, so Basalt receives a pinhole model
and the rectified camera matrix. This avoids applying distortion twice.

The repository calibration is deliberately unmeasured. The runner rejects it
unless `allow_unverified_calibration` is enabled or
`--allow-unverified-calibration` is supplied for an integration test. Do not
interpret a trajectory produced with sample calibration as physical motion.

## Run without UI

```powershell
conda run -n egoglass python src\run_vio.py `
  --session local-data\recordings\<session-id> `
  --output local-data\vio-runs\<run-id>
```

The output contains the exported dataset, frozen calibration,
`trajectory.csv`, `run.json`, and captured Basalt logs. The command imports no
Qt, gateway, or task queue code.

## UI processing route

The video workbench has one processing action. Its background job attempts
Basalt first, then runs offline hand tracking and temporal processing for the
same session or selected clip. Successful world-space hand optimization binds
to that exact `vio_run_id`; it never selects an unrelated newer trajectory.

VIO runs are stored under:

```text
<session>/derived/vio/basalt/<run-id>/
```

If Basalt fails, the processing job is marked `partial`, 2D overlays and
camera-space 3D hands remain available, and no world translation is invented.
The live collection page never starts Basalt.
