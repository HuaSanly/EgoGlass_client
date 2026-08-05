# Spatial Perception

`spatial-perception` will turn prepared sensor sequences into session-local
spatial evidence. It owns visual-inertial odometry, hand tracking, coordinate
fusion, and artifact export. Sensor-session reading, calibration binding, and
clock association belong to `sensor_preprocessing`.

This package currently defines the spatial-perception boundary only. No
perception algorithm or third-party source has been copied into the package.

## Directory purpose

| Path | Purpose |
| --- | --- |
| `src/hand_tracking/` and `src/slam_vio/` | Hand tracking and planned SLAM/VIO services. |
| `tests/` | Shared fast tests, identified by `test_spatial_perception_` filenames. |
| `evals/` | Shared periodic evaluations, identified by `test_spatial_perception_` filenames. |
| `config/` | Shared versioned client configuration. |
| `pyproject.toml` | Workspace package, tool, and dependency declaration. |

## Module purpose

| Module | Future responsibility |
| --- | --- |
| `pipeline.py` | Orchestrate the complete spatial-perception run. |
| `models.py` | Define internal typed records and coordinate transforms. |
| `vio.py` | Isolate the selected SLAM/VIO backend behind a project-owned interface. |
| `hand_tracking.py` | Produce left/right hand observations and confidence. |
| `coordinate_fusion.py` | Combine camera motion and camera-relative hand evidence. |
| `export.py` | Write an immutable, versioned spatial-perception artifact. |

## Planned boundary

Input will be a versioned prepared-sensor sequence. Output will be a future versioned
`spatial-perception-v0` artifact under the session's `derived/` directory. The
contract must be defined before algorithm implementation starts.

## Verification

```powershell
conda run -n egoglass python -m pytest
conda run -n egoglass python -m pytest -q evals
conda run -n egoglass ruff check src tests evals
```
