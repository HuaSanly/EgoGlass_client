# Spatial Perception

`spatial-perception` turns prepared sensor sequences into session-local spatial
evidence. It owns visual-inertial odometry, hand tracking, coordinate fusion,
and artifact export. Sensor-session reading, calibration binding, and clock
association belong to `sensor_preprocessing`.

The Python package owns the project boundary; Basalt remains an external native
process and is not copied into the package.

## Directory purpose

| Path | Purpose |
| --- | --- |
| `src/hand_tracking/` and `src/slam_vio/` | Hand tracking and the Basalt offline VIO adapter. |
| `tests/` | Shared fast tests, identified by `test_spatial_perception_` filenames. |
| `evals/` | Shared periodic evaluations, identified by `test_spatial_perception_` filenames. |
| `config/` | Shared versioned client configuration. |
| `pyproject.toml` | Workspace package, tool, and dependency declaration. |

## Module purpose

| Module | Future responsibility |
| --- | --- |
| `src/slam_vio/runner.py` | Export a prepared session to EuRoC, run Basalt, and parse poses. |
| `src/slam_vio/calibration.py` | Convert the sensor calibration to Basalt's cereal JSON. |
| `src/hand_tracking/` | Produce left/right hand observations and confidence. |
| `ui/processing/` | Own the UI-backed task workflow and artifact indexing. |

## Boundary

The current VIO input is a versioned prepared-sensor sequence. The Basalt
adapter writes an isolated run directory containing the EuRoC export, logs,
`run.json`, and a parsed `trajectory.csv`. The trajectory is valid only when a
measured camera-IMU calibration is supplied; sample calibration is accepted
only for I/O smoke tests.

## Verification

```powershell
conda run -n egoglass python -m pytest
conda run -n egoglass python -m pytest -q evals
conda run -n egoglass ruff check src tests evals
```
