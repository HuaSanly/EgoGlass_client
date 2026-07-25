# Spatial Perception

`spatial-perception` will turn one immutable EgoGlass capture session into
session-local spatial evidence. It owns calibration, visual-inertial odometry,
hand tracking, post-perception time association, quality evaluation, and
artifact export. It does not own object-interaction preprocessing or training
dataset generation.

This commit creates the service boundary only. No perception algorithm or
third-party source has been copied into the package.

## Directory purpose

| Path | Purpose |
| --- | --- |
| `src/egoglass_spatial_perception/` | Python package for the service. |
| `tests/` | Fast deterministic gate tests for package structure and pure behavior. |
| `evals/` | Periodic quality and service-boundary evaluations. |
| `config/` | Versioned runtime and quality-threshold configuration. |
| `pyproject.toml` | Package, tool, and dependency declaration. |

## Module purpose

| Module | Future responsibility |
| --- | --- |
| `pipeline.py` | Orchestrate the complete spatial-perception run. |
| `session_input.py` | Read capture manifests, MP4 PTS, frame metadata, and IMU samples without mutating them. |
| `models.py` | Define internal typed records and coordinate transforms. |
| `calibration.py` | Load and validate camera intrinsics, distortion, IMU calibration, and camera-to-IMU extrinsics. |
| `vio.py` | Isolate the selected SLAM/VIO backend behind a project-owned interface. |
| `hand_tracking.py` | Produce left/right hand observations and confidence. |
| `time_association.py` | Associate source clocks after perception while preserving every original timestamp. |
| `quality.py` | Report tracking coverage, drift evidence, gaps, and uncertainty. |
| `export.py` | Write an immutable, versioned spatial-perception artifact. |

## Planned boundary

Input will be `capture-session-v1`. Output will be a future versioned
`spatial-perception-v0` artifact under the session's `derived/` directory. The
contract must be defined before algorithm implementation starts.

## Verification

```powershell
uv sync --group dev
uv run pytest
uv run pytest -q evals
uv run ruff check src tests evals
```
