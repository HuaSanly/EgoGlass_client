# Sensor Preprocessing

`sensor_preprocessing` converts immutable `capture-session-v1` records into
versioned sensor sequences for spatial perception. It preserves raw timestamps,
represents mapped session time with explicit status and uncertainty, and keeps
MP4 presentation time exact.

The package does not estimate camera motion, track hands, mutate capture files,
or claim that a camera callback timestamp is an exposure timestamp.

## Current boundary

The initial implementation defines immutable records for:

- `TimeStatus`: verified, estimated, or unavailable time evidence.
- `TimeEstimate`: a raw source timestamp and its optional mapped session time.
- `Mp4Timestamp`: an exact integer PTS plus rational time base.

Future modules will read finalized sessions, bind externally produced sensor-rig
calibration, resolve clocks, decode frames lazily, prepare IMU intervals, and
export deterministic derived manifests.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_sensor_preprocessing_models.py
.\.venv\Scripts\python.exe -m pytest -q evals\test_sensor_preprocessing_boundary.py
.\.venv\Scripts\ruff.exe check src tests evals
```
