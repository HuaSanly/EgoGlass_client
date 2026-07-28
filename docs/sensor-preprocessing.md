# Sensor Preprocessing

`sensor_preprocessing` converts immutable `capture-session-v1` records into
versioned sensor sequences for spatial perception. It preserves raw timestamps,
represents mapped session time with explicit status and uncertainty, and keeps
MP4 presentation time exact.

The package does not estimate camera motion, track hands, mutate capture files,
or claim that a camera callback timestamp is an exposure timestamp.

## Implemented boundary

`CaptureSessionReader.open()` accepts only finalized `capture-session-v1`
directories. Before exposing data, it validates:

- `schema_version == 1.0`, `contract_id == capture-session-v1`, and a complete
  lifecycle;
- the directory name against `session_id` and every referenced path against
  directory traversal;
- complete clip metadata, media existence, and media SHA256 by default;
- telemetry schema/session identity and each clip's frame count.

The reader opens telemetry SQLite with `mode=ro&immutable=1`. Completed sessions
are checkpointed before finalization, so this prevents database writes and also
prevents SQLite from creating `-wal` or `-shm` files in the raw session. A
non-empty WAL or rollback journal is rejected before immutable mode can ignore
it and expose stale database pages. Passing `verify_media_hashes=False` is an
explicit fast-replay option for already trusted local data; it does not weaken
manifest, path, database, or frame-index checks.

`iter_frames(clip_id)` returns immutable `RawFrameRef` values in `frame_index`
order. Each value preserves MP4 PTS/time base, source decoder PTS/time base,
camera metadata match ID/status/error, all available device/client clocks, and
the stored alignment fields. Joined camera metadata must be accepted and agree
with its frame match and frame index. `iter_imu_samples()` returns immutable
`RawImuSample` values in database `sample_id` order, preserving out-of-order
evidence, raw axes, units, sensor identity, all clocks, and stored alignment
fields. Stored alignment is either `pending` with no derived fields or `mapped`
with a complete time, uncertainty, and mapping-segment reference.

The current reader deliberately does not decode video, sort/filter/resample IMU,
fit clocks, infer exposure time, bind calibration, or split sequences. Those are
separate preprocessing stages. It also has no runtime import dependency on
`ingest_gateway`, `operator_console`, or `spatial_perception`; only tests use the
real gateway writer to produce contract-compatible synthetic fixtures.

## Public API

```python
from perception.sensor_preprocessing import CaptureSessionReader

reader = CaptureSessionReader.open(session_directory)
frames = reader.iter_frames(clip_id)
imu_samples = reader.iter_imu_samples()
```

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_sensor_preprocessing_models.py tests\test_sensor_preprocessing_capture_reader.py
.\.venv\Scripts\python.exe -m pytest -q evals\test_sensor_preprocessing_boundary.py
.\.venv\Scripts\ruff.exe check src tests evals
```
