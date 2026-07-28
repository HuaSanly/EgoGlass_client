# Sensor Preprocessing

`sensor_preprocessing` converts immutable `capture-session-v1` records into
versioned sensor sequences for spatial perception. It preserves raw timestamps,
represents mapped session time with explicit status and uncertainty, and keeps
MP4 presentation time exact.

The package does not estimate camera motion, track hands, mutate capture files,
or claim that a camera callback timestamp is an exposure timestamp.

## Clock mapping

`clock_mapping.py` applies versioned, externally supplied clock mappings. It
does not fit a clock model from capture data and does not infer camera exposure
time. Every `TimeObservation` identifies four independent facts:

- `source_clock_id`: the clock and its unit;
- `source_instance_id`: the boot, connection, camera generation, or clip that
  owns the clock value;
- `source_timestamp`: the original integer; only MP4 presentation ticks may be
  negative;
- `timestamp_semantic`: the event represented by the value, such as a sensor
  event, camera callback, SDK timestamp, media presentation, or client receipt.

Clock/semantic combinations are validated. In particular, an Android camera
callback cannot be labeled as camera exposure, and a client receipt time cannot
be labeled as an IMU sensor event.

`ClockMappingSegment` is bound to one capture session and maps one source clock
instance and inclusive source range directly to `session_time_ns`. The affine
mapping uses integer numerator and denominator fields rather than floats:

```text
session_time_ns = target_anchor_ns
                + (source_timestamp - source_anchor)
                  * scale_numerator_ns
                  / scale_denominator_source_units
```

Fractional nanoseconds use deterministic half-away-from-zero rounding. One
nanosecond is added to the uncertainty bound whenever rounding is required.
Caller-supplied attachment uncertainty is added to the mapping uncertainty;
`rtp_match_error_to_uncertainty_ns()` converts the frame match error from 90 kHz
ticks with conservative upward rounding.

The mapping ID is a SHA256 content ID derived from all normalized mapping,
session, uncertainty, and provenance fields. `VERIFIED` mappings require an
explicit evidence ID; a fitted mapping remains `ESTIMATED`. Segment uncertainty
must be a conservative bound including source quantization, residual maximum,
and any extrapolation bound. A percentile alone is not an error bound.

`SegmentedClockMapper` selects by session, clock ID, source instance, and valid
range. Overlapping ranges, reversed indices, and target-time regression are
rejected. If no segment applies, it returns `TimeStatus.UNAVAILABLE` while
preserving the complete raw time observation. Camera restarts therefore cannot
silently reuse the previous camera generation's mapping. Association status is
combined with mapping status using the weaker value, so estimated frame
association cannot inherit a verified clock status.

Source-instance helpers encode the required reset boundary deterministically:

- glasses elapsed/IMU: session + connection;
- Rokid SDK: session + connection + camera generation;
- MP4 presentation: session + clip + rational time base;
- client perf counter: session + connection proxy.

The helpers serialize these components as canonical JSON. Persist and replay the
returned string unchanged; hand-written JSON, alternate key order, and
delimiter-joined aliases are rejected when a mapping segment is loaded.

Raw Rokid SDK and callback timestamps cannot carry `CAMERA_EXPOSURE` semantics.
That semantic belongs to a later calibrated frame-timing result with explicit
provenance and uncertainty.

This module is an in-memory application contract. It neither reads nor writes
the ingest database's legacy floating-point `clock_mapping_segments` rows.
Future persistence must use a separate versioned derived artifact containing
the integer anchors and scale numerator/denominator; completed raw telemetry is
never updated by preprocessing.

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

```python
from perception.sensor_preprocessing import (
    ClockId,
    ClockMappingSegment,
    SegmentedClockMapper,
    TimeObservation,
    TimestampSemantic,
    TimeStatus,
    glasses_elapsed_source_instance_id,
)

segment = ClockMappingSegment(
    session_id="session-1",
    source_clock_id=ClockId.GLASSES_ELAPSED_REALTIME_NS,
    source_instance_id=glasses_elapsed_source_instance_id(
        "session-1",
        "connection-1",
    ),
    segment_index=0,
    source_from=1_000_000_000,
    source_to=2_000_000_000,
    source_anchor=1_000_000_000,
    target_anchor_ns=0,
    scale_numerator_ns=1,
    scale_denominator_source_units=1,
    uncertainty_ns=200,
    status=TimeStatus.ESTIMATED,
    fit_method="android_elapsed_realtime_identity",
    provenance_id="glass3-clock-contract-v1",
    uncertainty_basis="device_test_residual_max",
)
mapper = SegmentedClockMapper("session-1", (segment,))
aligned = mapper.map(
    TimeObservation(
        session_id="session-1",
        source_clock_id=ClockId.GLASSES_ELAPSED_REALTIME_NS,
        source_instance_id=segment.source_instance_id,
        source_timestamp=1_100_000_000,
        timestamp_semantic=TimestampSemantic.SENSOR_EVENT,
    )
)
```

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_sensor_preprocessing_models.py tests\test_sensor_preprocessing_capture_reader.py tests\test_sensor_preprocessing_clock_mapping.py
.\.venv\Scripts\python.exe -m pytest -q evals\test_sensor_preprocessing_boundary.py evals\test_sensor_preprocessing_clock_mapping.py
.\.venv\Scripts\ruff.exe check src tests evals
```
