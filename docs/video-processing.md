# Video Processing

Video processing is the default client workflow. Live inference is optional;
completed recordings are the authoritative input for heavier algorithms.

## Queue and lifecycle

`VideoProcessingService` runs one worker inside the unified client process. Its
SQLite queue is `<recordings-root>/.processing/jobs.sqlite3`. Jobs use these
states:

```text
queued -> preparing -> running -> completed
                    -> canceling -> canceled
                    -> failed
```

An active job found after a process restart becomes `interrupted`. Failed,
interrupted, and canceled jobs are retried by creating a new job, so previous
history and partial run evidence are not overwritten. The queue records start,
finish, progress, elapsed time, and failure detail. Offline work owns the GPU;
the service waits for any in-flight online inference, releases its model memory,
and leaves preview and recording active. The offline model is released before
optional online inference may resume.

Automatic enqueue after session finalization is persisted but off by default.
The operator can enable it from the processing inspector.

## Pipeline and outputs

The first preset pipeline is:

```text
capture validation
  -> recorded clock mapping
  -> sensor preprocessing
  -> HumanEgo-compatible hand tracking
  -> frame-addressable result index
```

Each attempt writes an immutable directory:

```text
<session>/derived/video-processing/<run-id>/
  run.json
  results.sqlite
  run.log
```

`run.json` records the job, preset, timestamps, counters, terminal state, and
error. `results.sqlite` indexes JSON results by `clip_id + frame_index +
session_time_ns`. `run.log` is flushed during lifecycle transitions.

The original MP4, telemetry database, session manifest, and quality report are
read-only inputs. Annotated video is generated only after an explicit Export
command and is derived from the original media plus `results.sqlite`; export
does not rerun the model.

## Legacy result cleanup

The migration tool targets only exact
`<session>/perception/hand-tracking/` directories. It enumerates and hashes
every target file, records SHA256 for all `<session>/media/` files before and
after deletion, and writes an atomic JSON audit under `.processing`.

```powershell
conda run -n egoglass python scripts\cleanup_legacy_hand_tracking.py
conda run -n egoglass python scripts\cleanup_legacy_hand_tracking.py --apply
```

The first command is a dry run. The apply command fails if a target escapes the
exact legacy path or if any raw-media hash changes.

## Playback and comparison

`ReplayPlayer` opens a complete session or one selected clip. It verifies each
decoded frame against the recorded MP4 PTS index and publishes a typed
`PlaybackFrame`. Multiple clips are ordered on their mapped session timeline.
Recorded IMU is fused once while opening and queried at the same
`session_time_ns` as the frame.

The UI queries one primary run and one optional comparison run for each frame.
Both overlays and the OpenGL hand pose share one decoded RGB frame. Seeking and
single-frame stepping therefore cannot advance video, IMU, and results
independently.

## Verification

```powershell
conda run -n egoglass python -m pytest -q `
  tests\test_video_processing_store.py `
  tests\test_video_processing_service.py `
  tests\test_video_processing_runner.py `
  tests\test_video_processing_export.py `
  tests\test_ui_replay_player.py
conda run -n egoglass python -m pytest -q `
  evals\test_video_processing_contract.py `
  evals\test_video_processing_pipeline.py `
  evals\test_video_processing_workspace.py
```
