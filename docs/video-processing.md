# Video Processing

Video processing is the default client workflow. Live inference is optional;
completed recordings are the authoritative input for heavier algorithms.

## Queue and lifecycle

`VideoProcessingService` runs one worker inside the unified client process. Its
SQLite queue is `<recordings-root>/.processing/jobs.sqlite3`. Jobs use these
states:

```text
queued -> preparing -> running -> completed
                    -> partial
                    -> canceling -> canceled
                    -> failed
```

An active job found after a process restart becomes `interrupted`. Partial,
failed, interrupted, and canceled jobs are retried by creating a new job, so previous
history and partial run evidence are not overwritten. The queue records start,
finish, progress, elapsed time, and failure detail. Offline work owns the GPU;
the service waits for any in-flight online inference, releases its model memory,
and leaves preview and recording active. The offline model is released before
optional online inference may resume.

Automatic enqueue after session finalization is persisted but off by default.
The operator can enable it from system settings. The quality preset and output
type are stored in `config/video-processing.yaml`, validated before save, and
applied only to newly submitted jobs. Every new job has inference stride 1. The
queue database keeps historical preset and stride values for older jobs but is
no longer the source of truth for new work.

## Pipeline and outputs

The first preset pipeline is:

```text
capture validation
  -> recorded clock mapping
  -> sensor preprocessing
  -> Basalt VIO attempt
  -> per-frame HumanEgo-compatible hand tracking
  -> temporal cleanup and world kinematic optimization
  -> frame-addressable result index
```

Each attempt writes an immutable directory:

```text
<session>/derived/video-processing/<run-id>/
  run.json
  results.sqlite
  run.log
```

`run.json` records the job, preset, timestamps, counters, terminal state,
error, submitted configuration revision, and SHA256 of all six configuration
files. The queue also stores the validated preprocessing, calibration, and hand
tracking snapshot under `offline_hand_tracking` used at submission, so a queued job cannot silently consume
later YAML edits. Schema v2 stores raw inference in `raw_frame_results` and the
final temporal result in `frame_results`, indexed by `clip_id + frame_index +
session_time_ns`; v1 stores remain readable. `run.log` is flushed during lifecycle
transitions.

The manifest binds `vio_run_id` to the exact Basalt run used by temporal
processing. A failed or incomplete VIO stage produces a viewable `partial` run:
camera-space overlays remain available, while world view is explicitly
unavailable for unmatched frames. Temporal metrics record confidence filtering,
interpolation, segment suppression, both grasp-smoothing passes, world mapping,
and kinematic optimization durations. Short internal VIO gaps may be filled only
inside the smoothing calculation; unmatched frames never receive invented world
coordinates.

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

## Playback and results

`ReplayPlayer` opens a complete session or one selected clip. It verifies each
decoded frame against the recorded MP4 PTS index and publishes a typed
`PlaybackFrame`. Multiple clips are ordered on their mapped session timeline.
Recorded IMU is fused once while opening and queried at the same
`session_time_ns` as the frame.

Session playback always indexes every complete clip and exposes their relative
time spans. An optional initial clip selects the first displayed frame without
discarding the rest of the session timeline. `unload()` releases the active
decoder and frame buffers when the operator returns to the video hall while
keeping the replay worker available for the next selection.

The UI queries the selected run once for each frame. The video overlay and the
OpenGL hand pose share one decoded RGB frame. Seeking and single-frame stepping
therefore cannot advance video, IMU, and results independently.

Run discovery returns typed `ProcessingRunInfo` records. A completed manifest
is viewable only when its `results.sqlite` exists and passes the result-store
schema check. Missing or corrupt stores remain unavailable instead of appearing
as selectable playback results.

## Video hall and navigation

Video Processing opens in a flat Fluent card hall rather than immediately
allocating a decoder. Cards fill each row from left to right and wrap at the
viewport edge; the session name is carried by the card instead of a containing
session section. Each card displays an in-memory first-frame thumbnail and the
number of viewable runs that cover that clip. Session-wide runs count for every
clip; clip-scoped runs count only for their matching clip. Incomplete sessions
stay visible but disabled.

Opening a clip loads the complete session and positions playback at that clip.
The workbench selects the newest valid run by completion time, with Raw Video
always available. Result changes only replace the result-store query; video is
not decoded again. The 2D overlay and 3D hand pose use the same
`clip_id + frame_index + session_time_ns` identity. Returning to the hall or
switching navigation unloads the active decoder.

The Pipeline page owns queued and historical jobs, including cancel and retry.
System Settings persists the quality preset, automatic-enqueue policy, and
result type through `ConfigurationService`. Real-time and offline hand tracking
have separate settings modules and files. The
workbench's Process Current Video command always submits the current clip with
that saved configuration and records its provenance in the queue entry.

## Verification

```powershell
conda run -n egoglass python -m pytest -q `
  tests\test_video_processing_store.py `
  tests\test_video_processing_service.py `
  tests\test_video_processing_runner.py `
  tests\test_video_processing_export.py `
  tests\test_configuration_service.py `
  tests\test_configuration_runtime.py `
  tests\test_ui_processing_settings.py `
  tests\test_ui_replay_player.py `
  tests\test_ui_video_hall.py `
  tests\test_ui_native_views.py
conda run -n egoglass python -m pytest -q `
  evals\test_video_processing_contract.py `
  evals\test_video_processing_pipeline.py `
  evals\test_video_processing_workspace.py
```
