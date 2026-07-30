# Native UI

EgoGlass uses Dear PyGui 2.3.1 as a native Windows operator UI. There is no
browser, local UI server, WebView, JPEG preview, or MJPEG transport in the
application path.

## Runtime ownership

One `python -m ui` process owns all client modules:

```text
Dear PyGui main thread
  -> reads immutable RuntimeSnapshot values
  -> uploads newest RGB frame to one raw texture
  -> sends commands directly to UnifiedRuntimeHost

asyncio runtime thread
  -> Uvicorn/FastAPI for Glass3 signaling and diagnostics
  -> aiortc WebRTC termination
  -> recording and perception orchestration
  -> 10 Hz coherent status collection

bounded workers
  -> decoded frame to contiguous RGB conversion
  -> online hand inference
  -> MP4 recording
  -> PTS-driven replay decoding
  -> IMU Madgwick orientation preview
```

The gateway sends the same decoded `av.VideoFrame` object to bounded display
and perception consumers. `LiveFrameBuffer` keeps at most one pending frame so
UI conversion cannot block WebRTC reception. The UI consumes contiguous RGB
without network serialization. Slow inference can drop its own pending input
without stopping the video surface.

## Views

The Live view owns the application's only `VideoSurface` and `ReplayPlayer`.
It switches the large raw RGB texture between live input and stored replay.
Hand boxes and keypoints are drawn on a Dear PyGui layer above that texture.

The Library view reads `RecordingRuntime.library()` through the unified
snapshot. It shows session quality, clip metadata, rename, replay generation,
and confirmed session/clip deletion. Opening a clip routes it to the Live
view's existing replay surface.

The Annotation view uses `src/annotation/AnnotationController` directly. It
supports manual Episode and phase intervals, whole-clip and fixed-window
proposals, semantic labels, undo/redo, draft revision checks, and immutable
publication. Source MP4 and telemetry files remain read-only. The selected
clip is reviewed on the Live view's same replay surface.

The Diagnostics view shows WebRTC state, input/display rates, metadata pairing,
IMU rates and gaps, orientation queue overflow, recording state, inference
latency/drop counts, and the latest runtime command events.

## Frame and replay behavior

Live display always uses gateway-decoded RGB frames. It never reconnects a
viewer or waits for inference output. `VideoSurface` preserves a stable
960x540 draw area and letterboxes source coordinates using the result's source
image dimensions.

Replay uses PyAV on one worker. Frame presentation time is `PTS * time_base`;
wall scheduling applies the selected 0.25x to 2.0x rate. Pause, step, seek, and
play-to-end use the same decoder. A bad or missing file enters an error state
without terminating the worker, so a later valid clip can still be opened.

## IMU preview

`ImuPreviewRuntime` uses the maintained Python `ahrs` Madgwick filter on a
bounded worker. It reports a visualization quaternion, Euler angles, rates,
sample age, and queue overflow. This orientation is only a UI preview. Future
VIO remains authoritative for spatial pose.

## Performance gate

Run the native texture benchmark:

```powershell
conda run -n egoglass python scripts\benchmark_native_texture.py
```

The current Windows test at 1280x720 measured 596.84 effective FPS, 1.675 ms
mean frame work, 2.027 ms p95 frame work, 1.076 ms mean upload, and 1.310 ms p95
upload. The benchmark creates a real viewport, runs a fixed frame count, prints
JSON, and closes automatically.

## Run and build

```powershell
.\scripts\start-client.ps1
.\scripts\build-desktop.ps1
```

The launcher checks TCP `8770` and UDP `8771`, then runs exactly one foreground
Python process. Closing the viewport executes orderly capture finalization and
worker shutdown. The build script packages `packaging/native-entry.py`, runs
the executable's `--smoke-test`, and writes ignored output under `dist/EgoGlass/`.

## Verification

```powershell
conda run -n egoglass python -m pytest
conda run -n egoglass python -m pytest -q evals
conda run -n egoglass ruff check src ui tests evals scripts
```
