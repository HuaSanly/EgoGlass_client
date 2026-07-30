# Native UI

EgoGlass uses Dear PyGui 2.3.1 as a native Windows operator UI. There is no
browser, local UI server, WebView, JPEG preview, or MJPEG transport in the
application path.

## Runtime ownership

One `python -m ui` process owns all client modules:

```text
Dear PyGui main thread
  -> reads immutable RuntimeSnapshot values
  -> swaps the newest RGB frame into a double-buffered raw texture
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

`LiveFrameBuffer` is the only live consumer that reads the decoded
`av.VideoFrame`. It keeps at most one pending frame, converts it once to an
immutable contiguous RGB array, and publishes that same array to the UI and
perception. This avoids concurrent PyAV color conversion on one decoded frame.
The UI consumes RGB without network serialization. Slow inference can drop its
own pending input without stopping the video surface.

The live aiortc relay is unbuffered: if UI bookkeeping briefly falls behind,
the next callback receives the newest decoded frame instead of replaying an
old queue. Recording uses its own buffered relay subscription. PyAV frames
marked corrupt are counted and discarded before either RGB display or
perception. Receiver packet loss, jitter, and corrupt-frame drops are exposed
in the diagnostics view.

## Views

The Live view owns the application's only `VideoSurface` and `ReplayPlayer`.
It switches the large raw RGB texture between live input and stored replay.
Hand boxes and keypoints are drawn on a Dear PyGui layer above that texture.
The overlay identity includes session, stream, and inference frame index, so a
new result on the same WebRTC connection replaces the previous result.

The Library view reads `RecordingRuntime.library()` through the unified
snapshot. The recording directory is scanned once at startup and only again
when the operator presses Refresh; there is no polling scan. Scans run off the
media event loop, so hashing stored clips cannot pause WebRTC, IMU, or UI status
updates. The view shows session quality, clip metadata, rename, replay
generation, and confirmed session/clip deletion. Opening a clip routes it to
the Live view's existing replay surface.

The Annotation view uses `src/annotation/AnnotationController` directly. It
supports manual Episode and phase intervals, whole-clip and fixed-window
proposals, semantic labels, undo/redo, draft revision checks, and immutable
publication. Source MP4 and telemetry files remain read-only. The selected
clip is reviewed on the Live view's same replay surface.

The Diagnostics view shows WebRTC state, input/display rates, metadata pairing,
IMU rates and gaps, orientation queue overflow, recording state, inference
latency/drop counts, and the latest runtime command events.

Only the selected Dear PyGui tab is updated on each render iteration. Hidden
library, annotation, and diagnostics widgets retain their last snapshot until
selected, avoiding periodic hidden-widget work on the live video path. The
Live header reports actual raw-texture swaps per second; RGB conversion FPS
remains a separate diagnostic.

## Frame and replay behavior

Live display always uses gateway-decoded RGB frames. It never reconnects a
viewer or waits for inference output. `VideoSurface` preserves a stable
960x540 draw area and letterboxes source coordinates using the result's source
image dimensions.

Decoded RGB frames enter a four-frame maximum presentation queue before the
native texture. The queue learns cadence from video PTS, starts with about three
frames of prebuffer, and presents at that cadence instead of copying network
arrival bursts directly to the screen. This adds about 100 ms of display latency
at 30 FPS. If the queue grows beyond its target it drops the oldest frame and
never accumulates latency. Perception still receives the canonical RGB frame
immediately. Diagnostics report queue depth, smoothing drops, starvation
events, the learned presentation interval, UI poll FPS, and actual presentation
FPS. These separate a slow Dear PyGui render loop from receive-side jitter.
After an underflow, presentation holds the last frame until the target prebuffer
is rebuilt or the startup wait expires, preventing repeated one-frame restarts.

WebRTC may still change encoded resolution after a source or session change.
`VideoSurface` owns two raw textures at the active dimensions. It writes only
the texture that is not currently displayed, then swaps the image binding.
This prevents the renderer from observing a partially updated RGB buffer. A
resolution transition creates two uniquely tagged replacements before deleting
the old pair, so it cannot reuse a pending Dear PyGui alias. The perception
preprocessor accepts proportional transport downscales and restores them to the
calibrated raster before undistortion and inference. A different aspect ratio
remains an error.

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

The current Windows double-buffer test at 1280x720 measured 246.63 effective
FPS, 4.055 ms mean frame work, 5.253 ms p95 frame work, 1.659 ms mean RGB
conversion, and 2.229 ms p95 RGB conversion. This remains well below the
33.3 ms budget of a 30 FPS source. The benchmark creates a real viewport, runs
a fixed frame count, prints JSON, and closes automatically.

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
