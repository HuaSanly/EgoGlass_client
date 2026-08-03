# Native UI

EgoGlass uses PyQt6 6.8.1 and PyQt6-Fluent-Widgets 1.11.2 for its native
Windows interface. The display path has no browser, WebView, JPEG preview, or
MJPEG transport.

## Runtime ownership

One `python -m ui` process owns the whole client:

```text
PyQt main thread
  -> FluentWindow navigation
  -> default VideoProcessingView
  -> secondary HomeView for live capture
  -> 16 ms frame timers and 100 ms state timers

asyncio runtime thread
  -> Uvicorn/FastAPI signaling
  -> aiortc WebRTC ingress
  -> recording, optional live inference, and coherent status collection

bounded workers
  -> RGB conversion and PTS pacing
  -> persistent offline video-processing queue
  -> optional online hand inference
  -> MP4 recording and playback decoding
  -> live and recorded IMU orientation fusion
```

The UI owns presentation and commands only. WebRTC, recording, preprocessing,
hand tracking, and processing history remain in their owning `src/` packages.

## Video processing workspace

`VideoProcessingView` is the first navigation page. Its `CommandBar` exposes
manual library refresh, processing preset selection, start, cancel, retry, and
annotated-video export. The recording library is scanned once at startup and
again only after Refresh.

The workspace has three always-visible columns:

- a Fluent `TreeWidget` containing completed sessions and clips;
- one central 4:3 `VideoCanvas` with playback controls;
- one right-side OpenGL `SpatialSyncCanvas` above a Fluent result inspector.

Video and space are not Pivot pages. A single immutable `PlaybackFrame` carries
`clip_id`, `frame_index`, `pts_ns`, `session_time_ns`, and RGB. The same
`ReplaySnapshot` supplies the recorded IMU pose. Structured results are queried
with the same clip, frame, and session-time identity, so seeking, stepping, and
cross-clip playback update both views together.

The result inspector selects one primary run and an optional A/B run. Both
overlays reuse the one decoded RGB frame. The canvas draws a split comparison;
it never starts a second decoder. The task table is collapsible and shows
state, progress, elapsed time, and the failure detail retained by SQLite.

`SpatialSyncCanvas` uses pyqtgraph's `GLViewWidget` and PyOpenGL. It renders the
glasses mesh, camera origin, camera XYZ axes, frustum, recorded IMU orientation,
and both hands in rectified camera coordinates. It is a visualization of the
camera frame, not a VIO world pose.

## Live capture

`HomeView` is the second navigation page. It contains stream, recording,
session, and optional live-inference controls. Live inference is off by default
and is disabled while an offline GPU job is active. Model memory changes owner
only after the previous inference worker drains. Preview and recording keep
running during offline processing.

The live video reads immutable NumPy RGB buffers directly from the bounded
frame buffer. Hand results arrive through a single-item queue and are accepted
only for the same connection and a bounded source-frame age. The frame-chain
strip reports receive, conversion, presentation, paint, queue, drop, skip, and
overlay-age measurements.

The live IMU row reports gyroscope pose-update cadence rather than the combined
accelerometer and gyroscope message count. The relative Madgwick preview can be
re-zeroed without clearing filter state. Absolute yaw can drift because Glass3
does not provide a magnetometer.

## Four-by-three video

Every video surface computes the largest 4:3 rectangle available. A 640x480
source fills it. Other aspect ratios are letterboxed without crop or stretch.
`VideoCanvas` creates `QImage.Format_RGB888` over a contiguous immutable NumPy
array and retains the frame object until replacement.

## Run and verify

```powershell
.\scripts\start-client.ps1
conda run -n egoglass python -m pytest
conda run -n egoglass python -m pytest -q evals
conda run -n egoglass ruff check src ui tests evals scripts
conda run -n egoglass python scripts\benchmark_native_texture.py
```

The optional 30-second presentation soak is:

```powershell
$env:EGOGLASS_RUN_UI_SOAK = "1"
conda run -n egoglass python -m pytest -q -s `
  evals\test_ui_native_runtime.py::test_four_by_three_canvas_sustains_thirty_fps_for_thirty_seconds
```

Closing the native window finalizes active capture, stops playback and workers,
and releases signaling and discovery ports.
