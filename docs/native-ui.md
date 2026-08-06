# Native UI

EgoGlass uses PyQt6 6.8.1 and PyQt6-Fluent-Widgets 1.11.2 for its native
Windows interface. The display path has no browser, WebView, JPEG preview, or
MJPEG transport.

## Runtime ownership

One `python -m ui` process owns the whole client:

```text
PyQt main thread
  -> FluentWindow navigation
  -> VideoProcessingView with hall and workbench states
  -> ProcessingPipelineView and ProcessingSettingsView
  -> HomeView for live capture
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

## Video hall and processing workspace

The sidebar order is Video Processing, Pipeline, Live Capture, and System
Settings. Video Processing opens in the video hall. A single Fluent
`FlowLayout` places cards left-to-right and wraps only when the viewport is
full; there are no per-session containers or dark framed content panels.
Incomplete sessions remain as disabled cards. Each clip card shows its session,
a real asynchronously decoded first frame, capture time, duration, format,
frame count, file size, processing state, and viewable-result count. Two
bounded thumbnail workers only inspect exact `media/<clip-id>.mp4` or legacy
`<clip-id>.mp4` paths. The frameless `SmoothScrollArea` uses the same opaque
background as the page, which prevents hover repaint artifacts while remaining
visually unframed. It is used only for overflow. The recording library is
scanned once at startup and again only after the operator presses Refresh.

Selecting a card opens the workbench, indexes the complete session, and seeks
to that clip. The workbench keeps one central 4:3 `VideoCanvas` and one
right-side OpenGL `SpatialSyncCanvas` visible together. Returning to the hall
or navigating away calls `ReplayPlayer.unload()` so the active decoder and RGB
frame are released.

Video and space are not Pivot pages. A single immutable `PlaybackFrame` carries
`clip_id`, `frame_index`, `pts_ns`, `session_time_ns`, and RGB. The same
`ReplaySnapshot` supplies the recorded IMU pose. Structured results are queried
with the same clip, frame, and session-time identity, so seeking, stepping, and
cross-clip playback update both views together.

The result selector lists Raw Video plus completed, schema-valid session and
clip runs that cover the current clip. The newest valid result is selected by
default. An optional A/B selector only lists another run covering that clip.
Switching either selector changes structured-result queries without reopening
the decoder. The canvas draws a split comparison over one RGB frame. Export is
explicit and reports the output path in an `InfoBar`.

The real session clip spans form a clickable timeline below the video. Slice
candidate and marker regions are visibly read-only placeholders: they expose no
editor, annotation store, or file-write path until a slicing algorithm exists.

Pipeline owns task progress, cancel, retry, elapsed time, and failure history.
System Settings contains an internal Fluent navigation rail for Client and
Gateway, Sensor Preprocessing, Hand Tracking, and Offline Video Processing.
Each module uses typed controls, shows the field's application level, and keeps
edits in memory until Save or Save and Apply. Discard restores the last saved
revision; Restore Defaults affects only the selected module. Calibration JSON
is selected as a file and validated rather than edited as a matrix form.

Settings never parse YAML directly. `ConfigurationService` validates and writes
the five configuration files, while `UnifiedRuntimeHost` applies live inference
and future-job defaults. Restart-bound gateway values remain visibly pending.
Device capture resolution and bitrate are read-only because the glasses own
those settings.

`SpatialSyncCanvas` uses pyqtgraph's `GLViewWidget` and PyOpenGL. It renders a
head coordinate frame and both hands, with a Fluent camera/world selector. The
camera view shows only hands and camera XYZ axes. The world view uses the first
Basalt pose as the display origin, applies the frozen camera-to-IMU extrinsic,
and falls back to relative IMU orientation without translation when VIO is not
available. No glasses mesh, frustum, or trajectory is drawn.

## Live capture

`HomeView` is the third navigation page. It contains stream, recording,
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

The local-recording replay soak loops a real session if it is shorter than 30
seconds and verifies at least 300 presented frames:

```powershell
$env:EGOGLASS_RUN_REPLAY_SOAK = "1"
conda run -n egoglass python -m pytest -q -s `
  evals\test_video_processing_workspace.py::test_existing_recording_plays_continuously_for_thirty_seconds
```

The optional 30-second presentation soak is:

```powershell
$env:EGOGLASS_RUN_UI_SOAK = "1"
conda run -n egoglass python -m pytest -q -s `
  evals\test_ui_native_runtime.py::test_four_by_three_canvas_sustains_thirty_fps_for_thirty_seconds
```

Closing the native window finalizes active capture, stops playback and workers,
and releases signaling and discovery ports.
