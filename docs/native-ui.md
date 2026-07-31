# Native UI

EgoGlass uses PyQt6 6.8.1 and PyQt6-Fluent-Widgets 1.11.2 for its native
Windows operator interface. There is no browser, WebView, JPEG preview, or
MJPEG transport in the display path.

## Runtime ownership

One `python -m ui` process owns all client modules:

```text
PyQt main thread
  -> FluentWindow with one Home interface
  -> 16 ms frame timer and 100 ms status timer
  -> direct immutable RGB QImage painting
  -> direct commands to UnifiedRuntimeHost

asyncio runtime thread
  -> Uvicorn/FastAPI signaling
  -> aiortc WebRTC termination
  -> recording and perception orchestration
  -> coherent status collection

bounded workers
  -> RGB conversion and bounded PTS pacing
  -> online hand inference
  -> MP4 recording and replay decoding
  -> IMU Madgwick preview
```

The UI owns presentation only. WebRTC, recording, perception, storage, and
annotation business logic remain in their owning `src/` packages.

## Fluent home interface

`FluentWindow` supplies the collapsible left navigation used by the official
Fluent Gallery. Only Home is registered in this iteration. Common controls use
PyQt-Fluent-Widgets components: `SegmentedWidget`, `HeaderCardWidget`, Fluent
buttons, `ComboBox`, `Slider`, `InfoBadge`, `ProgressRing`, and `InfoBar`.
All UI text uses Fluent label components; raw Qt `QLabel` is not used.
The mode strip combines live/replay selection, source/session context, and the
compact capture controls for video, recording, and session creation. Offline
generation uses `StateToolTip` for its real task lifecycle, with `InfoBar`
reserved for terminal success and error feedback.

Only VideoCanvas uses custom painting because the component library has no
direct RGB frame surface:

- VideoCanvas paints immutable NumPy RGB buffers and frame-aligned hand data.
- SpatialSyncCanvas uses pyqtgraph 0.14.0's OpenGL GLViewWidget with the
  environment's explicit PyOpenGL 3.1.0 dependency for the
  right-side spatial synchronization view, with IMU pose and hand-tracking 3D
  keypoints rendered as real 3D line and scatter items. pyqtgraph is MIT
  licensed and compatible with the GPLv3 client.

The same `VideoCanvas` switches between live input and replay. Switching modes
does not reconnect WebRTC, create another media surface, or stop the current
stream.

In live mode the right side has two peer sections: a standalone dark space sync
OpenGL viewport and one `同步数据` Fluent card. The data card contains compact
IMU, left-hand, right-hand, and frame-link status blocks. The old capture, hand,
frame, and IMU vertical cards are not part of the home page.

## Four-by-three video

The display canvas is always 4:3. It selects the largest 4:3 rectangle inside
the available workspace, such as 960x720 or 1024x768. The current 640x480
Glass3 stream fills it without crop or stretch. A non-4:3 recording is fitted
inside the canvas with letterboxing; the UI never crops it to fill the area.

`VideoCanvas` creates `QImage.Format_RGB888` over the immutable, contiguous
NumPy frame and keeps the `LiveFrame` alive until replacement. No float texture
copy or encoded preview is introduced. The hand result is painted only when
its session, connection, and frame index match the displayed frame.

## Replay and storage

Replay uses the existing PyAV worker. Frame time remains `PTS * time_base`,
with pause, step, seek, and 0.25x to 2.0x playback on the same decoder.
Recording storage is scanned once at startup and only after the operator
presses Refresh. Hashing stored clips never runs on a UI timer.

## Run and verify

```powershell
.\scripts\start-client.ps1
conda run -n egoglass python -m pytest
conda run -n egoglass python -m pytest -q evals
conda run -n egoglass ruff check src ui tests evals scripts
conda run -n egoglass python scripts\benchmark_native_texture.py
```

The optional 60-second UI soak is:

```powershell
$env:EGOGLASS_RUN_UI_SOAK = "1"
conda run -n egoglass python -m pytest -q -s `
  evals\test_ui_native_runtime.py::test_four_by_three_canvas_sustains_thirty_fps_for_sixty_seconds
```

Closing the native window finalizes active capture, stops replay and runtime
workers, and releases the signaling and discovery ports.
