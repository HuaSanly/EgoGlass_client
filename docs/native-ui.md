# Native Recording UI

The native Windows client uses PyQt6 6.8.1 and
PyQt6-Fluent-Widgets 1.11.2. It has no browser, WebView, MJPEG preview,
perception page, processing queue, dataset page, annotation page, settings
page, or OpenGL spatial scene.

## Recording Console

The first screen is the live recording console. Its header reports connection,
resolution, source FPS, and recording state. The main 4:3 video canvas displays
the newest immutable RGB frame without crop or stretch. Controls start or stop
camera capture and start or stop one recording.

The mode strip shows the active `recording_id`, elapsed duration, encoded frame
count, and written IMU row count. There is no session or clip command.

The right side is a raw IMU monitor:

- accelerometer X/Y/Z plot
- gyroscope X/Y/Z plot
- observed sample rates
- latest-sample latency
- sequence gaps, duplicates, out-of-order rows, and queue overflow
- recording time window and CSV writer state

Plots use a bounded history and update existing curves in place. No attitude
filter, pose reset, head model, hand model, or spatial trajectory is loaded.

## Recording Library

The library shows one item per validated recording. Each item displays a video
thumbnail, recorded time, duration, resolution, frame count, IMU row count,
protocol status, and delete action. Invalid and partial directories are excluded.

Opening an item starts lightweight replay from `video.mp4`. The IMU cursor uses
`camera.csv.device_monotonic_ns` and `imu.csv.timestamp_ns`, so accelerometer
and gyroscope samples follow the displayed frame. MP4 PTS are indexed only in
memory. Replay does not run algorithms or write output.

Deletion is available only for inactive recordings and targets the exact
recording directory selected by its validated ID.

## Runtime Ownership

```text
PyQt main thread
  -> Fluent recording console and library
  -> video upload, labels, badges, and IMU plots

asyncio runtime thread
  -> Uvicorn, discovery, and aiortc
  -> capture control and raw telemetry

bounded workers
  -> RGB conversion
  -> MP4 and CSV recording
  -> recording finalization and protocol validation
  -> lightweight replay decode
```

Closing the window stops capture, finalizes an active recording when possible,
stops gateway workers, and releases signaling and discovery ports.

## Run and Verify

```powershell
.\scripts\start-client.ps1
conda run -n egoglass python -m pytest
conda run -n egoglass python -m pytest -q evals
conda run -n egoglass ruff check src ui tests evals scripts
```

No program restart is needed after adding or deleting a stored recording. Code
or configuration changes require closing and restarting the native client.
