# Ingest Gateway

The gateway owns Glass3 discovery, WebRTC signaling, decoded video ingress,
raw IMU ingress, recording, and loopback diagnostics. It does not import Qt or
any perception or offline-processing package.

## Start

The normal entry point starts the gateway and native UI together:

```powershell
.\scripts\start-client.ps1
```

For isolated signaling diagnostics:

```powershell
conda run -n egoglass python -m ui.gateway.app `
  --host 0.0.0.0 `
  --port 8770 `
  --recordings-root local-data\recordings
```

Discovery listens on UDP 8771. Signaling listens on TCP 8770. WebRTC media is
protected by DTLS-SRTP; the v1 HTTP signaling exchange must remain on a trusted
LAN. Control, recording, files, and diagnostics are loopback-only.

## Capture Path

aiortc terminates the H.264 WebRTC stream. The canonical ingest subscription is
losslessly buffered so every MP4 frame can be joined to its Glass3 metadata.
`LiveFrameBuffer` keeps only bounded
newest-frame state for preview and never accumulates decoded media behind the
UI. Recording has its own relay subscription, waits briefly for each frame's
authoritative metadata, skips only unmatched frames, preserves monotonic source
PTS, and writes H.264 MP4 through PyAV. This makes sparse metadata backpressure
visible as dropped frame IDs instead of invalidating the complete recording.

The Glass3 `stream-control-v1` data channel starts and stops camera capture.
The experimental `imu-telemetry-experimental-v0` channel carries raw Android
accelerometer and gyroscope samples. The recording client caches raw samples
for plots and statistics only; it does not estimate orientation.

`connection_session_id` remains an internal source identifier across WebRTC
reconnects. It is used only while matching frames and is never persisted in the
four-file recording contract.

The reliable ordered `recording-control-v1` channel carries wearer recording
commands separately from camera control. Glass3 sends a strict
`recording_control_command` after a temple double tap; the client serializes it
with Qt and HTTP recording commands, starts the 640x480 stream when needed, and
returns authoritative `recording_control_status` snapshots. Active countdown,
recording, and finalization states are refreshed every 250 ms. Duplicate command
IDs and events from replaced WebRTC peers cannot repeat a recording transition.

## One Recording Contract

Each Start-to-Stop operation owns one 32-character `recording_id` and one
directory:

```text
<recordings-root>/<recording_id>/
  video.mp4
  camera.csv
  imu.csv
  calibration.yaml
```

Writing occurs under `.recording-<recording_id>.partial/`. Completion validates
the exact layout, strict CSV schemas, decoded frame count, clock monotonicity,
IMU coverage, H.264 profile, and calibration resolution, then atomically
renames the directory. A partial recording is never published by the library.

`camera.csv` has exactly one row per encoded MP4 frame:

```csv
frame_idx,frame_id,rokid_timestamp_ns,device_monotonic_ns
```

`rokid_timestamp_ns` is `captured_at_rokid_sdk_ms * 1_000_000`.
`device_monotonic_ns` is the Glass3
`received_at_elapsed_realtime_ns` callback timestamp.

`imu.csv` preserves every raw accelerometer and gyroscope row accepted from
countdown start through the last video frame:

```csv
sensor_type,sequence,timestamp_ns,x,y,z
```

`timestamp_ns` is the unchanged Android `SensorEvent.timestamp`. Capture does
not filter, interpolate, resample, or transform IMU coordinates.

`calibration.yaml` contains a typed snapshot at the recorded resolution. Until
device calibration is available, the client writes unit intrinsics, zero
distortion, identity `T_cam_imu`, and null IMU noise values.

The gateway never writes `telemetry.sqlite`, collection sessions, clips,
processing jobs, or algorithm results.

## Lifecycle

Start begins the server-authoritative countdown and opens the partial writer so
countdown IMU is retained. Stop flushes the MP4, closes the IMU window at the
last encoded frame, validates all artifacts, removes temporary matching data,
and publishes atomically. WebRTC reception continues while finalization runs.

An interrupted partial directory remains distinguishable from a completed
recording. Recovery either validates and completes the exact partial contract
or marks it incomplete; it cannot silently appear in the library.

## Loopback API

The recording-only API surface contains:

- `GET /api/v1/health`
- `GET /api/v1/webrtc/status`
- `GET /api/v1/webrtc/imu/status`
- `POST /api/v1/webrtc/sessions`
- `GET /api/v1/webrtc/control`
- `POST /api/v1/webrtc/control/commands`
- `GET /api/v1/native-display/status`
- `GET /api/v1/recordings/status`
- `POST /api/v1/recordings/commands`
- `GET /api/v1/recordings/library`
- `DELETE /api/v1/recordings/{recording_id}`
- `GET /api/v1/recordings/{recording_id}/video.mp4`
- `GET /api/v1/recordings/{recording_id}/camera.csv`
- `GET /api/v1/recordings/{recording_id}/imu.csv`
- `GET /api/v1/recordings/{recording_id}/calibration.yaml`

There are no session, clip, perception, processing, VIO, annotation, or
dataset endpoints.

Interactive schema: <http://127.0.0.1:8770/api/docs>

## Verification

```powershell
conda run -n egoglass python scripts\inspect-recording.py `
  local-data\recordings\<recording-id>
conda run -n egoglass python -m pytest
conda run -n egoglass python -m pytest -q evals
```
