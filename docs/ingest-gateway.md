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

aiortc terminates the H.264 WebRTC stream. `LiveFrameBuffer` keeps only bounded
newest-frame state for preview and never accumulates decoded media behind the
UI. Recording has its own relay subscription, preserves monotonic source PTS,
and writes H.264 MP4 through PyAV.

The Glass3 `stream-control-v1` data channel starts and stops camera capture.
The experimental `imu-telemetry-experimental-v0` channel carries raw Android
accelerometer and gyroscope samples. The recording client caches raw samples
for plots and statistics only; it does not estimate orientation.

`connection_session_id` remains an internal source identifier across WebRTC
reconnects. It is present in CSV evidence but is never the recording primary
key or a user-facing collection entity.

## One Recording Contract

Each Start-to-Stop operation owns one 32-character `recording_id` and one
directory:

```text
<recordings-root>/<recording_id>/
  manifest.json
  video.mp4
  imu.csv
  frames.csv
  quality.json
  annotations/
  derived/
```

Writing occurs under `.recording-<recording_id>.partial/`. Completion validates
the exact layout, strict CSV schemas, counts, video span, artifact sizes, and
SHA256 hashes, then atomically renames the directory. A partial recording is
never published by the library.

`frames.csv` has one row per encoded MP4 frame and preserves:

- `frame_index` and `recording_time_ns`
- MP4 PTS and time base
- internal connection ID
- Glass3 frame ID and camera-start generation
- Rokid SDK, Android elapsed-realtime, RTP, and client monotonic timestamps
- metadata match status and match residual

`imu.csv` preserves every raw accelerometer and gyroscope row accepted from
countdown start through the last video frame. It includes the internal
connection ID, sensor sequence, Android and client times, accuracy, XYZ values,
and `inside_video_span`.

`quality.json` reports matched-frame coverage, IMU sensor counts, sequence
gaps, duplicates, out-of-order rows, queue overflow, and timestamp mapping
evidence. `manifest.json` records device/software provenance, time origin,
media profile, row counts, sizes, and SHA256 values.

The gateway never writes `telemetry.sqlite`, collection sessions, clips,
processing jobs, or algorithm results.

## Lifecycle

Start begins the server-authoritative countdown and opens the partial writer so
countdown IMU is retained. The first encoded frame establishes the video-span
start. Stop flushes the MP4 and indexes its actual PTS, closes the time window
at the final video frame, validates all artifacts, and publishes atomically.
WebRTC reception continues while finalization runs.

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
- `GET /api/v1/recordings/{recording_id}/frames.csv`
- `GET /api/v1/recordings/{recording_id}/imu.csv`

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
