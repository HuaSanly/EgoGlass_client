# Ingest Gateway

The ingest gateway terminates the direct Glass3 WebRTC stream, relays its live
track to the loopback operator console, receives IMU telemetry, and records
operator-selected clips. It is an isolated client package and does not depend
on the operator console.

## Direct WebRTC path

For normal use, start the complete Windows client from the client repository:

~~~powershell
.\scripts\start-client.ps1
~~~

After it reports `EgoGlass client is ready`, open EgoGlass directly from the
Glass3 application list. The glasses discovers the client on UDP port 8771 and
starts WebRTC without stored configuration or ADB parameters. A newly
authenticated glasses offer replaces any stale peer left by an earlier app
process, so restarting the glasses app does not require restarting Windows.

For isolated signaling diagnostics only, start the gateway directly and pass
the generated runtime values through Intent extras:

~~~powershell
uv run egoglass-ingest-gateway --host 0.0.0.0 --port 8770
~~~

~~~powershell
adb shell am start -n com.egoglass.glasses/.MainActivity `
  --es signaling_url http://192.168.1.20:8770/api/v1/webrtc/sessions `
  --es pairing_token <runtime-token>
~~~

WebRTC video is encrypted with DTLS-SRTP. The HTTP signaling exchange is
cleartext in v1 and must stay on a trusted LAN; the pairing token is never
returned by status or error APIs.

The gateway uses aiortc's `MediaRelay` to expose the active decoded track to
one loopback WebRTC viewer. The viewer subscription is unbuffered so stale
frames do not accumulate. It does not create JPEG snapshots or reduce the
incoming frame size or cadence.

Decoded aiortc PTS starts from a receiver-relative RTP origin and therefore
must not be anchored to the first Glass3 metadata message: the encoder can drop
capture callbacks before its first output frame. The gateway first pairs each
decoded frame with the metadata message nearest in client receipt time, then
selects the dominant RTP-offset cluster. It waits for at least 60 candidates
and a 1.5x lead over the next cluster before releasing buffered matches. The
1,000-tick match tolerance covers the measured 622-tick device residual while
remaining below half of a 30 fps frame. Camera-start generation changes clear
the pending queues and require a fresh calibration.

## Capture sessions and HD recording

The first recording request automatically creates a collection session before
the server-authoritative three-second countdown. It immediately starts
persisting IMU telemetry. Stopping or cancelling a clip does not stop the
session or its IMU timeline, and WebRTC reconnection creates a connection
segment instead of a new collection session. One collection session can own
many MP4 clips.

The loopback New Session command finalizes the current session and arms the
next recording request to create another one. It does not create an empty
directory. New Session returns HTTP 409 while a countdown or clip is active.
The idempotent finalize command used during client shutdown is different: it
cancels a countdown or flushes an active MP4, drains telemetry, checkpoints
SQLite WAL, writes the final quality report and manifest, and only then
returns.

Recording accepts only an active 1280x720 Glass3 source. It uses the incoming
decoded frames at their full dimensions, a nominal 30 FPS H.264 stream, and a
buffered aiortc `MediaRelay` subscription so the operator preview does not
consume recording frames. After muxing, PyAV reads back the actual MP4 PTS and
time base for every encoded frame. Missing frame timing fails the clip; the
gateway never derives PTS as `frame_index / fps`. PyAV supplies the FFmpeg
bindings and the `libx264` encoder; no separate `ffmpeg.exe` process is
required.

New sessions use the versioned `capture-session-v1` layout:

~~~text
<session-id>/
  session.json
  quality.json
  media/
    <clip-id>.mp4
  telemetry/
    telemetry.sqlite
  annotations/
  derived/
~~~

`telemetry.sqlite` runs in WAL mode and preserves raw accelerometer and
gyroscope samples, capabilities, connection segments, every valid video
metadata message, decoded-frame matches, every MP4 frame index, lifecycle
events, and every source timestamp needed by later perception processing. New
capture records remain alignment-pending and do not generate clock mappings. A
bounded queue keeps SQLite work out of WebRTC callbacks and a single worker
commits batches. Queue overflow is counted and makes the session ineligible for
training instead of silently disappearing.

Set the root with `--recordings-root` or `EGOGLASS_RECORDINGS_ROOT`. The CLI
default is `local-data/recordings` relative to the launch directory:

~~~powershell
uv run egoglass-ingest-gateway `
  --recordings-root F:\data\Project\EgoGlass\EgoGlass_client\local-data\recordings
~~~

Only checksummed files listed in a valid session manifest appear in the library
or media endpoint. A source disconnect finalizes frames already received. An
encoder or manifest failure moves the runtime to `error`, removes an invalid
partial output, and preserves the collection telemetry for recovery. On the
next startup, sessions left `active` or `finalizing` are marked `incomplete`,
SQLite WAL is replayed and checkpointed, and `quality.json` records the unclean
recovery. Incomplete sessions are never silently treated as training-ready.

The fixed 1280x720 profile applies to new recordings only. Library manifests
retain each clip's actual dimensions so recordings from earlier profiles,
including 1920x1080 sessions, remain visible, playable, renameable, and
deletable after a profile change.

Deleting a clip is loopback-only. The gateway first moves the MP4 out of its
published path, atomically removes it from `session.json`, and then deletes the
file. A failed operation restores both the MP4 and manifest. Deleting the last
clip from a collection session retains its telemetry and session directory.
The separate session-delete endpoint atomically removes the entire inactive
session directory; active sessions return HTTP 409. Legacy video-only sessions
retain their earlier last-clip deletion behavior.

Session display names are optional, limited to 64 visible characters, and
stored in `session.json`. Renaming changes only this display metadata; the
stable session ID, directory, media URLs, and video files are unchanged. Old
manifests without `display_name` remain valid.

Validate a real-device result by opening and decoding it with the same PyAV
runtime used by the service:

~~~powershell
uv run python scripts/inspect-recording.py F:\path\to\session\clip.mp4
~~~

The command exits nonzero unless the file is a finalized, playable MP4 with
one H.264 1280x720 video stream, nominal 30 FPS, and at least one decodable
frame. It prints the measured stream properties and decoded frame count as
JSON on success.

The glasses also opens an ordered, reliable `stream-control-v1` DataChannel.
The loopback API can stop camera capture without closing WebRTC, then start it
again on the same peer connection. Command IDs are generated by the gateway
and matched to strict status acknowledgements from the glasses. The control
API is not available to LAN callers.

The IMU path opens a separate unordered, zero-retransmit
`imu-telemetry-experimental-v0` DataChannel. It accepts only Android
accelerometer and gyroscope capabilities and raw three-axis samples. The
loopback status endpoint reports bounded counters, observed arrival rates, and
the latest sample, while an active collection session persists every sample.

The gateway does not fit video-to-IMU clock mappings. It preserves Rokid camera
time, Android sensor time, callback elapsed-realtime, Windows receipt time, RTP
time, and exact MP4 PTS/time bases. A later versioned perception pipeline uses
those values with VIO output to associate frames, camera pose, IMU, and hands
without rewriting raw capture records. Missing device, firmware, glasses
application, or Git revision provenance still makes a session ineligible, and
raw capture remains ineligible until perception processing emits its own
quality decision.

By default the gateway also listens for EgoGlass discovery v1 requests on UDP
port 8771. It responds only to private or loopback IPv4 sources and returns the
source-routed LAN signaling URL plus the current process-only pairing token.
Discovery is intended only for the same trusted LAN as the cleartext v1 HTTP
signaling path.

## Run

This service uses PyAV, the maintained Python binding for FFmpeg, so a separate
system ffmpeg.exe is not required.

~~~powershell
uv sync --group dev
uv run egoglass-ingest-gateway
~~~

The CLI binds to `0.0.0.0:8770` by default for Glass3 signaling. Viewer,
control, IMU, recording, and media-library APIs enforce loopback access.

## API

- GET /api/v1/health
- GET /api/v1/webrtc/status
- GET /api/v1/webrtc/imu/status (loopback-only experimental IMU evidence)
- POST /api/v1/webrtc/sessions
- POST /api/v1/webrtc/viewer/sessions (loopback-only WebRTC preview)
- GET /api/v1/webrtc/control (loopback-only capture state)
- POST /api/v1/webrtc/control/commands (loopback-only `{ "action": "start|stop" }`)
- GET /api/v1/recordings/status (loopback-only recording state)
- POST /api/v1/recordings/commands (loopback-only `{ "action": "start|stop" }`)
- POST /api/v1/recordings/session-commands (loopback-only `{ "action": "new|finalize" }`)
- GET /api/v1/recordings/library (loopback-only active and historical session groups)
- PATCH /api/v1/recordings/sessions/{session_id} (loopback-only display-name update)
- DELETE /api/v1/recordings/sessions/{session_id} (loopback-only whole-session deletion)
- DELETE /api/v1/recordings/clips/{session_id}/{clip_id} (loopback-only deletion)
- GET /api/v1/recordings/media/{session_id}/{clip_id} (loopback-only MP4)
- Interactive schema: http://127.0.0.1:8770/api/docs

## Verification

~~~powershell
uv run ruff check src tests evals
uv run pytest
uv run pytest -q evals
~~~

Gate tests use in-process WebRTC peers and synthetic frames and never access an
external stream. The real-device gate requires an online Glass3 on the same LAN
and records the device, firmware, media profile, and observed transport result.
