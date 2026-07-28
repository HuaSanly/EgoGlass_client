# Operator Console

The operator console is a self-contained FastAPI service for Windows. It serves
an authenticated local UI that displays the live Glass3 WebRTC preview from the
ingest gateway. It does not create placeholder media, trajectories, or session
state.

WebView2 receives a real WebRTC video track from the gateway and measures the
displayed cadence from rendered frames. There is no JPEG polling path. The UI
keeps the negotiated frame dimensions and cadence unchanged.

When the Glass3 control DataChannel is online, the signal panel shows its real
capture state and enables start/stop commands. The console polls
`GET http://127.0.0.1:8770/api/v1/webrtc/control` and sends commands to
`POST http://127.0.0.1:8770/api/v1/webrtc/control/commands`. Unavailable,
transitioning, and failed control states keep both buttons disabled. A command
or status error is shown in the control panel without interrupting the live
preview connection.

The lower right workspace renders the live Glass3 IMU as a Three.js model. It
polls the loopback-only `GET http://127.0.0.1:8770/api/v1/webrtc/imu/status`
endpoint and feeds the latest accelerometer and gyroscope samples into the
vendored Madgwick filter from `ahrs`. The model shows relative orientation from
the current reference, raw vector magnitudes, arrival rates, and the device
axes. It does not integrate position or claim an absolute heading because the
verified Glass3 sensor set has no magnetometer. The reset control makes the
current orientation the new relative origin. Because the model faces along its
positive Z axis, the display adapter reverses rotation around the horizontal X
axis when converting the fused Glass3 pose to Three.js. This makes downward
head motion render downward while leaving raw IMU samples and fusion inputs
unchanged.

At the start of each IMU session, the console waits for consecutive stationary
samples, initializes the vendored AHRS filter from averaged gravity, and then
requires a stable quaternion window before publishing the automatic relative
origin. Motion or continued filter convergence resets that window. This keeps
the filter's startup correction out of the rendered pose while preserving the
manual reset control.

The home page uses one two-state control for starting and stopping the Glass3
stream. Its adjacent recording control starts a gateway-owned three-second
countdown, can cancel during that countdown, and stops an active recording. The
first recording request automatically creates a capture session before the
countdown, and the gateway persists IMU continuously until that session ends.
The countdown shown over the live video is calculated from the gateway's
`recording_starts_at_unix_ms`; the console does not start its own recorder or
invent recording or session state. The compact session strip polls the gateway
library and shows the current time-based name, lifecycle state, persisted IMU
count, recorded-frame metadata coverage, and deferred perception-alignment
state. New clips are fixed at 1280x720, 30 FPS, H.264 in MP4; the storage
library still displays historical clips at their recorded dimensions.

The New Session button is available only for an active session with no clip or
countdown in progress. It finalizes the current session and arms the next
recording request; it does not create an empty directory. The next recording
automatically starts a new session and continuous IMU persistence.

Closing the native window or pressing Ctrl+C asks the ingest gateway to
finalize the active capture session before the Windows Job Object performs its
port-cleanup fallback. A failed finalization is reported as a warning and the
process tree is still terminated so ports are not leaked.

The workspace below the live video is the hand-tracking monitor. It shows the
rectified algorithm image, left/right confidence and backend, grasp state,
input frame, inference duration, processed frames, and dropped frames. The
preview image is produced from the exact rectified frame used by HaMeR, so its
keypoints are not drawn over a differently rotated raw WebRTC image. The same
panel selects completed capture sessions, starts offline replay, reports frame
progress, and plays the generated annotated MP4.

The `/storage` page polls the loopback recording library and treats each capture
session as one data folder. A folder remains visible when it has no video clips
but does contain continuous IMU or recovered session data. Its detail view
shows lifecycle completeness, IMU counts and sequence anomalies, recorded-frame
metadata coverage, connection segments, and source-time status above all
playable clips. New captures state that source time is preserved and perception
alignment is pending. Historical video-only manifests remain manageable and are
explicitly labelled instead of being assigned invented telemetry quality.
Media URLs are accepted only after the complete v1 JSON payload and loopback
origin are validated. Loading, empty, error, and current recording states have
separate visible treatments. The document remains fixed to the native window;
only the session list scrolls. Operators can rename a folder without changing
its stable storage path. A clip or complete/incomplete session folder can be
deleted after explicit confirmation; active and finalizing sessions cannot be
deleted. The console updates the library only after gateway success.

The `/annotations` page owns its local annotation workflow. It provides manual
task-attempt boundaries, reviewed whole-clip and non-overlapping fixed-window
proposals, episode semantic labels, and internal phase intervals. Drafts
autosave with optimistic revision checks. Publishing requires complete task
labels and at least one contained phase per episode, then writes an immutable
revision without changing source MP4 or telemetry.

## Run as a Windows app

For the complete workspace client, use `.\scripts\start-client.ps1` from
the client repository root. It owns the ingest gateway and this desktop window.

For operator-console-only development:

```powershell
conda run -n egoglass python -m operator_console.desktop
```

The desktop launcher provides a native title bar, a minimum 1100x700 window,
single-instance enforcement, a dynamic loopback port, an HttpOnly desktop
session cookie, rotating logs under `%LOCALAPPDATA%\EgoGlass\logs`, and graceful
server shutdown when the window closes.

The browser development server remains available for UI development:

```powershell
conda run -n egoglass python -m operator_console.app
```

The separate ingest gateway is the only service that binds to the LAN. The
desktop server and browser development server remain loopback-only.

## Build

```powershell
.\scripts\build-desktop.ps1
```

The script uses the configured Conda environment, builds the one-folder application with
PyInstaller, verifies the installed WebView2 Runtime, and runs the packaged
`--smoke-test`. Build output stays outside Git under `dist/EgoGlass/`.

## API

- `GET /api/v1/health`

The UI consumes these loopback-only ingest-gateway contracts:

- `GET /api/v1/recordings/status`
- `POST /api/v1/recordings/commands`
- `POST /api/v1/recordings/session-commands`
- `GET /api/v1/recordings/library`
- `PATCH /api/v1/recordings/sessions/{session_id}`
- `DELETE /api/v1/recordings/sessions/{session_id}`
- `DELETE /api/v1/recordings/clips/{session_id}/{clip_id}`
- `GET /api/v1/recordings/media/{session_id}/{clip_id}`
- `GET /api/v1/perception/hand-tracking/status`
- `GET /api/v1/perception/hand-tracking/preview.jpg`
- `POST /api/v1/perception/hand-tracking/replays`
- `GET /api/v1/perception/hand-tracking/replays/{session_id}/{run_id}/{clip_id}`

The annotation page uses the operator console's same-origin API:

- `GET /api/v1/annotations/workspace`
- `GET /api/v1/annotations/sessions/{session_id}`
- `PUT /api/v1/annotations/sessions/{session_id}/draft`
- `POST /api/v1/annotations/sessions/{session_id}/proposals`
- `POST /api/v1/annotations/sessions/{session_id}/publish`
- `GET /api/v1/annotations/media/{session_id}/{clip_id}`

## Verification

```powershell
conda run -n egoglass python -m pytest
conda run -n egoglass python -m pytest -q evals
conda run -n egoglass ruff check src tests evals
```

Gate tests cover desktop authentication, static application delivery, removed
API behavior, real preview connection state, stream and recording state
binding, live hand-tracking status, offline replay, session-grouped playable media, strict
recording payload validation, and the local Three.js/AHRS IMU visualization
contract.
The eval suite prevents placeholder data paths or independently invented
control state from returning to the shipped runtime.
