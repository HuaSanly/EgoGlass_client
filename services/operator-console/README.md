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

The home page uses one two-state control for starting and stopping the Glass3
stream. Its adjacent recording control starts a gateway-owned three-second
countdown, can cancel during that countdown, and stops an active recording. The
countdown shown over the live video is calculated from the gateway's
`recording_starts_at_unix_ms`; the console does not start its own recorder or
invent recording state. Clips are fixed at 1920x1080, 30 FPS, H.264 in MP4.

The `/storage` page polls the loopback recording library and groups playable
clips by the originating Glass3 WebRTC session. Media URLs are accepted only
after the complete v1 JSON payload and loopback origin are validated. Loading,
empty, error, and current recording states have separate visible treatments.
The document remains fixed to the native window; only the session list scrolls.
The first storage view contains time-named session folders rather than video
players. Opening one folder renders only that session's clips. Operators can
rename a folder; the custom name is persisted by the gateway without changing
the stable media path.
Each clip can be deleted after an explicit confirmation. The console removes a
card only after the gateway confirms that both its MP4 and manifest entry were
deleted.

## Run as a Windows app

For the complete workspace client, use `..\..\scripts\start-client.ps1` from
the client repository root. It owns both the ingest gateway and this desktop
window.

For operator-console-only development:

```powershell
uv sync --group dev
uv run egoglass-desktop
```

The desktop launcher provides a native title bar, a minimum 1100x700 window,
single-instance enforcement, a dynamic loopback port, an HttpOnly desktop
session cookie, rotating logs under `%LOCALAPPDATA%\EgoGlass\logs`, and graceful
server shutdown when the window closes.

The browser development server remains available for UI development:

```powershell
uv run egoglass-console
```

The separate ingest gateway is the only service that binds to the LAN. The
desktop server and browser development server remain loopback-only.

## Build

```powershell
.\scripts\build-desktop.ps1
```

The script synchronizes dependencies, builds the one-folder application with
PyInstaller, verifies the installed WebView2 Runtime, and runs the packaged
`--smoke-test`. Build output stays outside Git under `dist/EgoGlass/`.

## API

- `GET /api/v1/health`

The UI consumes these loopback-only ingest-gateway contracts:

- `GET /api/v1/recordings/status`
- `POST /api/v1/recordings/commands`
- `GET /api/v1/recordings/library`
- `PATCH /api/v1/recordings/sessions/{session_id}`
- `DELETE /api/v1/recordings/clips/{session_id}/{clip_id}`
- `GET /api/v1/recordings/media/{session_id}/{clip_id}`

## Verification

```powershell
uv run pytest
uv run pytest -q evals
uv run ruff check src tests evals
```

Gate tests cover desktop authentication, static application delivery, removed
API behavior, real preview connection state, stream and recording state
binding, session-grouped playable media, strict recording payload validation,
and the local Three.js/AHRS IMU visualization contract.
The eval suite prevents placeholder data paths or independently invented
control state from returning to the shipped runtime.
