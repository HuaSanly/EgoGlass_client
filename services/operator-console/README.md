# Operator Console

The operator console is a self-contained FastAPI service for Windows. It serves
an authenticated local UI that displays the live Glass3 WebRTC preview from the
ingest gateway. It does not create placeholder media, trajectories, recording,
or session state.

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
current orientation the new relative origin.

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

## Verification

```powershell
uv run pytest
uv run pytest -q evals
uv run ruff check src tests evals
```

Gate tests cover desktop authentication, static application delivery, removed
API behavior, real preview connection state, stream-control state binding, and
the local Three.js/AHRS IMU visualization contract.
The eval suite prevents placeholder data paths or independently invented
control state from returning to the shipped runtime.
