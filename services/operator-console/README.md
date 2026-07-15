# Operator Console

The operator console is a self-contained FastAPI service for Windows. It serves
the browser UI, owns operator settings, and streams telemetry through a
versioned WebSocket endpoint.

The primary viewer displays the live Glass3 WebRTC preview from the local
ingest gateway. Hand trajectories and inference telemetry remain deterministic
and synthetic until the online inference service replaces them.

## Run as a Windows app

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
- `GET /api/v1/state`
- `PUT /api/v1/settings`
- `POST /api/v1/session/start`
- `POST /api/v1/session/stop`
- `POST /api/v1/recording/start`
- `POST /api/v1/recording/stop`
- `WS /api/v1/telemetry`

## Verification

```powershell
uv run pytest
uv run pytest -q evals
uv run ruff check src tests evals
```

Gate tests cover validation, state transitions, trajectory ranges, freshness,
and static application delivery. The eval suite checks the complete simulated
session contract and quality thresholds separately.
