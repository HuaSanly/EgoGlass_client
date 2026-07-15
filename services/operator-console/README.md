# Operator Console

The operator console is a self-contained FastAPI service for Windows. It serves
an authenticated local UI that displays the live Glass3 WebRTC preview from the
ingest gateway. It does not create placeholder media, trajectories, metrics,
calibration, recording, or session state.

WebView2 receives a real WebRTC video track from the gateway and measures the
displayed cadence from rendered frames. There is no JPEG polling path. The UI
keeps the negotiated frame dimensions and cadence unchanged.

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

## Verification

```powershell
uv run pytest
uv run pytest -q evals
uv run ruff check src tests evals
```

Gate tests cover desktop authentication, static application delivery, removed
API behavior, and the real preview connection state. The eval suite prevents
placeholder data paths from returning to the shipped runtime.
