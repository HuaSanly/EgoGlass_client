# EgoGlass Client

The client repository owns the Windows ingest, online inference orchestration,
operator tooling, and data platform. The first runnable slice is the operator
console with a deterministic synthetic source. It keeps the UI and API usable
before the Glass3 WebRTC transport is implemented.

## Services

- `services/operator-console/`: FastAPI API, WebSocket telemetry, simulated
  three-dimensional hand trajectories, and the browser operator console.

Future services will be added behind versioned contracts rather than imported
from the operator console.

## Windows desktop

```powershell
cd services/operator-console
uv sync --group dev
uv run egoglass-desktop
```

The desktop command opens the operator console in a native Windows WebView2
window. It does not open the system browser. The bundled FastAPI server uses a
dynamic loopback port and shuts down with the window.

Build and verify the local executable with:

```powershell
cd services/operator-console
.\scripts\build-desktop.ps1
```

The ignored build output is `dist/EgoGlass/EgoGlass.exe`.

## Verification

```powershell
cd services/operator-console
uv run pytest
uv run pytest -q evals
uv run ruff check src tests evals
```

The simulator is not evidence of real-world calibration. A real device stream
must carry a verified calibration profile before the glasses application is
allowed to render world-aligned feedback.
