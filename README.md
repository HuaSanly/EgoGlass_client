# EgoGlass Client

The client repository owns the Windows ingest, online inference orchestration,
operator tooling, and data platform. The current runnable slice receives direct
Glass3 WebRTC video and renders its live preview in the operator console. It
does not generate placeholder trajectories, metrics, calibration, or session
data.

## Services

- `services/ingest-gateway/`: terminates the direct Glass3 WebRTC video and
  frame-metadata channels, relays the live track to one loopback viewer, and
  also probes GB28181-generated RTSP as a fallback.
- `services/operator-console/`: authenticated local UI for the real Glass3
  WebRTC track, measured displayed FPS, and client-side connection events.

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

The direct video path has passed a real-device first-frame check. World-aligned
feedback is a separate future service and must carry a verified calibration
profile before the glasses application is allowed to render it.
