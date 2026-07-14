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

## Quick start

```powershell
cd services/operator-console
python -m uv sync --group dev
python -m uv run egoglass-console
```

Open `http://127.0.0.1:8765`.

## Verification

```powershell
cd services/operator-console
python -m uv run pytest
python -m uv run pytest -q evals
python -m uv run ruff check src tests evals
```

The simulator is not evidence of real-world calibration. A real device stream
must carry a verified calibration profile before the glasses application is
allowed to render world-aligned feedback.
