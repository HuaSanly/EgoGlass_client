# Operator Console

The operator console is a self-contained FastAPI service for Windows. It serves
the browser UI, owns operator settings, and streams telemetry through a
versioned WebSocket endpoint.

The current media source is deterministic and synthetic. Its output follows the
same coordinate, calibration, and freshness rules expected from the future
Glass3 ingest gateway.

## Run

```powershell
python -m uv sync --group dev
python -m uv run egoglass-console
```

Use `--host` and `--port` to override the default `127.0.0.1:8765` binding.
Only a future ingest gateway will bind to the LAN by default.

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
python -m uv run pytest
python -m uv run pytest -q evals
python -m uv run ruff check src tests evals
```

Gate tests cover validation, state transitions, trajectory ranges, freshness,
and static application delivery. The eval suite checks the complete simulated
session contract and quality thresholds separately.
