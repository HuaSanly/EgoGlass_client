# Operator Console Evals

The periodic eval suite measures the operator console contract separately from
gate tests. The shipped runtime consumes only the real Glass3 preview.

Current pass criteria:

- the Glass3 preview endpoint is the only viewer source;
- waiting, connected, and disconnected UI states follow real image load events;
- no placeholder data generator, WebSocket, canvas renderer, or stale control
  API is present in the shipped package;
- desktop authentication and repeated start/stop lifecycle remain recoverable.

Run with:

```powershell
python -m uv run pytest -q evals
```

Real-device WebRTC and long-session evals remain separate named hardware gates.
