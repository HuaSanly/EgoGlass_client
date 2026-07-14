# Operator Console Evals

The periodic eval suite measures the quality contract separately from gate
tests. It uses the deterministic source so regressions can be reproduced
without a Glass3 device.

Current pass thresholds:

- both hands are present in every evaluated prediction;
- each trajectory has the configured horizon and strictly increasing offsets;
- every waypoint stays in front of the camera and inside the synthetic work
  volume;
- reported feedback latency remains below the 500 ms freshness budget;
- the source reports no queue buildup and less than two percent dropped frames.

Run with:

```powershell
python -m uv run pytest -q evals
```

Real-device WebRTC, calibration, and long-session evals remain separate named
hardware gates and cannot be satisfied by this suite.
