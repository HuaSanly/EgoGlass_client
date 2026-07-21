# Data Platform Evals

Run with:

```powershell
uv run pytest -q evals
```

The initial eval verifies that a fixed-window proposal can be manually reviewed
and published as a traceable immutable revision while source MP4 and telemetry
hashes remain unchanged. It also prevents planned model-backed providers from
being reported as implemented or returning placeholder candidates.
