# Spatial Perception Evals

The scaffold eval enforces service independence by rejecting imports from the
interaction-processing and dataset-builder private Python packages. Later
device evals will measure calibration quality, VIO tracking coverage, drift,
hand coverage, timing uncertainty, and failure reporting.

Run from this service directory:

```powershell
uv run pytest -q evals
```
