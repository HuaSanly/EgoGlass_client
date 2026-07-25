# Interaction Processing Evals

The scaffold eval enforces contract-only communication by rejecting private
imports from spatial-perception and dataset-builder packages. Later evals will
measure phase, segmentation, tracking, pose, inpainting, and trajectory quality
on reviewed clips.

Run from this service directory:

```powershell
uv run pytest -q evals
```
