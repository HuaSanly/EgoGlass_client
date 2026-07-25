# Dataset Builder Evals

The scaffold eval enforces contract-only communication by rejecting private
imports from spatial-perception and interaction-processing packages. Later
evals will measure usable-sample coverage, invalid references, split leakage,
lineage completeness, and deterministic rebuilds.

Run from this service directory:

```powershell
uv run pytest -q evals
```
