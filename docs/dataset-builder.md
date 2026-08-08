# Dataset Builder

`ui.dataset_builder` assembles immutable, versioned training datasets from raw
capture references, compatible derived artifacts, and reviewed annotations. It
owns sample construction, coordinate conversion, schema validation, dataset
splits, provenance, and export. It does not run perception algorithms or mutate
source sessions.

The algorithm stages remain in `src/phase_analysis` and `src/object_tracking`.
The UI-owned builder only performs quality review, virtual episode assembly,
provenance, deterministic session-grouped splitting, and publication.

## Directory purpose

| Path | Purpose |
| --- | --- |
| `ui/dataset_builder/` | Candidate discovery, quality gates, virtual episodes, and publication. |
| `src/schemas/dataset.py` | Stable JSONL and Manifest contracts. |
| `tests/` | Shared fast tests, identified by `test_dataset_builder_` filenames. |
| `evals/` | Shared periodic evaluations, identified by `test_dataset_builder_` filenames. |
| `config/` | Shared versioned client configuration. |
| `pyproject.toml` | Workspace package, tool, and dependency declaration. |

## Module purpose

| Module | Responsibility |
| --- | --- |
| `builder.py` | Assemble and atomically publish Manifest plus JSONL artifacts. |
| `quality.py` | Enforce hard gates and auditable soft-gate overrides. |
| `episodes.py` | Split valid frame runs into virtual episode spans. |
| `catalog.py` | Discover processing runs for the Fluent dataset hall. |
| `models.py` | UI workflow state that does not belong in public schemas. |

## Publication boundary

Inputs are immutable capture references, completed offline runs, verified
Basalt calibration, object-stage artifacts, and reviewed annotations. Output is
a separately versioned dataset. Source sessions and previously published
datasets remain read-only. Video cuts are stored as frame/time spans; MP4 files
are only created by a separate explicit media export.

Every virtual episode must be fully covered by an immutable
`episode-annotation-v1` revision. Published episode labels and phases are copied
into `episodes.jsonl`; every frame keeps explicit hand-result, object-result,
annotation-revision, processing-run, and VIO-run references.

Publication is transactional: all JSONL, quality, split, and provenance files
are written to a same-volume staging directory and moved into the final dataset
ID only after every artifact hash is available. `provenance.json` records source
media hashes, the processing configuration snapshot, frozen object task profile,
model revisions, verified Basalt calibration evidence, the annotation revision
hash, and hashes for all derived object masks and stage artifacts.

## Verification

```powershell
conda run -n egoglass python -m pytest
conda run -n egoglass python -m pytest -q evals
conda run -n egoglass ruff check src tests evals
```
