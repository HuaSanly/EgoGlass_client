# Dataset Builder

`dataset-builder` will assemble immutable, versioned training datasets from raw
capture references, compatible derived artifacts, and reviewed annotations. It
owns sample construction, coordinate conversion, schema validation, dataset
splits, provenance, and export. It does not run perception algorithms or mutate
source sessions.

This package currently defines the dataset-builder boundary only. No HumanEgo
or other third-party source has been copied into the package.

## Directory purpose

| Path | Purpose |
| --- | --- |
| `src/perception/dataset_builder/` | Planned dataset-assembly package. |
| `tests/` | Shared fast tests, identified by `test_dataset_builder_` filenames. |
| `evals/` | Shared periodic evaluations, identified by `test_dataset_builder_` filenames. |
| `config/` | Shared versioned client configuration. |
| `pyproject.toml` | Workspace package, tool, and dependency declaration. |

## Module purpose

| Module | Future responsibility |
| --- | --- |
| `pipeline.py` | Orchestrate one reproducible dataset build. |
| `sample_builder.py` | Assemble frame-level and episode-level samples. |
| `coordinate_conversion.py` | Convert versioned perception frames into the training representation. |
| `schema_validation.py` | Validate every emitted sample and cross-reference. |
| `dataset_split.py` | Create deterministic train, validation, and test splits. |
| `provenance.py` | Record source sessions, annotations, code, configuration, and model versions. |
| `export.py` | Publish a new immutable dataset version. |

## Planned boundary

Inputs will be immutable capture references, compatible derived artifacts, and
reviewed annotations. Output will be a separately versioned dataset; source
sessions and previously registered datasets remain read-only.

## Verification

```powershell
conda run -n egoglass python -m pytest
conda run -n egoglass python -m pytest -q evals
conda run -n egoglass ruff check src tests evals
```
