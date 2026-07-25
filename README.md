# EgoGlass Client

The client is one Python workspace for Glass3 ingest, the Windows operator
console, annotation, spatial perception, interaction processing, and dataset
assembly. It has one dependency declaration, one lock file, and one virtual
environment. Code is separated by package responsibility, not by independent
service scaffolds.

## Layout

```text
src/
  ingest_gateway/            # WebRTC ingress, recording, and raw sessions
  operator_console/          # Native Windows operator UI
  data_platform/             # Episode annotation and immutable revisions
  spatial_perception/        # Calibration, VIO, hands, and time association
  interaction_processing/    # Phases, objects, keypoints, and trajectories
  dataset_builder/           # Training samples, splits, and provenance
config/                      # Shared client configuration
tests/                       # Fast tests, named with package prefixes
evals/                       # Periodic evaluations and device evidence
scripts/                     # Client launch, lifecycle, build, and inspection
packaging/                   # Windows executable packaging assets
docs/                        # Component-specific operation notes
```

The perception, interaction, and dataset packages are boundaries only. They
contain no copied HumanEgo or other third-party algorithm code yet.

Detailed notes are available in [ingest gateway](docs/ingest-gateway.md),
[operator console](docs/operator-console.md), [data platform](docs/data-platform.md),
[spatial perception](docs/spatial-perception.md),
[interaction processing](docs/interaction-processing.md), and
[dataset builder](docs/dataset-builder.md).

## Setup

Run once from this directory:

```powershell
uv sync --group dev
```

This creates the only workspace environment at `.venv/`.

## Windows Client

Start the complete client from this directory:

```powershell
.\scripts\start-client.ps1
```

The command starts the LAN ingest gateway and local data platform, enables
Glass3 discovery, and opens the native Windows operator console. Closing the
desktop window or pressing `Ctrl+C` stops all three process trees and releases
ports `8770`, `8771`, and `8780`.

Completed recordings live under `local-data/recordings/`. The entire
`local-data/` tree is ignored by Git. Annotation writes only beneath each
session's ignored `annotations/` directory and never changes source MP4 or
telemetry files.

Build the local Windows executable with:

```powershell
.\scripts\build-desktop.ps1
```

The ignored output is `dist/EgoGlass/EgoGlass.exe`.

## Verification

```powershell
uv run pytest
uv run pytest -q evals
uv run ruff check src tests evals
```
