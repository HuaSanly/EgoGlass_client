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
  operator_console/          # Native UI, annotation, and annotation persistence
  perception/                # Independently reusable perception core
    sensor_preprocessing/    # Sensor time, calibration binding, and preparation
    spatial_perception/      # Planned VIO, hands, and coordinate fusion
config/                      # Shared client configuration
tests/                       # Fast tests, named with package prefixes
evals/                       # Periodic evaluations and device evidence
scripts/                     # Client launch, lifecycle, build, and inspection
packaging/                   # Windows executable packaging assets
docs/                        # Component-specific operation notes
```

The `perception` package isolates the research and runtime core from the ingest
gateway and operator console. It contains no copied HumanEgo or other
third-party algorithm code yet.

Detailed notes are available in [ingest gateway](docs/ingest-gateway.md),
[operator console](docs/operator-console.md),
[sensor preprocessing](docs/sensor-preprocessing.md),
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

The command starts the LAN ingest gateway, enables Glass3 discovery, and opens
the native Windows operator console. The annotation page uses the console's
same-origin API. Closing the desktop window or pressing `Ctrl+C` stops both
process trees and releases ports `8770` and `8771`.

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
