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
    spatial_perception/      # HaMeR hands; planned VIO and coordinate fusion
config/                      # Shared client configuration
tests/                       # Fast tests, named with package prefixes
evals/                       # Periodic evaluations and device evidence
scripts/                     # Client launch, lifecycle, build, and inspection
packaging/                   # Windows executable packaging assets
docs/                        # Component-specific operation notes
```

The `perception` package isolates the research and runtime core from the ingest
gateway and operator console. Hand tracking adapts HumanEgo's
ViTPose/MediaPipe + HaMeR pipeline under its noncommercial license; source and
license records live beside the adapted module.

Detailed notes are available in [ingest gateway](docs/ingest-gateway.md),
[operator console](docs/operator-console.md),
[sensor preprocessing](docs/sensor-preprocessing.md),
[hand tracking](docs/hand-tracking.md),
[spatial perception](docs/spatial-perception.md),
[interaction processing](docs/interaction-processing.md), and
[dataset builder](docs/dataset-builder.md).

## Setup

Run once from this directory:

```powershell
.\scripts\setup_client.ps1
```

This creates the only client environment: native Windows Conda environment
`egoglass`, with Python 3.11, PyTorch 2.5.1, and CUDA 12.1.

```powershell
conda run -n egoglass python scripts\download_hand_tracking_models.py
```

Neither command imports or executes `reference_code/HumanEgo`.

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
conda run -n egoglass python -m pytest
conda run -n egoglass python -m pytest -q evals
conda run -n egoglass ruff check src tests evals
```

Run the CUDA model eval from the same client environment:

```powershell
$env:EGOGLASS_RUN_HAND_MODEL_EVAL = "1"
conda run -n egoglass python -m pytest -q -s evals\test_hand_tracking_model.py
```
