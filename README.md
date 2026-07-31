# EgoGlass Client

The Windows client receives Glass3 video and IMU, records capture sessions,
runs online perception, replays stored media, and provides a native operator UI.
The runtime uses one Python process and one Conda environment named `egoglass`.

## Layout

```text
ui/                         # PyQt Fluent application and native home view
src/
  annotation/               # Annotation contracts, editor state, and persistence
  ingest_gateway/           # WebRTC ingress, recording, and raw sessions
  perception/               # Independently reusable research/runtime core
    sensor_preprocessing/   # Time mapping, calibration, and prepared inputs
    spatial_perception/     # Hand tracking; planned VIO and coordinate fusion
config/                     # Shared client configuration
tests/                      # Deterministic gate tests
evals/                      # Periodic quality evaluations and device evidence
scripts/                    # Setup, launch, benchmark, and inspection
docs/                       # Component operation notes
```

PyQt owns the main thread. WebRTC and Uvicorn use one asyncio thread;
RGB conversion, inference, recording, replay, and IMU orientation use bounded
workers inside that same process. The native UI calls runtime objects directly.
HTTP port `8770` remains available for Glass3 signaling and external diagnostics.

Hand tracking adapts HumanEgo's MediaPipe/ViTPose + HaMeR flow. Source and
license records for adapted model code live beside the hand-tracking module.

See [native UI](docs/native-ui.md), [ingest gateway](docs/ingest-gateway.md),
[sensor preprocessing](docs/sensor-preprocessing.md),
[hand tracking](docs/hand-tracking.md), [spatial perception](docs/spatial-perception.md),
[interaction processing](docs/interaction-processing.md), and
[dataset builder](docs/dataset-builder.md).

## Setup

Run once from this directory:

```powershell
.\scripts\setup_client.ps1
```

This creates or updates native Windows Conda environment `egoglass` with
Python 3.11, PyTorch 2.5.1, and CUDA 12.1. Download model artifacts separately:

```powershell
conda run -n egoglass python scripts\download_hand_tracking_models.py
```

Neither command executes code from `reference_code/HumanEgo`.

## Run

```powershell
.\scripts\start-client.ps1
```

This launches one `python -m ui` process. Closing the native window or pressing
`Ctrl+C` stops discovery, signaling, capture, inference, workers, and the UI.
Active capture sessions are finalized before shutdown.

Completed recordings live under ignored path `local-data/recordings/`.
Annotation revisions are written beneath each session's `annotations/`
directory and do not modify source MP4 or telemetry files.

## License

The client source is licensed under GPLv3 because its native UI links PyQt6
and PyQt6-Fluent-Widgets. Dependency license details are recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Verification

```powershell
conda run -n egoglass python -m pytest
conda run -n egoglass python -m pytest -q evals
conda run -n egoglass ruff check src ui tests evals scripts
```

Run the CUDA model eval separately:

```powershell
$env:EGOGLASS_RUN_HAND_MODEL_EVAL = "1"
conda run -n egoglass python -m pytest -q -s evals\test_hand_tracking_model.py
```
