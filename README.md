# EgoGlass Client

The Windows client receives Glass3 video and IMU, records capture sessions,
processes stored video, optionally runs online perception, and provides a native UI.
The runtime uses one Python process and one Conda environment named `egoglass`.

## Layout

```text
src/
  schemas/                  # Public frame, IMU, playback, and result types
  sensor_preprocessing/     # Time mapping, calibration, and prepared inputs
  hand_tracking/            # Live and offline hand-tracking algorithms
  slam_vio/                 # Basalt-backed offline VIO adapter
  process_video.py          # Thin no-UI offline processing entry point
ui/
  application/              # Runtime lifecycle and native client host
  gateway/                  # WebRTC ingress, recording, and raw sessions
  processing/               # UI-backed queue, results, and export
config/                     # Shared client configuration
tests/                      # Deterministic gate tests
evals/                      # Periodic quality evaluations and device evidence
scripts/                    # Setup, launch, benchmark, and inspection
docs/                       # Component operation notes
```

PyQt owns the main thread. WebRTC and Uvicorn use one asyncio thread;
RGB conversion, inference, recording, playback, and IMU orientation use bounded
workers inside that same process. The native UI calls runtime objects directly.
HTTP port `8770` remains available for Glass3 signaling and external diagnostics.

Hand tracking adapts HumanEgo's MediaPipe/ViTPose + HaMeR flow. Source and
license records for adapted model code live beside the hand-tracking module.

See [native UI](docs/native-ui.md), [video processing](docs/video-processing.md),
[ingest gateway](docs/ingest-gateway.md),
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

For a single no-UI offline run, use the thin CLI against a complete capture
session. It does not start the gateway or Qt:

```powershell
conda run -n egoglass python src\process_video.py `
  --session local-data\recordings\<session-id> `
  --output local-data\headless-runs\<run-id>
```

The command writes immutable per-frame inference to `raw_results.jsonl`, the
temporalized camera/world result to `results.jsonl`, and provenance plus stage
metrics to `run.json`. Basalt is attempted before hand processing. If VIO is
unavailable or misses a final hand frame by more than 100 ms, the run is marked
`partial`: camera-space results remain usable and no world coordinate is
invented for the unmatched frame.

Basalt VIO is an optional native dependency. After installing `basalt_vio`,
run it against the same prepared capture session without starting Qt or the
gateway:

```powershell
conda run -n egoglass python src\run_vio.py `
  --session local-data\recordings\<session-id> `
  --output local-data\vio-runs\<run-id>
```

The default `config/basalt-vio.yaml` rejects the repository's unverified sample
calibration. Use `--allow-unverified-calibration` only for integration tests;
real trajectory work requires a measured camera-IMU calibration. The runner
exports an EuRoC `mav0/` dataset, `calibration.json`, Basalt logs, and parses
`trajectory.csv` into typed poses.

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
