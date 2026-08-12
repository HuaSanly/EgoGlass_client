# EgoGlass Recording Client

The Windows client discovers a Rokid Glass3 Enterprise device, terminates its
WebRTC connection, previews the live video and raw IMU streams, and writes one
independent directory per recording. It intentionally contains no perception,
SLAM/VIO, annotation, dataset, or offline-processing runtime.

## Layout

```text
src/schemas/                 # Recording manifest, CSV, quality, and API contracts
ui/application/              # Native client lifecycle
ui/gateway/                  # Discovery, WebRTC ingress, recording, and HTTP API
ui/views/                    # Recording console and recording library
ui/widgets/                  # Video, raw-IMU plots, and synchronized playback
config/client-runtime.yaml   # Gateway and recording-root settings
tests/                       # Deterministic recording gate tests
evals/                       # Recording quality and soak evaluations
scripts/                     # Setup, launch, benchmark, and inspection
docs/                        # Operator and gateway documentation
```

PyQt owns the main thread. Uvicorn, aiortc, discovery, and capture run on the
runtime thread. RGB conversion, MP4 writing, CSV writing, and replay decoding
use bounded workers so recording finalization does not stop WebRTC reception.

## Setup

Use the existing Windows Conda environment definition:

```powershell
.\scripts\setup_client.ps1
```

`environment.yml` is intentionally unchanged in this branch so existing
workstations are not pruned. The installable project in `pyproject.toml` has no
Torch, HaMeR, ViTPose, MediaPipe, Basalt, OpenCV, SciPy, or PyOpenGL dependency.

## Run

```powershell
.\scripts\start-client.ps1
```

Open the EgoGlass application on Glass3. The device discovers the client on
UDP port 8771 and connects to the signaling service on TCP port 8770. The
native window then provides live preview, raw accelerometer and gyroscope
plots, recording controls, and the recording library.

One Start-to-Stop operation creates one `recording_id`:

```text
local-data/recordings/<recording_id>/
  manifest.json
  video.mp4
  imu.csv
  frames.csv
  quality.json
  annotations/
  derived/
```

The writer first uses `.recording-<recording_id>.partial/`. It validates the
MP4, both CSV files, counts, and SHA256 values before atomically publishing the
final directory. IMU rows start at countdown start; `inside_video_span` marks
the samples between the first and last saved video frames. Stopping a recording
closes its time window, so later IMU samples cannot enter the completed files.

The library can replay `video.mp4` with an IMU cursor driven by the shared
`recording_time_ns`. Playback is an integrity check only and does not run any
algorithm.

Inspect a completed recording independently:

```powershell
conda run -n egoglass python scripts\inspect-recording.py `
  local-data\recordings\<recording-id>
```

The command fails on an invalid layout, CSV row, count, hash, or MP4 stream.

See [native UI](docs/native-ui.md) and [ingest gateway](docs/ingest-gateway.md).

## Verification

```powershell
conda run -n egoglass python -m pytest
conda run -n egoglass python -m pytest -q evals
conda run -n egoglass ruff check src ui tests evals scripts
git diff --check
```

## License

The client source is GPLv3 because its native UI links PyQt6 and
PyQt6-Fluent-Widgets. Dependency notices are in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
