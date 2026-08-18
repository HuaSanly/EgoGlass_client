# EgoGlass Recording Client

The Windows client discovers a Rokid Glass3 Enterprise device, terminates its
WebRTC connection, previews the live video and raw IMU streams, and writes one
independent directory per recording. It intentionally contains no perception,
SLAM/VIO, annotation, dataset, or offline-processing runtime.

## Layout

```text
src/schemas/                 # Camera, IMU, calibration, status, and library contracts
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
runtime thread. Per-frame RGB conversion and H.264 encoding, MP4 finalization,
CSV writing, and replay decoding run outside the WebRTC event loop so recording
cannot stall IMU DataChannel consumption.

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
  video.mp4
  camera.csv
  imu.csv
  calibration.yaml
```

The writer first uses `.recording-<recording_id>.partial/`. It validates the
H.264 stream, both CSV files, frame counts, monotonic clocks, IMU coverage, and
calibration resolution before atomically publishing the final directory. IMU
rows start at countdown start and stop at the final encoded frame. RTP, MP4 PTS,
client receive clocks, and other matching evidence never enter the published
four-file contract.

If both IMU streams stop before the final video frame, finalization trims the MP4
and `camera.csv` to their last common covered frame before validation. The ready
status reports the number of removed frames. Full camera metadata is staged
before this recovery step so a later failure remains diagnosable.

The library can replay `video.mp4` with an IMU cursor driven by the shared
Android monotonic clock in `camera.csv.device_monotonic_ns` and
`imu.csv.timestamp_ns`. MP4 PTS are indexed in memory for playback only.

Inspect a completed recording independently:

```powershell
conda run -n egoglass python scripts\inspect-recording.py `
  local-data\recordings\<recording-id>
```

The command fails on an invalid layout, CSV row, frame count, calibration, IMU
coverage, or MP4 stream.

See the [recording protocol](docs/data-recording-protocol.md),
[native UI](docs/native-ui.md), and [ingest gateway](docs/ingest-gateway.md).

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
