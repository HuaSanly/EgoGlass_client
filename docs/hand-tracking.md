# Hand Tracking

## Boundary

`HumanEgoHandTrackingPipeline` consumes the read-only, rectified BGR image and
camera calibration in `PreparedFrameBundle`. The same `process_frame()` method
is used by persistent offline processing and the optional live gateway path.

The output is `HandTrackingResult` contract version `1.0`. Each hand contains:

- wearer-relative left/right handedness;
- detector confidence, HaMeR reconstruction quality, and the separate depth,
  2D coverage, and 3D compactness quality components;
- final confidence used by filtering, with legacy `confidence` retained as an
  alias of `final_confidence`;
- HumanEgo's Aria-compatible 21-joint order;
- rectified-image pixel coordinates;
- 3D points in the rectified camera coordinate system, in meters;
- model-estimated or physical-size-estimated depth provenance;
- 20 HumanEgo joint angles and the thumb/index grasp ratio;
- the actual reconstruction backend, so MediaPipe fallback cannot be mistaken
  for HaMeR.

No world coordinates are emitted. HumanEgo's `c2w` transform and world-space
temporal optimizer require a same-time VIO pose. They will be connected only
after VIO provides a real `T_camera_world`; the implementation never substitutes
an identity transform.

## Native Windows Environment

From the client repository:

```powershell
.\scripts\setup_client.ps1
conda run -n egoglass python scripts\download_hand_tracking_models.py `
  --config config\live-hand-tracking.yaml `
  --config config\offline-hand-tracking.yaml
```

The setup follows HumanEgo's Python 3.11, PyTorch 2.5.1, torchvision 0.20.1,
and CUDA 12.1 versions. The live profile uses MediaPipe detection, HaMeR
reconstruction, CUDA AMP, and newest-frame scheduling. The offline profile uses
ViTPose-H detection and required HaMeR reconstruction on CUDA in FP32. The
setup applies HumanEgo's Chumpy/HaMeR
compatibility patches. It does not run `setup.sh`, import a HumanEgo package,
or add the reference repository to `PYTHONPATH`.

Weights are downloaded by EgoGlass code to
`local-data/models/hand-tracking/`. The directory is ignored by Git. A local
`model-manifest.json` records every source revision, byte size, and SHA256.

## Runtime

`HandTrackingRuntime` reads `config/live-hand-tracking.yaml`. The ingest gateway
submits each decoded `av.VideoFrame` to one enqueue-only
perception sink. A single worker serializes CUDA access. Its pending buffer
holds only the newest frame, so HaMeR latency cannot delay the WebRTC receive
loop. Live receipt time is explicitly reported as estimated until device clock
alignment is available.

MediaPipe is the normal live 2D detection path. Offline processing reads
`config/offline-hand-tracking.yaml` and runs every frame through ViTPose-H and
HaMeR. CUDA uses FP16 autocast only in the live profile; offline processing
stays in full precision. All detector
crops from one frame are collated into one HaMeR batch and one model forward.
Each result reports preparation, detector, reconstruction, and postprocessing
durations, reconstruction batch size, and whether AMP was active. This version
does not interpolate missing detections, suppress short segments, or smooth
results across frames.

For HaMeR results with model-estimated depth, `final_confidence` is detector
confidence multiplied by `reconstruction_quality`. The reconstruction quality
is derived from `depth_score`, `coverage_score`, and `compactness_score`. When
invalid HaMeR depth is replaced by the physical-size estimate, final confidence
falls back to detector confidence while the original HaMeR quality components
remain visible as diagnostics. A MediaPipe-only reconstruction reports those
four HaMeR-specific values as `null` rather than inventing comparable scores.

External diagnostics can read these loopback-only endpoints:

- `GET /api/v1/perception/hand-tracking/status`
- `GET /api/v1/perception/hand-tracking/events` (SSE status push)

Live visualization reads the in-process decoded RGB buffer and draws structured
hand-tracking results directly on the PyQt `VideoCanvas`. The model does not
encode a separate live preview image.

Offline hand tracking runs through `VideoProcessingService`. Each immutable run
writes `run.json`, `results.sqlite`, and `run.log` under
`<session>/derived/video-processing/<run-id>/`. Presets select inference stride;
intermediate frames may hold the most recent same-clip result only within that
explicit stride. The UI dynamically overlays structured results on original
media. It encodes an annotated H.264 MP4 only after an explicit export.

Before inference, the service derives one strict video/IMU session timeline
from immutable timing evidence. Missing timing evidence fails the job; there is
no video-only or unchecked-clock mode. Raw capture files remain unchanged.

```python
from perception.spatial_perception.hand_tracking import (
    HumanEgoHandTrackingPipeline,
)

hands = HumanEgoHandTrackingPipeline.from_config_file(
    "config/offline-hand-tracking.yaml"
)

for bundle in sensor_pipeline.iter_recorded_session(session_directory):
    result = hands.process_frame(bundle)
```

Startup fails when `require_cuda` or `require_hamer` is true and the requested
backend cannot load. Per-frame logs include detector name, `hamer_loaded`, AMP
state, HaMeR batch size, HaMeR and fallback hand counts, and all four stage
durations.

## Verification

Fast gate tests do not load Torch or model weights:

```powershell
conda run -n egoglass python -m pytest -q `
  tests\test_perception_hand_tracking.py tests\test_perception_runtime.py
```

The opt-in model eval checks a fixed-hash public true-hand image and five
no-hand frames from the local Glass3 recording:

```powershell
$env:EGOGLASS_RUN_HAND_MODEL_EVAL = "1"
conda run -n egoglass python -m pytest -q -s `
  evals\test_hand_tracking_model.py
```

The current measured result on RTX 3060 12 GB is two CUDA HaMeR hands on the
positive image with confidence `0.7738` and `0.7210`, plus `0/5` false-positive
Glass3 frames. These are small initial eval sets, not a final accuracy claim.
