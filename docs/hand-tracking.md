# Hand Tracking

## Boundary

`HumanEgoHandTrackingPipeline` consumes the read-only, rectified BGR image and
camera calibration in `PreparedFrameBundle`. The same `process_frame()` method
is used by recorded replay and live gateway paths.

The output is `HandTrackingResult` contract version `1.0`. Each hand contains:

- wearer-relative left/right handedness;
- detector and reconstruction confidence;
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
  --config config\hand-tracking.yaml
```

The setup follows HumanEgo's Python 3.11, PyTorch 2.5.1, torchvision 0.20.1,
and CUDA 12.1 versions. It installs HaMeR and easy_ViTPose at fixed Git commits
and applies HumanEgo's Chumpy/HaMeR compatibility patches. It does
not run `setup.sh`, import a HumanEgo package, or add the reference repository to
`PYTHONPATH`.

Weights are downloaded by EgoGlass code to
`local-data/models/hand-tracking/`. The directory is ignored by Git. A local
`model-manifest.json` records every source revision, byte size, and SHA256.

## Runtime

The ingest gateway submits each decoded `av.VideoFrame` to one enqueue-only
perception sink. A single worker serializes CUDA access. Its pending buffer
holds only the newest frame, so HaMeR latency cannot delay the WebRTC receive
loop. Live receipt time is explicitly reported as estimated until device clock
alignment is available.

The operator console reads these loopback-only endpoints:

- `GET /api/v1/perception/hand-tracking/status`
- `POST /api/v1/perception/hand-tracking/replays`
- `GET /api/v1/perception/hand-tracking/replays/{session_id}/{run_id}/{clip_id}`

Live visualization uses the gateway's decoded MJPEG preview as the image layer
and draws the structured hand-tracking result on a browser canvas. The model no
longer encodes a separate live preview image.

Offline replay renders an annotated MP4 for every clip and one `results.jsonl`
under `<session>/perception/hand-tracking/<run-id>/`. It never overwrites raw
recordings or earlier runs. By default every fifth frame runs inference and
intermediate frames reuse the latest same-clip skeleton.

The equivalent command-line replay is:

```powershell
conda run -n egoglass python -m perception.main hand-replay `
  --session-directory local-data\recordings\<session-id> `
  --output-directory local-data\recordings\<session-id>\perception\hand-tracking\manual-run
```

```python
from perception.spatial_perception.hand_tracking import (
    HumanEgoHandTrackingPipeline,
)

hands = HumanEgoHandTrackingPipeline.from_config_file(
    "config/hand-tracking.yaml"
)

for bundle in sensor_pipeline.iter_recorded_session(session_directory):
    result = hands.process_frame(bundle)
```

Startup fails when `require_cuda` or `require_hamer` is true and the requested
backend cannot load. Per-frame logs include detector name, `hamer_loaded`, HaMeR
and fallback hand counts, and inference duration.

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
