# Source and license record

## HumanEgo adaptation

- Source repository: `https://github.com/TX-Leo/HumanEgo`
- Source commit: `18fb1082abb87b79f88e53f2abb5bfb9f61de19b`
- Source file: `preprocess/HaMeRHands.py`
- License: PolyForm Noncommercial 1.0.0
- Required notice: Copyright (c) 2026 The HumanEgo Authors -
  https://humanego-ai.github.io

`humanego_hamer.py` retains HumanEgo's YOLO/easy_ViTPose detector, MediaPipe
fallback, HaMeR crop inference, confidence calculation, 21-joint remapping,
physical-size depth recovery, grasp ratio, and joint-angle definitions.

The EgoGlass adaptation removes HumanEgo's Project Aria/MPS file layout,
`AriaCam`, CLI, plotting, per-frame JSON writer, and world-coordinate temporal
optimizer. It accepts `PreparedFrameBundle` directly and returns immutable,
typed camera-coordinate results. World transforms and world-space smoothing
must wait for a real VIO `T_camera_world`; an identity transform is not used as
fake pose data.

Observed false-positive paths are corrected at the boundary: degenerate
ViTPose boxes are filtered by relative geometry, and
neither egocentric fallback nor physical depth recovery raises a low detector
confidence to `0.5`. Neither change modifies ViTPose or HaMeR model inference.
The default final confidence threshold is `0.5`, selected from the committed
model eval's public true-hand and local Glass3 no-hand samples.

The full HumanEgo license is in `HUMANEGO_LICENSE.txt` beside this file.

## Runtime dependencies

| Component | Pinned revision | License status |
| --- | --- | --- |
| HaMeR | `3a01849f4148352e9260b69bf28b65d1671a4905` | MIT |
| easy_ViTPose | `bb9860359e55b099a507c8000e360d48a27cc36d` | Apache-2.0 |
| MediaPipe | installed package | Apache-2.0 |
| WiLoR-mini | `ebec42f94c389070cdd7dda6fd1bf0b4a659c960` | no repository license detected |
| MANO model | WiLoR-mini Hugging Face artifact | restricted model terms; do not redistribute |

Model weights are downloaded to `local-data/models/hand-tracking`, are never
committed, and are recorded in a local manifest with upstream revision, size,
and SHA256. This integration is intended only for the project's declared
noncommercial personal research use.
