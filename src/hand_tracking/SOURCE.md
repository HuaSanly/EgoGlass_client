# HumanEgo source provenance

`temporal.py` adapts the sequence cleanup and kinematic optimization design from
HumanEgo commit `18fb1082abb87b79f88e53f2abb5bfb9f61de19b`:

- `preprocess/HaMeRHands.py`
- `preprocess/AriaHandsOptimizer.py`
- `preprocess/AriaHandsTypes.py`

Upstream project: <https://github.com/WesLee88524/HumanEgo>

The adapted implementation keeps confidence filtering, bounded interpolation,
short-segment suppression, grasp smoothing, Savitzky-Golay position smoothing,
EMA orientation smoothing, Gram-Schmidt orthogonalization, and world kinematic
derivation. EgoGlass replaces Project Aria I/O with immutable EgoGlass schemas,
uses actual `session_time_ns` deltas, binds Basalt poses and frozen camera-to-IMU
extrinsics, separates clips and hands, and fixes edge-run grasp inversion.

HumanEgo code is licensed under PolyForm Noncommercial 1.0.0. The complete
terms are in `HUMANEGO-LICENSE.txt`.

Required Notice: Copyright (c) 2026 The HumanEgo Authors -
https://humanego-ai.github.io
