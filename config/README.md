# Client Configuration

This directory holds shared client configuration. Runtime values belong in
named files such as `capture.toml`, `spatial-perception.toml`,
`interaction-processing.toml`, and `dataset-builder.toml` only after their
contracts and supported parameters are defined.

`sensor-preprocessing.yaml` is the sensor-preprocessing runtime configuration.
It selects the calibration JSON and controls the common recorded, image, and
live-path settings. Relative file paths are resolved from the YAML directory.

`sensor-calibration-640x480-sample.json` is the active horizontal 4:3 integration
profile. `sensor-calibration.sample.json` retains the previous 1280x720 profile
for controlled comparisons. Both assume an upright decoded frame and contain
unmeasured intrinsics, distortion, IMU noise, and Camera-to-IMU values. The
pipeline loads whichever calibration file `sensor-preprocessing.yaml` selects.

`hand-tracking.yaml` selects the native Windows CUDA device, detector fallback,
confidence/geometry thresholds, ignored model directory, and exact upstream
code and model revisions for the HumanEgo-compatible HaMeR pipeline.
