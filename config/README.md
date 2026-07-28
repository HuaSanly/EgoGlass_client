# Client Configuration

This directory holds shared client configuration. Runtime values belong in
named files such as `capture.toml`, `spatial-perception.toml`,
`interaction-processing.toml`, and `dataset-builder.toml` only after their
contracts and supported parameters are defined.

`sensor-preprocessing.yaml` is the sensor-preprocessing runtime configuration.
It selects the calibration JSON and controls the common recorded, image, and
live-path settings. Relative file paths are resolved from the YAML directory.

`sensor-calibration.sample.json` exercises the calibration interface. Its zero
distortion, guessed intrinsics, IMU noise values, and identity Camera-to-IMU
transform are not measured Glass3 calibration. The pipeline loads whichever
calibration file `sensor-preprocessing.yaml` selects, without a separate
classification or opt-in flag.

`hand-tracking.yaml` selects the native Windows CUDA device, detector fallback,
confidence/geometry thresholds, ignored model directory, and exact upstream
code and model revisions for the HumanEgo-compatible HaMeR pipeline.
