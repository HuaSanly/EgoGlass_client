# Client Configuration

This directory holds shared client configuration. Runtime values belong in
named files such as `capture.toml`, `spatial-perception.toml`,
`interaction-processing.toml`, and `dataset-builder.toml` only after their
contracts and supported parameters are defined.

`sensor-calibration.sample.json` is an interface-only placeholder for the
sensor-preprocessing pipeline. Its zero distortion, guessed intrinsics, IMU
noise values, and identity Camera-to-IMU transform are not measured Glass3
calibration. Runtime code rejects it unless placeholder use is explicitly
enabled.
