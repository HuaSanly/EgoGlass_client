# Recording Client Configuration

The recording-only client reads `client-runtime.yaml` for gateway binding,
discovery, and the recording root. It also copies
`rokid-glass3-calibration.yaml` into every completed recording unit as the
published `calibration.yaml`. These settings take effect after restarting the
client.

Relative paths are resolved from the configuration file directory. Rokid
camera and IMU capture properties remain device-owned and are reported as
observed input capabilities; this client does not invent or override them.

Algorithm, VIO, perception, and offline-processing configuration files are
intentionally absent from this branch.
