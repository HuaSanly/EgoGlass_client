# IMU calibration capture service

This package owns the headless Glass3 IMU noise capture workflow. It reuses the
gateway discovery, signaling, stream-control, and typed telemetry contracts but
does not create Qt widgets, video recordings, or wearer recording controls.

## Lifecycle

1. Wait for a Glass3 WebRTC connection and both sensor capabilities.
2. Stop the video stream through `stream-control-v1`.
3. Start at the first valid IMU sample.
4. Batch raw accel/gyro rows through a bounded writer queue.
5. Flush, fsync, validate both streams, and atomically publish `imu.csv`.

Connection replacement, channel closure, malformed telemetry, non-monotonic
samples, inactivity, backpressure, and disk errors fail the capture and remove
the owned partial directory. Configuration is supplied exclusively through the
CLI documented in `docs/imu-calibration-capture.md`.
