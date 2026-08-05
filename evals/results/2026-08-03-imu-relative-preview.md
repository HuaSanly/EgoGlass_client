# IMU relative preview device eval - 2026-08-03

## Scope

- Device: Rokid Glass3 Enterprise connected over the live WebRTC DataChannel.
- Requested Android sampling period: 10,000 microseconds for both sensors.
- Client: native PyQt home view with the relative Madgwick preview fix.
- Device state: stationary during the measurement window.

## Pass criteria

- The UI reports the gyroscope pose-update cadence near 100 Hz instead of the
  combined accelerometer and gyroscope message rate near 200 Hz.
- Stationary Pitch changes by no more than 1 degree over 30 seconds.
- Refresh immediately makes the current fused orientation the relative origin
  without restarting the Madgwick filter.

## Result

PASS.

- Accelerometer arrival rate: 100.888 Hz.
- Gyroscope arrival rate: 100.835 Hz.
- UI rate across 31 one-second observations: 100.9 Hz for every observation.
- Stationary Pitch across 30 seconds: 32.9 to 32.7 degrees, a 0.2-degree
  change before applying a new relative reference.
- The UI Automation invocation of Refresh changed Pitch from 32.7 degrees to
  -0.0 degrees immediately; the next observation was 0.1 degrees.
- The client was closed after the test, and TCP 8770 and UDP 8771 were free.

The large pre-refresh angle was the current relative pose after the glasses had
been moved, not filter drift. The deterministic 30-second tilted-sensor eval in
`evals/test_imu_preview_stability.py` covers continued stability after a reset.
