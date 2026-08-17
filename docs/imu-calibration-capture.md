# IMU calibration capture

Run the dedicated headless capture process while the glasses are connected:

```powershell
.\scripts\start-imu-calibration.ps1 -DurationHours 3
```

Use `-UntilInterrupted` for a manual run. The process waits for both
accelerometer and gyroscope capabilities, stops the video stream once, and
starts the capture at the first valid IMU sample. It does not start Qt or the
normal recording workflow.

Successful captures are published atomically under
`local-data/imu-calibration/<capture_id>/imu.csv`. The file has exactly the
following columns and contains the raw device timestamp and values:

```text
sensor_type,sequence,timestamp_ns,x,y,z
```

Sequence gaps are retained and reported. Duplicate or out-of-order samples,
non-finite values, disconnects, channel closure, queue overflow, and write
errors fail the capture and remove its partial directory. `Ctrl+C` during a
healthy capture publishes the data; interrupting before the first samples
returns exit code 130.
