# IMU calibration capture

Run the dedicated headless capture process while the glasses are connected:

```powershell
.\scripts\start-imu-calibration.ps1 -DurationHours 3
```

When exactly one ADB device is connected, the launcher validates that Glass3
and Windows are on the same IPv4 subnet, keeps Glass3 awake while USB-powered,
and restarts the glasses app only after the capture gateway is listening. It
restores the previous Android stay-awake setting on exit. Use `-AdbSerial` when
more than one device is connected, or `-DisableAdbPreparation` to retain manual
discovery and device lifecycle control.

ADB is optional. Without it, keep the glasses app open and connect Glass3 and
Windows to the same Wi-Fi network before starting the command.

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
errors fail the capture and remove its partial directory. Failure to receive a
first sample within five seconds after video shutdown also fails instead of
waiting indefinitely. `Ctrl+C` during a
healthy capture publishes the data; interrupting before the first samples
returns exit code 130.
