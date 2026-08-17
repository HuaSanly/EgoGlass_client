# Recording Client Evals

Periodic evaluations measure real-device capture quality and recording-only
runtime cost. Run them from the client root:

```powershell
conda run -n egoglass python -m pytest -q evals
```

The device lane records frame metadata coverage, IMU sample rate, sequence
loss, timestamp residual, repeated Start/Stop isolation, startup time, memory,
and imported modules. Synthetic evaluations remain deterministic and do not
contact a Glass3.

`test_wearer_recording_control.py` exercises repeated wearer start/stop commands
through the same coordinator used by the Qt console and asserts one recording
transition per unique command ID.

`test_imu_calibration_capture.py` accelerates a three-hour 100 Hz accel/gyro
capture into 2.16 million rows and verifies bounded queue use, exact row counts,
sample rates, and atomic single-file publication.
