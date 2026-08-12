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
