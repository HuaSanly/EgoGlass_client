# Configuration service

`ConfigurationService` exposes the native client's six YAML files as five
typed modules. UI code stages mappings in memory and calls `validate()` before
`save()`; it never reads or writes YAML itself.

```python
service = ConfigurationService("config")
service.stage({"live_hand_tracking": {"runtime": {"enabled": True}}})
issues = service.validate()
result = service.save() if not issues else None
```

`save()` returns every changed field with its application level. Pass
`service.apply_request(result)` to the runtime host for immediate application.
`next_session`, `next_task`, and `restart_client` changes remain explicit and do
not terminate active work.

The service validates calibration JSON through `SensorCalibration`, resolves
paths relative to the config directory, writes through a same-directory
temporary file, and keeps the previous YAML as `.bak`. Its provenance snapshot
contains a persistent revision plus the SHA256 of all six YAML files for
offline-run manifests. The video-processing queue also stores the validated
sensor calibration and `offline_hand_tracking` values captured at job
submission, so queued work cannot silently consume later edits.

The live profile is read only by `HandTrackingRuntime`. The offline profile is
read only by `SessionProcessingRunner`. New offline jobs always use the
quality-only stride-1 preset; old queued jobs and manifests retain their stored
preset and legacy `hand_tracking` snapshot field for playback compatibility.
