# Client Configuration

`ui.configuration.ConfigurationService` is the single writer for these
files. It validates all edited modules before saving, replaces each YAML file
atomically, and retains the previous contents as `<name>.yaml.bak`. The native
UI must use that service instead of parsing or constructing YAML directly.

The settings page exposes five modules backed by six managed YAML files:

- `client-runtime.yaml`: gateway binding, discovery, and recordings root.
- `sensor-preprocessing.yaml`: calibration selection and frame/IMU preparation.
- `perception-runtime.yaml` plus `live-hand-tracking.yaml`: live scheduling and
  the low-latency MediaPipe-first hand-tracking profile.
- `offline-hand-tracking.yaml`: quality-first ViTPose-H + HaMeR processing on
  CUDA in FP32.
- `video-processing.yaml`: defaults captured by newly submitted offline jobs.

Saved fields carry one of four application levels: `immediate`, `next_session`,
`next_task`, or `restart_client`. Saving does not interrupt an active recording,
tracker inference, or offline GPU job. The runtime applies immediate changes and
reports the remaining levels to the operator.

`sensor-preprocessing.yaml` is the sensor-preprocessing runtime configuration.
It selects the calibration JSON and controls the common recorded, image, and
live-path settings. Relative file paths are resolved from the YAML directory.

`sensor-calibration-640x480-sample.json` is the active horizontal 4:3 integration
profile. `sensor-calibration.sample.json` retains the previous 1280x720 profile
for controlled comparisons. Both assume an upright decoded frame and contain
unmeasured intrinsics, distortion, IMU noise, and Camera-to-IMU values. The
pipeline loads whichever calibration file `sensor-preprocessing.yaml` selects.

`live-hand-tracking.yaml` and `offline-hand-tracking.yaml` independently select
their thresholds, model directory, and pinned upstream revisions. The offline
quality policy fixes ViTPose-H, HaMeR, CUDA, FP32, and disables detector and
reconstruction fallback. Its remaining thresholds take effect only for newly
submitted tasks.

If an installation contains only the former `hand-tracking.yaml`, the
configuration service copies it once into `live-hand-tracking.yaml`, creates the
offline quality profile, and records `hand_tracking_profiles_v2` in
`configuration-state.json`. Runtime code never reads the legacy path after the
migration.

On first use, valid `auto_enqueue` and `default_preset_id` metadata from
`<recordings-root>/.processing/jobs.sqlite3` is migrated to
`video-processing.yaml`. A migration marker prevents old values from overwriting
later edits. New saves mirror those two values back to the legacy metadata while
the existing processing service still reads it.

Each successful save increments a revision in
`<recordings-root>/.processing/configuration-state.json`. Call
`ConfigurationService.provenance()` when an offline run is submitted to capture
that revision and the SHA256 of all six YAML files in its `run.json`. The queue
also stores the validated sensor calibration and offline perception values captured at
submission, so later edits cannot change an already queued task.

`basalt-vio.yaml` controls the optional offline VIO adapter. It is separate from
hand tracking and is not consumed by the Qt runtime yet. The adapter exports a
prepared session to EuRoC, runs `basalt_vio`, and stores the parsed trajectory
alongside the run. Keep `allow_unverified_calibration: false` for real work;
the checked-in calibration profiles are integration samples only.
