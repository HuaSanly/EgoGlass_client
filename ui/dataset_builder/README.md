# Dataset builder

`DatasetBuilder` converts completed offline processing runs into immutable,
traceable dataset versions. It never edits or copies a source MP4 during normal
publication. Episodes hold frame and `session_time_ns` ranges that refer to the
capture-session media SHA256.

Publication requires all hard gates to pass:

- complete capture session with valid media hashes and clock evidence;
- completed processing run and readable `results.sqlite`;
- completed Basalt run using verified calibration;
- complete world-coordinate coverage;
- a frozen object task profile and valid object triangulation.
- a published annotation revision that fully covers every virtual episode.

Soft gates, such as low hand coverage, excessive interpolation, or duplicate
left/right detections, block publication until an operator records a reasoned
override. The override stores operator, time, and reason in the quality report.

The output is written under `<recordings-root>/datasets/<dataset-id>/`. A
dataset ID is immutable: publishing to an existing directory is rejected. The
builder writes a same-volume staging directory and atomically renames it only
after all artifacts and hashes have been produced.

```text
manifest.json
episodes.jsonl
samples.jsonl
splits.json
quality-report.json
provenance.json
```

Splits are deterministic and grouped by `session_id`, so frames from one
capture session cannot leak across train, validation, and test.

`samples.jsonl` stores the per-frame hand/object values plus stable references
to `results.sqlite#frame_results` and `objects/object-result.json`.
`provenance.json` hashes the selected annotation revision, source media, run
manifests, model/configuration snapshots, masks, point tracks, triangulation,
and QA artifacts.
