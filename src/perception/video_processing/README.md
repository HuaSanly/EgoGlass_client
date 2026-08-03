# Video processing service

`VideoProcessingService` owns the persistent offline perception queue. It reads
completed `capture-session-v1` directories without mutating their raw media and
writes one immutable run under:

```text
<session>/derived/video-processing/<run-id>/
  run.json
  results.sqlite
  run.log
```

The queue database lives at `<recordings-root>/.processing/jobs.sqlite3`.
Preparing, running, or canceling jobs become `interrupted` after a process
restart and require an explicit retry. A retry creates a new job so previous
attempts and run artifacts remain inspectable.

The first pipeline reuses the sensor preprocessing and HumanEgo-compatible hand
tracking modules. Offline work owns the GPU while it runs; live video preview
and recording remain active, while optional live inference is paused.

Results are indexed by `clip_id`, `frame_index`, and `session_time_ns`.
Playback reads the original MP4 and applies these structured results at display
time instead of creating an annotated video unless the operator requests an
export.
