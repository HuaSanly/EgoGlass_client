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
tracking modules, then runs the offline Basalt VIO stage for the same selected
clip. Offline work owns the GPU while it runs; live video preview and recording
remain active. The runtime drains in-flight live inference and releases that
model before the offline runner starts, then releases the offline hand-tracking
model before Basalt starts. The processing job is complete only after both
stages finish.

Results are indexed by `clip_id`, `frame_index`, and `session_time_ns`.
Playback reads the original MP4 and applies these structured results at display
time instead of creating an annotated video unless the operator requests an
export.

The queue schema records preparing/running start time and terminal finish time.
Existing schema-v1 databases migrate in place without deleting job history.
Only one job can own the GPU. The runtime disables optional live inference for
that interval while keeping WebRTC preview and recording active.

The native workbench opens a whole session or one clip, shares one typed
`PlaybackFrame` between the 4:3 video and OpenGL spatial views, and queries
recorded IMU plus one or two immutable result runs at the frame's
`session_time_ns`. See `docs/video-processing.md` for the operator workflow.
