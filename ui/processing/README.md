# Video processing service

`VideoProcessingService` owns the persistent offline perception queue. It reads
completed `capture-session-v1` directories without mutating their raw media and
writes one immutable run under:

```text
<session>/derived/video-processing/<run-id>/
  run.json
  results.sqlite
  phases.jsonl
  run.log
  objects/
    masks/
    selected-keypoints.jsonl
    tracks.json
    triangulation.json
    object-qa.json
```

The queue database lives at `<recordings-root>/.processing/jobs.sqlite3`.
Preparing, running, or canceling jobs become `interrupted` after a process
restart and require an explicit retry. A retry creates a new job so previous
attempts and run artifacts remain inspectable.

The pipeline runs sensor preprocessing, Basalt VIO, HumanEgo-compatible hand
tracking and temporal cleanup, motion phase analysis, DINO-SAM object masks,
CoTracker point tracks, multi-view triangulation, and grasp-latched object pose
propagation. The task profile and its object prompts are frozen when the job is
submitted. The full DINO/SAM2/CoTracker configuration and its SHA256 are frozen
in the same job snapshot, so queued work cannot silently pick up later YAML
changes. Offline work owns the GPU while it runs; live video preview and recording
remain active.

Results are indexed by `clip_id`, `frame_index`, and `session_time_ns`.
Playback reads the original MP4 and applies these structured results at display
time instead of creating an annotated video unless the operator requests an
export.

The workbench can switch between immutable raw per-frame hand inference and the
final temporal result without decoding the source video again. Object masks,
tracked points, 3D point clouds, and object axes remain tied to the same
`clip_id + frame_index + session_time_ns`.

The queue schema records preparing/running start time and terminal finish time.
Existing schema-v1 databases migrate in place without deleting job history.
Only one job can own the GPU. The runtime disables optional live inference for
that interval while keeping WebRTC preview and recording active.

The native workbench opens a whole session or one clip, shares one typed
`PlaybackFrame` between the 4:3 video and OpenGL spatial views, and queries
recorded IMU plus one or two immutable result runs at the frame's
`session_time_ns`. See `docs/video-processing.md` for the operator workflow.
