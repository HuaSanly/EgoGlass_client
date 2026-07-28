# Hand tracking

This stage consumes `PreparedFrameBundle` from sensor preprocessing and follows
HumanEgo's hand pipeline:

```text
rectified BGR frame
  -> YOLO + easy_ViTPose whole-body hand boxes
  -> MediaPipe detector fallback when ViTPose cannot initialize
  -> HaMeR MANO reconstruction per crop
  -> HumanEgo depth recovery and Aria-compatible 21-joint remap
  -> immutable HandTrackingResult in rectified camera coordinates
```

`HumanEgoHandTrackingPipeline.process_frame()` is the common offline and live
entry point. `process_frames()` accepts any iterable, so replay and gateway
paths do not need separate inference implementations.

The default config requires CUDA and a loaded HaMeR checkpoint. A result records
the requested and actual device, whether HaMeR loaded, the per-hand backend,
and inference duration. These fields make an accidental MediaPipe-only run
visible in logs and stored output.

World-coordinate results are intentionally absent until VIO provides the same
frame's `T_camera_world`. See `SOURCE.md` for copied-code provenance and license
details.
