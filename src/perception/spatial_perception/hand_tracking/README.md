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
```text
校正后的 BGR 图像帧
→ 使用 YOLO + easy_ViTPose 检测全身手部边界框
→ 若 ViTPose 无法初始化，则回退至 MediaPipe 检测器
→ 对每个裁剪区域进行 HaMeR + MANO 三维手部重建
→ 通过 HumanEgo 恢复深度信息，并重映射为与 Aria 兼容的 21 关节关键点
→ 最终在矫正后的相机坐标系下生成不可变的 HandTrackingResult（手部跟踪结果）
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
