from __future__ import annotations

import argparse
import json
import statistics
import time

import dearpygui.dearpygui as dpg
import numpy as np


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def run_benchmark(width: int, height: int, frame_count: int) -> dict[str, float]:
    image = np.random.default_rng(7).integers(
        0,
        256,
        size=(height, width, 3),
        dtype=np.uint8,
    )
    textures = (
        np.empty((height, width, 3), dtype=np.float32),
        np.empty((height, width, 3), dtype=np.float32),
    )
    upload_ms: list[float] = []
    frame_ms: list[float] = []
    dpg.create_context()
    try:
        with dpg.texture_registry(show=False):
            for index, texture in enumerate(textures):
                dpg.add_raw_texture(
                    width=width,
                    height=height,
                    default_value=texture.ravel(),
                    format=dpg.mvFormat_Float_rgb,
                    tag=f"benchmark-texture-{index}",
                )
        with dpg.window(tag="benchmark-window"):
            dpg.add_image(
                "benchmark-texture-0",
                width=width,
                height=height,
                tag="benchmark-image",
            )
        dpg.create_viewport(
            title="EgoGlass texture benchmark",
            width=min(width + 32, 1320),
            height=min(height + 72, 800),
            vsync=False,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("benchmark-window", True)
        front_texture_index = 0
        for frame_index in range(frame_count):
            started_at_ns = time.perf_counter_ns()
            back_texture_index = 1 - front_texture_index
            np.multiply(
                image,
                np.float32(1.0 / 255.0),
                out=textures[back_texture_index],
                casting="unsafe",
            )
            dpg.configure_item(
                "benchmark-image",
                texture_tag=f"benchmark-texture-{back_texture_index}",
            )
            front_texture_index = back_texture_index
            upload_finished_at_ns = time.perf_counter_ns()
            dpg.render_dearpygui_frame()
            frame_finished_at_ns = time.perf_counter_ns()
            if frame_index >= 10:
                upload_ms.append((upload_finished_at_ns - started_at_ns) / 1_000_000)
                frame_ms.append((frame_finished_at_ns - started_at_ns) / 1_000_000)
    finally:
        dpg.destroy_context()
    return {
        "width": float(width),
        "height": float(height),
        "frames": float(len(frame_ms)),
        "upload_mean_ms": statistics.fmean(upload_ms),
        "upload_p95_ms": percentile(upload_ms, 0.95),
        "frame_mean_ms": statistics.fmean(frame_ms),
        "frame_p95_ms": percentile(frame_ms, 0.95),
        "effective_fps": 1000.0 / statistics.fmean(frame_ms),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Dear PyGui RGB texture updates")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=310)
    args = parser.parse_args()
    if args.width < 1 or args.height < 1 or args.frames < 20:
        parser.error("width, height, and frame count must be positive; use at least 20 frames")
    report = run_benchmark(args.width, args.height, args.frames)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
