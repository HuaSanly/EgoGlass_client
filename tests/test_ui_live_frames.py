from __future__ import annotations

import asyncio
import threading

import numpy as np
from av import VideoFrame

from ingest_gateway.app import create_app
from ingest_gateway.live_frames import LiveFrameBuffer


def _frame(value: int) -> VideoFrame:
    image = np.full((2, 3, 3), value, dtype=np.uint8)
    return VideoFrame.from_ndarray(image, format="rgb24")


def test_live_frame_buffer_publishes_read_only_contiguous_rgb() -> None:
    async def run() -> None:
        runtime = LiveFrameBuffer()
        try:
            await runtime.submit_gateway_frame(
                session_id="session",
                connection_session_id="connection",
                frame_index=7,
                received_at_client_monotonic_ns=100,
                decoded_frame=_frame(23),
            )
            for _ in range(100):
                latest = runtime.latest()
                if latest is not None:
                    break
                await asyncio.sleep(0.001)
            assert latest is not None
            assert latest.frame_index == 7
            assert latest.image_rgb.shape == (2, 3, 3)
            assert latest.image_rgb.dtype == np.uint8
            assert latest.image_rgb.flags.c_contiguous
            assert not latest.image_rgb.flags.writeable
            assert np.all(latest.image_rgb == 23)
        finally:
            await runtime.close()

    asyncio.run(run())


def test_live_frame_buffer_overwrites_pending_work_without_blocking_submitter() -> None:
    release = threading.Event()

    def convert(frame: VideoFrame) -> np.ndarray:
        release.wait()
        return frame.to_ndarray(format="rgb24")

    async def run() -> None:
        runtime = LiveFrameBuffer(converter=convert)
        try:
            await runtime.submit_gateway_frame(
                session_id="session",
                connection_session_id="connection",
                frame_index=0,
                received_at_client_monotonic_ns=1,
                decoded_frame=_frame(0),
            )
            await asyncio.sleep(0.01)
            for frame_index in range(1, 5):
                await runtime.submit_gateway_frame(
                    session_id="session",
                    connection_session_id="connection",
                    frame_index=frame_index,
                    received_at_client_monotonic_ns=frame_index + 1,
                    decoded_frame=_frame(frame_index),
                )
            assert runtime.status().pending_frames_overwritten == 3
            release.set()
            for _ in range(100):
                latest = runtime.latest()
                if latest is not None and latest.frame_index == 4:
                    break
                await asyncio.sleep(0.001)
            assert latest is not None
            assert latest.frame_index == 4
            assert runtime.status().frames_converted == 2
        finally:
            release.set()
            await runtime.close()

    asyncio.run(run())


def test_live_frame_buffer_forwards_the_single_converted_rgb_frame_to_perception() -> None:
    class RgbSink:
        def __init__(self) -> None:
            self.frames: list[dict[str, object]] = []

        async def submit_rgb_frame(self, **frame: object) -> None:
            self.frames.append(frame)

    async def run() -> None:
        sink = RgbSink()
        runtime = LiveFrameBuffer(rgb_frame_sink=sink)
        try:
            await runtime.submit_gateway_frame(
                session_id="session",
                connection_session_id="connection",
                frame_index=8,
                received_at_client_monotonic_ns=100,
                decoded_frame=_frame(37),
            )
            for _ in range(100):
                if sink.frames:
                    break
                await asyncio.sleep(0.001)

            latest = runtime.latest()
            assert latest is not None
            assert len(sink.frames) == 1
            assert sink.frames[0]["image_rgb"] is latest.image_rgb
            assert sink.frames[0]["frame_index"] == 8
            assert not latest.image_rgb.flags.writeable
            assert runtime.status().rgb_frames_forwarded == 1
            assert runtime.status().rgb_sink_failures == 0
        finally:
            await runtime.close()

    asyncio.run(run())


def test_unified_app_routes_perception_through_canonical_rgb_buffer() -> None:
    class WebRtcWiring:
        def __init__(self) -> None:
            self.perception_sink: object = "unset"
            self.display_sink: object | None = None

        def set_capture_telemetry_sink(self, _sink: object) -> None:
            return None

        def set_perception_live_frame_sink(self, sink: object | None) -> None:
            self.perception_sink = sink

        def set_display_frame_sink(self, sink: object) -> None:
            self.display_sink = sink

    class PerceptionSink:
        async def submit_rgb_frame(self, **_frame: object) -> None:
            return None

    webrtc = WebRtcWiring()
    perception = PerceptionSink()
    frame_buffer = LiveFrameBuffer()
    try:
        create_app(
            webrtc_runtime=webrtc,  # type: ignore[arg-type]
            recording_runtime=object(),  # type: ignore[arg-type]
            perception_runtime=perception,  # type: ignore[arg-type]
            live_frame_buffer=frame_buffer,
        )

        assert webrtc.perception_sink is None
        assert webrtc.display_sink is frame_buffer
        assert frame_buffer._rgb_frame_sink is perception
    finally:
        asyncio.run(frame_buffer.close())
