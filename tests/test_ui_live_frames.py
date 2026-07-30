from __future__ import annotations

import asyncio
import threading

import numpy as np
from av import VideoFrame

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
