from __future__ import annotations

import asyncio
import threading
from fractions import Fraction

import numpy as np
from av import VideoFrame

from ui.gateway.live_frames import LiveFrame, LiveFrameBuffer, LiveFramePacer


def _frame(value: int) -> VideoFrame:
    image = np.full((2, 3, 3), value, dtype=np.uint8)
    return VideoFrame.from_ndarray(image, format="rgb24")


def _live_frame(index: int, *, stream: str = "connection") -> LiveFrame:
    image = np.full((2, 3, 3), index, dtype=np.uint8)
    image.setflags(write=False)
    return LiveFrame(
        connection_session_id=stream,
        frame_index=index,
        received_at_client_monotonic_ns=index * 33_333_333,
        converted_at_client_monotonic_ns=index * 33_333_333,
        image_rgb=image,
        video_pts_ns=index * 33_333_333,
    )


def test_live_frame_pacer_turns_a_burst_into_pts_cadence() -> None:
    pacer = LiveFramePacer(
        target_queue_frames=2,
        startup_wait_ns=40_000_000,
    )
    pacer.enqueue(_live_frame(0))
    assert pacer.next_frame(0) is None
    pacer.enqueue(_live_frame(1))

    assert pacer.next_frame(1).frame_index == 0  # type: ignore[union-attr]
    assert pacer.next_frame(20_000_000).frame_index == 0  # type: ignore[union-attr]
    assert pacer.next_frame(34_000_000).frame_index == 1  # type: ignore[union-attr]
    status = pacer.status(34_000_000)
    assert status.interval_ms == 33.333
    assert status.frames_presented == 2
    assert status.recent_fps == 29.412


def test_live_frame_pacer_estimates_cadence_from_bursty_pts_span() -> None:
    pacer = LiveFramePacer()
    bursty_pts_ns = (0, 1_000_000, 66_666_666, 67_666_666, 133_333_332)
    for index, pts_ns in enumerate(bursty_pts_ns):
        frame = _live_frame(index)
        pacer.enqueue(
            LiveFrame(
                connection_session_id=frame.connection_session_id,
                frame_index=frame.frame_index,
                received_at_client_monotonic_ns=frame.received_at_client_monotonic_ns,
                converted_at_client_monotonic_ns=frame.converted_at_client_monotonic_ns,
                image_rgb=frame.image_rgb,
                video_pts_ns=pts_ns,
            )
        )

    status = pacer.status()
    assert status.interval_ms == 33.333
    assert status.source_gap_p95_ms == 65.667
    assert status.source_gap_max_ms == 65.667


def test_live_frame_pacer_bounds_latency_and_discards_oldest_burst_frames() -> None:
    pacer = LiveFramePacer(maximum_queue_frames=4, target_queue_frames=2)
    for index in range(6):
        pacer.enqueue(_live_frame(index))

    assert pacer.status().queue_depth == 4
    assert pacer.status().frames_dropped == 2
    assert pacer.next_frame(200_000_000).frame_index == 3  # type: ignore[union-attr]
    assert pacer.next_frame(234_000_000).frame_index == 4  # type: ignore[union-attr]
    assert pacer.status().queue_depth == 1
    assert pacer.status().frames_dropped == 3


def test_live_frame_pacer_keeps_target_depth_after_presenting_a_pair_burst() -> None:
    pacer = LiveFramePacer(
        target_queue_frames=2,
        startup_wait_ns=40_000_000,
    )
    pacer.enqueue(_live_frame(0))
    pacer.enqueue(_live_frame(1))
    assert pacer.next_frame(1).frame_index == 0  # type: ignore[union-attr]

    pacer.enqueue(_live_frame(2))
    pacer.enqueue(_live_frame(3))
    assert pacer.next_frame(34_000_000).frame_index == 1  # type: ignore[union-attr]
    assert pacer.status(34_000_000).queue_depth == 2
    assert pacer.status(34_000_000).frames_dropped == 0


def test_live_frame_pacer_recovers_immediately_after_one_counted_starvation() -> None:
    pacer = LiveFramePacer(startup_wait_ns=0)
    pacer.enqueue(_live_frame(0))
    assert pacer.next_frame(0).frame_index == 0  # type: ignore[union-attr]

    assert pacer.next_frame(40_000_000).frame_index == 0  # type: ignore[union-attr]
    assert pacer.next_frame(50_000_000).frame_index == 0  # type: ignore[union-attr]
    assert pacer.status().starvations == 1
    pacer.enqueue(_live_frame(1))
    assert pacer.next_frame(50_000_001).frame_index == 1  # type: ignore[union-attr]


def test_live_frame_pacer_rebuilds_prebuffer_after_starvation() -> None:
    pacer = LiveFramePacer(
        target_queue_frames=2,
        startup_wait_ns=40_000_000,
    )
    pacer.enqueue(_live_frame(0))
    pacer.enqueue(_live_frame(1))
    assert pacer.next_frame(1).frame_index == 0  # type: ignore[union-attr]
    assert pacer.next_frame(34_000_000).frame_index == 1  # type: ignore[union-attr]
    assert pacer.next_frame(70_000_000).frame_index == 1  # type: ignore[union-attr]

    pacer.enqueue(_live_frame(2))
    assert pacer.next_frame(71_000_000).frame_index == 1  # type: ignore[union-attr]
    pacer.enqueue(_live_frame(3))
    assert pacer.next_frame(100_000_000).frame_index == 2  # type: ignore[union-attr]
    assert pacer.status(100_000_000).starvations == 1


def test_live_frame_pacer_resets_without_showing_an_old_session_frame() -> None:
    pacer = LiveFramePacer(
        target_queue_frames=2,
        startup_wait_ns=40_000_000,
    )
    pacer.enqueue(_live_frame(0))
    pacer.enqueue(_live_frame(1))
    assert pacer.next_frame(1).connection_session_id == "connection"  # type: ignore[union-attr]

    pacer.enqueue(_live_frame(10, stream="replacement"))
    pacer.enqueue(_live_frame(11, stream="replacement"))

    replacement = pacer.next_frame(2)
    assert replacement is not None
    assert replacement.connection_session_id == "replacement"
    assert replacement.frame_index == 10


def test_live_frame_buffer_publishes_read_only_contiguous_rgb() -> None:
    async def run() -> None:
        runtime = LiveFrameBuffer()
        try:
            decoded = _frame(23)
            decoded.pts = 9_000
            decoded.time_base = Fraction(1, 90_000)
            await runtime.submit_gateway_frame(
                connection_session_id="connection",
                frame_index=7,
                received_at_client_monotonic_ns=100,
                decoded_frame=decoded,
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
            assert latest.video_pts_ns == 100_000_000
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
                connection_session_id="connection",
                frame_index=0,
                received_at_client_monotonic_ns=1,
                decoded_frame=_frame(0),
            )
            await asyncio.sleep(0.01)
            for frame_index in range(1, 5):
                await runtime.submit_gateway_frame(
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
