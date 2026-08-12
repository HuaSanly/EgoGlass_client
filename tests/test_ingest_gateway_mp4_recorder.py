from __future__ import annotations

import asyncio
import threading
from fractions import Fraction
from pathlib import Path

import av
import pytest
from aiortc import MediaStreamError

from ui.gateway.adapters.mp4_recorder import PyAvH264Mp4Recorder
from ui.gateway.recording_inspection import inspect_recording


class NonMonotonicVideoTrack:
    def __init__(self) -> None:
        self.frames = [
            self._frame(90_000),
            self._frame(180_000),
            self._frame(0),
            self._frame(45_000),
        ]
        self.index = 0

    @staticmethod
    def _frame(pts: int) -> av.VideoFrame:
        frame = av.VideoFrame(640, 480, "yuv420p")
        frame.pts = pts
        frame.time_base = Fraction(1, 90_000)
        return frame

    async def recv(self) -> av.VideoFrame:
        if self.index == len(self.frames):
            raise MediaStreamError
        frame = self.frames[self.index]
        self.index += 1
        return frame


class VariableRateVideoTrack(NonMonotonicVideoTrack):
    def __init__(self) -> None:
        self.frames = [
            self._frame(90_000),
            self._frame(93_000),
            self._frame(97_500),
            self._frame(101_100),
        ]
        self.index = 0


def test_recorder_normalizes_non_monotonic_webrtc_timestamps(tmp_path: Path) -> None:
    async def scenario() -> None:
        path = tmp_path / "clip.mp4"
        track = NonMonotonicVideoTrack()
        original_timestamps = [frame.pts for frame in track.frames]
        receipt_times = iter(
            [1_000_000_000, 2_000_000_000, 2_033_333_333, 2_533_333_333]
        )
        recorder = PyAvH264Mp4Recorder(
            path,
            track,
            perf_clock=lambda: next(receipt_times),
        )

        await recorder.start()
        await recorder.wait()
        await recorder.stop()

        inspection = inspect_recording(path)
        assert (inspection.width, inspection.height) == (640, 480)
        assert inspection.decoded_frames == 4
        assert recorder.frames_received == 4
        assert [
            frame.received_at_client_perf_counter_ns for frame in recorder.frame_records
        ] == [1_000_000_000, 2_000_000_000, 2_033_333_333, 2_533_333_333]
        mp4_times = [
            Fraction(frame.mp4_pts * frame.mp4_time_base_num, frame.mp4_time_base_den)
            for frame in recorder.frame_records
        ]
        assert all(
            current > previous
            for previous, current in zip(mp4_times, mp4_times[1:], strict=False)
        )
        assert [frame.pts for frame in track.frames] == original_timestamps

    asyncio.run(scenario())


def test_recorder_finalization_does_not_block_the_receive_event_loop(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        recorder = PyAvH264Mp4Recorder(
            tmp_path / "yielding-finalize.mp4",
            VariableRateVideoTrack(),
            perf_clock=iter(
                (1_000_000_000, 1_033_000_000, 1_083_000_000, 1_123_000_000)
            ).__next__,
        )
        await recorder.start()
        await recorder.wait()

        finalization_started = threading.Event()
        finalization_release = threading.Event()
        original_finalize = recorder._finalize_container

        def controlled_finalize(container: object, stream: object) -> None:
            finalization_started.set()
            assert finalization_release.wait(timeout=5)
            original_finalize(container, stream)  # type: ignore[arg-type]

        recorder._finalize_container = controlled_finalize  # type: ignore[method-assign]
        stopping = asyncio.create_task(recorder.stop())
        assert await asyncio.to_thread(finalization_started.wait, 5)
        assert await asyncio.sleep(0, result="receive-loop-responsive") == (
            "receive-loop-responsive"
        )
        assert not stopping.done()
        finalization_release.set()
        await stopping

    asyncio.run(scenario())


def test_recorder_preserves_variable_rate_source_presentation_time(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        path = tmp_path / "variable-rate.mp4"
        track = VariableRateVideoTrack()
        receipt_times = iter(
            [1_000_000_000, 1_033_000_000, 1_083_000_000, 1_123_000_000]
        )
        recorder = PyAvH264Mp4Recorder(
            path,
            track,
            perf_clock=lambda: next(receipt_times),
        )

        await recorder.start()
        await recorder.wait()
        await recorder.stop()

        mp4_times = [
            Fraction(frame.mp4_pts * frame.mp4_time_base_num, frame.mp4_time_base_den)
            for frame in recorder.frame_records
        ]
        assert mp4_times == [
            Fraction(0),
            Fraction(1, 30),
            Fraction(1, 12),
            Fraction(37, 300),
        ]
        inspection = inspect_recording(path)
        assert inspection.average_fps == pytest.approx(24.324, abs=0.001)
        assert inspection.presentation_span_seconds == pytest.approx(
            0.123333,
            abs=0.000001,
        )

    asyncio.run(scenario())
