from __future__ import annotations

import asyncio
from fractions import Fraction
from pathlib import Path

import av
from aiortc import MediaStreamError

from egoglass_ingest_gateway.adapters.mp4_recorder import PyAvH264Mp4Recorder
from egoglass_ingest_gateway.recording_inspection import inspect_recording


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
        frame = av.VideoFrame(1280, 720, "yuv420p")
        frame.pts = pts
        frame.time_base = Fraction(1, 90_000)
        return frame

    async def recv(self) -> av.VideoFrame:
        if self.index == len(self.frames):
            raise MediaStreamError
        frame = self.frames[self.index]
        self.index += 1
        return frame


def test_recorder_normalizes_non_monotonic_webrtc_timestamps(tmp_path: Path) -> None:
    async def scenario() -> None:
        path = tmp_path / "clip.mp4"
        track = NonMonotonicVideoTrack()
        original_timestamps = [frame.pts for frame in track.frames]
        receipt_times = iter([10, 20, 30, 40])
        recorder = PyAvH264Mp4Recorder(
            path,
            track,
            perf_clock=lambda: next(receipt_times),
        )

        await recorder.start()
        await recorder.wait()
        await recorder.stop()

        inspection = inspect_recording(path)
        assert inspection.nominal_fps == 30.0
        assert inspection.decoded_frames == 4
        assert recorder.frames_received == 4
        assert [
            frame.received_at_client_perf_counter_ns for frame in recorder.frame_records
        ] == [10, 20, 30, 40]
        assert [frame.pts for frame in track.frames] == original_timestamps

    asyncio.run(scenario())
