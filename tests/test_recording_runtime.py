from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from ui.gateway.adapters.mp4_recorder import RecordedVideoFrame
from ui.gateway.adapters.webrtc import WebRtcVideoRecordingSource
from ui.gateway.recording import COUNTDOWN_SECONDS, RecordingRuntime
from ui.gateway.webrtc_models import ImuSample, ImuSensorType


class _VideoSource:
    def subscribe(self, *, buffered: bool) -> object:
        assert buffered is True
        return object()


class _Countdown:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, seconds: float) -> None:
        assert seconds == COUNTDOWN_SECONDS
        self.started.set()
        await self.release.wait()


class _Recorder:
    def __init__(
        self,
        path: Path,
        _track: object,
        *,
        width: int,
        height: int,
        fps: int,
        receipt_clock: Callable[[], int],
        stop_gate: asyncio.Event | None = None,
    ) -> None:
        assert (width, height, fps) == (640, 480, 30)
        self.path = path
        self.receipt_clock = receipt_clock
        self.stop_gate = stop_gate
        self.frames_received = 1
        self.wait_forever = asyncio.Event()

    @property
    def frame_records(self) -> tuple[RecordedVideoFrame, ...]:
        return (
            RecordedVideoFrame(
                frame_index=0,
                source_frame_pts=100,
                source_frame_time_base_num=1,
                source_frame_time_base_den=90_000,
                mp4_pts=0,
                mp4_time_base_num=1,
                mp4_time_base_den=30,
                received_at_client_perf_counter_ns=self.receipt_clock(),
            ),
        )

    async def start(self) -> None:
        return None

    async def wait(self) -> None:
        await self.wait_forever.wait()

    async def stop(self) -> None:
        if self.stop_gate is not None:
            await self.stop_gate.wait()
        self.path.write_bytes(b"synthetic-h264-mp4")


async def _advance_until(predicate: Callable[[], bool]) -> None:
    for _ in range(50):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("asynchronous state did not advance")


def _imu(sequence: int, event_ns: int) -> ImuSample:
    return ImuSample(
        sensor_type=ImuSensorType.GYROSCOPE,
        android_sensor_type=4,
        sequence_number=sequence,
        sensor_event_monotonic_ns=event_ns,
        received_at_elapsed_realtime_ns=event_ns + 1_000_000,
        accuracy=3,
        values=(0.1, 0.2, 0.3),
    )


def test_consecutive_recordings_are_independent_and_keep_countdown_imu(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        connection_id = "a" * 32
        ids = iter(("1" * 32, "2" * 32))
        monotonic_ns = [10_000_000_000]
        unix_ns = [1_900_000_000_000_000_000]
        countdown = _Countdown()
        recorders: list[_Recorder] = []

        def recorder_factory(
            path: Path,
            track: object,
            *,
            width: int,
            height: int,
            fps: int,
        ) -> _Recorder:
            recorder = _Recorder(
                path,
                track,
                width=width,
                height=height,
                fps=fps,
                receipt_clock=lambda: monotonic_ns[0],
            )
            recorders.append(recorder)
            return recorder

        source = WebRtcVideoRecordingSource(
            connection_id,
            _VideoSource(),
            640,
            480,
            1,
        )
        runtime = RecordingRuntime(
            tmp_path,
            lambda: asyncio.sleep(0, result=source),
            recorder_factory=recorder_factory,
            sleep=countdown,
            unix_clock_ns=lambda: unix_ns[0],
            monotonic_clock_ns=lambda: monotonic_ns[0],
            recording_id_factory=lambda: next(ids),
        )

        first = await runtime.start()
        assert first.recording_id == "1" * 32
        monotonic_ns[0] += 100_000_000
        await runtime.on_imu_sample(connection_id, _imu(0, 1), monotonic_ns[0])
        countdown.release.set()
        await _advance_until(lambda: len(recorders) == 1)
        monotonic_ns[0] += 3_000_000_000
        await runtime.on_imu_sample(connection_id, _imu(1, 2), monotonic_ns[0])
        await runtime.stop()

        reader = await runtime.reader("1" * 32)
        imu = tuple(reader.iter_imu_samples())
        assert len(imu) == 2
        assert [row.inside_video_span for row in imu] == [False, True]
        assert reader.manifest.frame_count == 1
        assert (tmp_path / ("1" * 32) / "video.mp4").is_file()

        monotonic_ns[0] += 1_000_000_000
        await runtime.start()
        await _advance_until(lambda: len(recorders) == 2)
        monotonic_ns[0] += 3_000_000_000
        await runtime.stop()

        library = await runtime.library()
        assert {item.recording_id for item in library.recordings} == {
            "1" * 32,
            "2" * 32,
        }
        assert (tmp_path / ("2" * 32) / "manifest.json").is_file()
        assert not list(tmp_path.glob("*/telemetry.sqlite"))

    asyncio.run(scenario())


def test_recording_finalization_yields_to_other_receive_tasks(tmp_path: Path) -> None:
    async def scenario() -> None:
        connection_id = "a" * 32
        monotonic_ns = [10_000_000_000]
        countdown = _Countdown()
        stop_gate = asyncio.Event()
        recorders: list[_Recorder] = []

        def factory(
            path: Path,
            track: object,
            *,
            width: int,
            height: int,
            fps: int,
        ) -> _Recorder:
            recorder = _Recorder(
                path,
                track,
                width=width,
                height=height,
                fps=fps,
                receipt_clock=lambda: monotonic_ns[0],
                stop_gate=stop_gate,
            )
            recorders.append(recorder)
            return recorder

        runtime = RecordingRuntime(
            tmp_path,
            lambda: asyncio.sleep(
                0,
                result=WebRtcVideoRecordingSource(
                    connection_id,
                    _VideoSource(),
                    640,
                    480,
                    1,
                ),
            ),
            recorder_factory=factory,
            sleep=countdown,
            monotonic_clock_ns=lambda: monotonic_ns[0],
            recording_id_factory=lambda: "3" * 32,
        )
        await runtime.start()
        countdown.release.set()
        await _advance_until(lambda: bool(recorders))
        monotonic_ns[0] += 4_000_000_000

        stopping = asyncio.create_task(runtime.stop())
        await asyncio.sleep(0)
        heartbeat = asyncio.create_task(asyncio.sleep(0, result="received"))
        assert await heartbeat == "received"
        assert not stopping.done()
        stop_gate.set()
        await stopping

    asyncio.run(scenario())
