from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.recording_support import write_h264_video
from ui.gateway.adapters.mp4_recorder import RecordedVideoFrame
from ui.gateway.adapters.webrtc import WebRtcVideoRecordingSource
from ui.gateway.recording import (
    COUNTDOWN_SECONDS,
    RecordingFailureError,
    RecordingRuntime,
    RecordingUnavailableError,
)
from ui.gateway.webrtc_matcher import FrameMetadataMatch
from ui.gateway.webrtc_models import (
    ImuSample,
    ImuSensorType,
    VideoFrameMetadata,
)


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
        await asyncio.to_thread(
            write_h264_video,
            self.path,
            width=640,
            height=480,
            frame_count=1,
            fps=30,
        )


async def _advance_until(predicate: Callable[[], bool]) -> None:
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("asynchronous state did not advance")


def _imu(sensor: ImuSensorType, sequence: int, event_ns: int) -> ImuSample:
    return ImuSample(
        sensor_type=sensor,
        android_sensor_type=1 if sensor is ImuSensorType.ACCELEROMETER else 4,
        sequence_number=sequence,
        sensor_event_monotonic_ns=event_ns,
        received_at_elapsed_realtime_ns=event_ns + 1,
        accuracy=3,
        values=(0.1, 0.2, 0.3),
    )


def _match(frame_id: int, device_ns: int) -> FrameMetadataMatch:
    return FrameMetadataMatch(
        metadata=VideoFrameMetadata(
            frame_id=frame_id,
            camera_start_generation=1,
            captured_at_rokid_sdk_ms=1_000 + frame_id,
            received_at_elapsed_realtime_ns=device_ns,
            video_at_monotonic_ns=device_ns,
            rtp_timestamp_90khz=frame_id * 3_000,
            width=640,
            height=480,
            rotation_degrees=0,
            capture_config_id="capture-640x480",
        ),
        decoded_frame_pts=100,
        decoded_frame_index=0,
        decoded_frame_time_base_num=1,
        decoded_frame_time_base_den=90_000,
        decoded_frame_received_at_client_monotonic_ns=device_ns,
        timestamp_match_error_90khz=0,
    )


async def _submit_coverage(
    runtime: RecordingRuntime,
    connection_id: str,
    *,
    camera_ns: int,
    client_clock: list[int],
) -> None:
    for sensor in (ImuSensorType.ACCELEROMETER, ImuSensorType.GYROSCOPE):
        await runtime.on_imu_sample(
            connection_id,
            _imu(sensor, 0, camera_ns - 1),
            client_clock[0],
        )
        await runtime.on_imu_sample(
            connection_id,
            _imu(sensor, 1, camera_ns + 1),
            client_clock[0],
        )


def test_consecutive_recordings_publish_independent_four_file_units(tmp_path: Path) -> None:
    async def scenario() -> None:
        connection_id = "a" * 32
        ids = iter(("1" * 32, "2" * 32))
        clock = [10_000_000_000]
        countdown = _Countdown()
        recorders: list[_Recorder] = []

        def factory(path: Path, track: object, **profile: int) -> _Recorder:
            recorder = _Recorder(
                path,
                track,
                **profile,
                receipt_clock=lambda: clock[0],
            )
            recorders.append(recorder)
            return recorder

        source = WebRtcVideoRecordingSource(connection_id, _VideoSource(), 640, 480, 1)
        runtime = RecordingRuntime(
            tmp_path,
            lambda: asyncio.sleep(0, result=source),
            recorder_factory=factory,
            sleep=countdown,
            monotonic_clock_ns=lambda: clock[0],
            recording_id_factory=lambda: next(ids),
        )

        for index, recording_id in enumerate(("1" * 32, "2" * 32), start=1):
            status = await runtime.start()
            assert status.recording_id == recording_id
            camera_ns = 20_000_000_000 + index
            await runtime.on_imu_sample(
                connection_id,
                _imu(ImuSensorType.ACCELEROMETER, 0, camera_ns - 1),
                clock[0],
            )
            countdown.release.set()
            expected_recorder_count = index
            await _advance_until(
                lambda expected=expected_recorder_count: len(recorders) == expected
            )
            await runtime.on_frame_metadata_match(connection_id, _match(index, camera_ns))
            await runtime.on_imu_sample(
                connection_id,
                _imu(ImuSensorType.ACCELEROMETER, 1, camera_ns + 1),
                clock[0],
            )
            for sequence, timestamp in enumerate((camera_ns - 1, camera_ns + 1)):
                await runtime.on_imu_sample(
                    connection_id,
                    _imu(ImuSensorType.GYROSCOPE, sequence, timestamp),
                    clock[0],
                )
            await runtime.stop()
            camera_rows = tuple(
                (await runtime.reader(recording_id)).iter_camera_frames()
            )
            assert camera_rows[0].rokid_timestamp_ns == (1_000 + index) * 1_000_000
            assert {path.name for path in (tmp_path / recording_id).iterdir()} == {
                "video.mp4",
                "camera.csv",
                "imu.csv",
                "calibration.yaml",
            }
            clock[0] += 1_000_000_000

        library = await runtime.library()
        assert {item.recording_id for item in library.recordings} == {
            "1" * 32,
            "2" * 32,
        }
        assert not list(tmp_path.glob("*/telemetry.sqlite"))

    asyncio.run(scenario())


def test_missing_glasses_metadata_keeps_partial_and_prevents_publication(tmp_path: Path) -> None:
    async def scenario() -> None:
        connection_id = "a" * 32
        clock = [10_000_000_000]
        countdown = _Countdown()
        recorders: list[_Recorder] = []

        def factory(path: Path, track: object, **profile: int) -> _Recorder:
            recorder = _Recorder(
                path,
                track,
                **profile,
                receipt_clock=lambda: clock[0],
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
            monotonic_clock_ns=lambda: clock[0],
            recording_id_factory=lambda: "3" * 32,
        )
        await runtime.start()
        countdown.release.set()
        await _advance_until(lambda: bool(recorders))
        await _submit_coverage(
            runtime,
            connection_id,
            camera_ns=20_000_000_000,
            client_clock=clock,
        )

        with pytest.raises(RecordingFailureError, match="no matching Glass3 metadata"):
            await runtime.stop()

        assert not (tmp_path / ("3" * 32)).exists()
        assert (tmp_path / f".recording-{'3' * 32}.partial").is_dir()

    asyncio.run(scenario())


def test_recording_rejects_non_protocol_video_resolution(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = RecordingRuntime(
            tmp_path,
            lambda: asyncio.sleep(
                0,
                result=WebRtcVideoRecordingSource(
                    "a" * 32,
                    _VideoSource(),
                    1280,
                    720,
                    1,
                ),
            ),
        )

        with pytest.raises(RecordingUnavailableError, match="requires the 640x480"):
            await runtime.start()

        assert not tuple(tmp_path.iterdir())

    asyncio.run(scenario())


def test_recording_finalization_yields_to_receive_tasks(tmp_path: Path) -> None:
    async def scenario() -> None:
        connection_id = "a" * 32
        clock = [10_000_000_000]
        countdown = _Countdown()
        stop_gate = asyncio.Event()
        recorders: list[_Recorder] = []

        def factory(path: Path, track: object, **profile: int) -> _Recorder:
            recorder = _Recorder(
                path,
                track,
                **profile,
                receipt_clock=lambda: clock[0],
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
            monotonic_clock_ns=lambda: clock[0],
            recording_id_factory=lambda: "4" * 32,
        )
        await runtime.start()
        countdown.release.set()
        await _advance_until(lambda: bool(recorders))
        camera_ns = 20_000_000_000
        await runtime.on_frame_metadata_match(connection_id, _match(1, camera_ns))
        await _submit_coverage(
            runtime,
            connection_id,
            camera_ns=camera_ns,
            client_clock=clock,
        )

        stopping = asyncio.create_task(runtime.stop())
        await asyncio.sleep(0)
        assert await asyncio.create_task(asyncio.sleep(0, result="received")) == "received"
        assert not stopping.done()
        stop_gate.set()
        await stopping

    asyncio.run(scenario())
