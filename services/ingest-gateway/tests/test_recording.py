from __future__ import annotations

import asyncio
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import av
import pytest

from egoglass_ingest_gateway.adapters.webrtc import WebRtcVideoRecordingSource
from egoglass_ingest_gateway.recording import (
    COUNTDOWN_SECONDS,
    RecordingConflictError,
    RecordingRuntime,
    RecordingUnavailableError,
)
from egoglass_ingest_gateway.recording_inspection import inspect_recording
from egoglass_ingest_gateway.recording_models import RecordingCommandRequest

SESSION_ID = "a" * 32


class FakeVideoSource:
    def __init__(self) -> None:
        self.subscriptions: list[tuple[object, bool]] = []

    def subscribe(self, *, buffered: bool) -> object:
        track = object()
        self.subscriptions.append((track, buffered))
        return track


class ControlledCountdown:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def sleep(self, seconds: float) -> None:
        assert seconds == COUNTDOWN_SECONDS
        self.started.set()
        await self.release.wait()


class FakeRecorder:
    def __init__(self, path: Path, _track: object) -> None:
        self.path = path
        self.started = False
        self.frames_received = 1
        self.finished = asyncio.Event()

    async def start(self) -> None:
        self.started = True

    async def wait(self) -> None:
        await self.finished.wait()

    async def stop(self) -> None:
        self.path.write_bytes(b"finalized-mp4")


async def advance_until(predicate: Callable[[], bool]) -> None:
    for _attempt in range(20):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("asynchronous state did not advance")


def test_three_second_countdown_then_completed_clip_is_grouped_by_session(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        source = FakeVideoSource()
        video = WebRtcVideoRecordingSource(SESSION_ID, source, 1920, 1080)
        countdown = ControlledCountdown()
        writers: list[FakeRecorder] = []
        wall_ms = [1_000_000]
        monotonic_ns = [5_000_000_000]

        def factory(path: Path, track: object) -> FakeRecorder:
            writer = FakeRecorder(path, track)
            writers.append(writer)
            return writer

        runtime = RecordingRuntime(
            tmp_path,
            lambda: asyncio.sleep(0, result=video),
            recorder_factory=factory,
            sleep=countdown.sleep,
            unix_clock_ms=lambda: wall_ms[0],
            monotonic_clock_ns=lambda: monotonic_ns[0],
        )
        countdown_status = await runtime.start()
        assert countdown_status.state == "countdown"
        assert countdown_status.recording_starts_at_unix_ms == 1_003_000
        assert source.subscriptions == []

        await countdown.started.wait()
        wall_ms[0] = 1_003_006
        countdown.release.set()
        await advance_until(lambda: bool(writers and writers[0].started))
        recording_status = await runtime.status()
        assert recording_status.state == "recording"
        assert recording_status.recording_started_at_unix_ms == 1_003_006
        assert source.subscriptions[0][1] is True

        wall_ms[0] = 1_005_506
        monotonic_ns[0] = 7_500_000_000
        completed = await runtime.stop()
        assert completed.state == "ready"
        assert completed.recording_duration_ms == 2500
        assert not list(tmp_path.rglob("*.part.mp4"))

        library = await runtime.library()
        assert len(library.sessions) == 1
        assert library.sessions[0].session_id == SESSION_ID
        assert len(library.sessions[0].clips) == 1
        clip = library.sessions[0].clips[0]
        assert clip.duration_ms == 2500
        assert clip.file_size_bytes == len(b"finalized-mp4")
        assert await runtime.media_path(SESSION_ID, clip.clip_id) == (
            tmp_path / SESSION_ID / f"{clip.clip_id}.mp4"
        )
        assert await runtime.media_path("../outside", clip.clip_id) is None
        (tmp_path / SESSION_ID / f"{clip.clip_id}.mp4").write_bytes(b"truncated")
        assert await runtime.media_path(SESSION_ID, clip.clip_id) is None

    asyncio.run(scenario())


def test_stopping_during_countdown_creates_no_clip(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = FakeVideoSource()
        countdown = ControlledCountdown()
        runtime = RecordingRuntime(
            tmp_path,
            lambda: asyncio.sleep(
                0,
                result=WebRtcVideoRecordingSource(SESSION_ID, source, 1920, 1080),
            ),
            recorder_factory=FakeRecorder,
            sleep=countdown.sleep,
        )
        await runtime.start()
        await countdown.started.wait()
        status = await runtime.stop()

        assert status.state == "ready"
        assert status.clip_id is None
        assert source.subscriptions == []
        assert (await runtime.library()).sessions == []
        assert not list(tmp_path.rglob("*"))
        with pytest.raises(RecordingConflictError):
            await runtime.stop()

    asyncio.run(scenario())


def test_source_end_finalizes_current_recording(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = FakeVideoSource()
        countdown = ControlledCountdown()
        writers: list[FakeRecorder] = []

        def factory(path: Path, track: object) -> FakeRecorder:
            writer = FakeRecorder(path, track)
            writers.append(writer)
            return writer

        runtime = RecordingRuntime(
            tmp_path,
            lambda: asyncio.sleep(
                0,
                result=WebRtcVideoRecordingSource(SESSION_ID, source, 1920, 1080),
            ),
            recorder_factory=factory,
            sleep=countdown.sleep,
        )
        await runtime.start()
        countdown.release.set()
        await advance_until(lambda: bool(writers and writers[0].started))
        writers[0].finished.set()
        await advance_until(
            lambda: (tmp_path / SESSION_ID / "session.json").is_file()
        )

        assert (await runtime.status()).state == "ready"
        assert len((await runtime.library()).sessions[0].clips) == 1

    asyncio.run(scenario())


def test_session_replacement_during_countdown_publishes_nothing(tmp_path: Path) -> None:
    async def scenario() -> None:
        countdown = ControlledCountdown()
        sources = [
            WebRtcVideoRecordingSource(SESSION_ID, FakeVideoSource(), 1920, 1080),
            WebRtcVideoRecordingSource("b" * 32, FakeVideoSource(), 1920, 1080),
        ]

        async def source_provider() -> WebRtcVideoRecordingSource:
            return sources.pop(0)

        runtime = RecordingRuntime(
            tmp_path,
            source_provider,
            recorder_factory=FakeRecorder,
            sleep=countdown.sleep,
        )
        await runtime.start()
        countdown.release.set()
        await advance_until(lambda: not sources)
        await advance_until(lambda: not list(tmp_path.rglob("*")))

        status = await runtime.status()
        assert status.state == "error"
        assert status.detail == "recording failed: RecordingUnavailableError"
        assert (await runtime.library()).sessions == []

    asyncio.run(scenario())


def test_recording_rejects_non_full_hd_source(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = WebRtcVideoRecordingSource(
            SESSION_ID,
            FakeVideoSource(),
            1280,
            720,
        )
        runtime = RecordingRuntime(
            tmp_path,
            lambda: asyncio.sleep(0, result=source),
            recorder_factory=FakeRecorder,
        )
        with pytest.raises(RecordingUnavailableError, match="must be 1920x1080"):
            await runtime.start()
        assert not list(tmp_path.iterdir())

    asyncio.run(scenario())


def test_recording_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError):
        RecordingCommandRequest.model_validate({"action": "start", "delay": 0})


def test_inspector_decodes_full_hd_h264_mp4(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    with av.open(str(path), mode="w", format="mp4") as container:
        stream = container.add_stream(
            "libx264",
            rate=30,
            options={"preset": "ultrafast"},
        )
        stream.width = 1920
        stream.height = 1080
        stream.pix_fmt = "yuv420p"
        for pts in range(2):
            frame = av.VideoFrame(1920, 1080, "yuv420p")
            frame.pts = pts
            frame.time_base = Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)

    result = inspect_recording(path)

    assert result.video_codec == "h264"
    assert (result.width, result.height) == (1920, 1080)
    assert result.nominal_fps == 30.0
    assert result.decoded_frames == 2
