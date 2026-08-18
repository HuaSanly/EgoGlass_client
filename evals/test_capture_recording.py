from __future__ import annotations

import asyncio
import threading
from fractions import Fraction
from pathlib import Path

import av
from aiortc import MediaStreamError

from schemas.recording import RecordingOutput
from tests.recording_support import (
    append_covering_imu,
    create_recording,
    staged_camera_frames,
    write_h264_video,
)
from ui.gateway.adapters.aiortc_peer import AiortcPeer, AiortcVideoSource
from ui.gateway.adapters.mp4_recorder import PyAvH264Mp4Recorder
from ui.gateway.adapters.webrtc import WebRtcPeerCallbacks
from ui.gateway.capture_recording import CaptureRecordingReader, CaptureRecordingWriter
from ui.gateway.recording import MetadataMatchedVideoTrack


def test_thirty_second_recording_satisfies_minimal_protocol(tmp_path: Path) -> None:
    recording_id = "c" * 32
    writer = CaptureRecordingWriter.create(
        tmp_path,
        recording_id=recording_id,
        video_profile=RecordingOutput(width=32, height=24, fps=10),
    )
    video_index = write_h264_video(
        writer.video_path,
        width=32,
        height=24,
        frame_count=300,
        fps=10,
    )
    frames = staged_camera_frames(video_index)
    append_covering_imu(writer, frames)
    reader = writer.finalize(frames)

    reopened = CaptureRecordingReader.open(reader.directory)
    summary = reopened.summary()

    assert summary.protocol_validated
    assert summary.frame_count == 300
    assert summary.duration_ns == 29_900_000_000
    assert {path.name for path in reader.directory.iterdir()} == {
        "video.mp4",
        "camera.csv",
        "imu.csv",
        "calibration.yaml",
    }


def test_mp4_finalize_keeps_the_capture_loop_responsive(tmp_path: Path) -> None:
    class Track:
        def __init__(self) -> None:
            self.sent = False

        async def recv(self) -> av.VideoFrame:
            if self.sent:
                raise MediaStreamError
            self.sent = True
            frame = av.VideoFrame(32, 24, "yuv420p")
            frame.pts = 0
            frame.time_base = Fraction(1, 30)
            return frame

    async def scenario() -> None:
        recorder = PyAvH264Mp4Recorder(
            tmp_path / "eval-finalize.mp4",
            Track(),
            width=32,
            height=24,
            perf_clock=lambda: 1_000_000_000,
        )
        await recorder.start()
        await recorder.wait()
        started = threading.Event()
        release = threading.Event()
        original_finalize = recorder._finalize_container

        def controlled_finalize(container: object, stream: object) -> None:
            started.set()
            assert release.wait(timeout=5)
            original_finalize(container, stream)  # type: ignore[arg-type]

        recorder._finalize_container = controlled_finalize  # type: ignore[method-assign]
        stopping = asyncio.create_task(recorder.stop())
        assert await asyncio.to_thread(started.wait, 5)
        heartbeat = asyncio.create_task(asyncio.sleep(0, result=True))
        assert await heartbeat
        assert not stopping.done()
        release.set()
        await stopping

    asyncio.run(scenario())


def test_ten_consecutive_recordings_never_share_files(tmp_path: Path) -> None:
    recording_ids = tuple(f"{index:032x}" for index in range(10))

    for recording_id in recording_ids:
        create_recording(
            tmp_path,
            recording_id=recording_id,
            frame_count=1,
        )

    directories = tuple(tmp_path / recording_id for recording_id in recording_ids)
    assert all(directory.is_dir() for directory in directories)
    assert len({path.resolve() for path in directories}) == 10
    assert all(
        {artifact.name for artifact in directory.iterdir()}
        == {"video.mp4", "camera.csv", "imu.csv", "calibration.yaml"}
        for directory in directories
    )
    assert not tuple(tmp_path.glob(".recording-*.partial"))


def test_canonical_ingest_keeps_every_frame_needed_by_recording_metadata(
    monkeypatch: object,
) -> None:
    async def scenario() -> None:
        subscriptions: list[bool] = []

        async def ignore(*_args: object) -> None:
            return None

        class BlockingTrack:
            async def recv(self) -> object:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

        def subscribe(
            _source: AiortcVideoSource,
            *,
            buffered: bool,
        ) -> object:
            subscriptions.append(buffered)
            return BlockingTrack()

        monkeypatch.setattr(AiortcVideoSource, "subscribe", subscribe)  # type: ignore[attr-defined]
        callbacks = WebRtcPeerCallbacks(
            on_connection_state=ignore,
            on_video_source=ignore,
            on_video_frame=ignore,
            on_metadata=ignore,
            on_control_channel_ready=ignore,
            on_control_channel_closed=ignore,
            on_control_status=ignore,
            on_imu_channel_ready=ignore,
            on_imu_channel_closed=ignore,
            on_imu_telemetry=ignore,
        )
        peer = AiortcPeer(callbacks)
        try:
            peer._peer.emit("track", type("VideoTrack", (), {"kind": "video"})())
            await asyncio.sleep(0)
            assert subscriptions == [True]
        finally:
            await peer.close()

    asyncio.run(scenario())


def test_sparse_metadata_loss_drops_only_unpublishable_video_frames(
    tmp_path: Path,
) -> None:
    class Track:
        def __init__(self) -> None:
            self.frame_index = 0

        async def recv(self) -> av.VideoFrame:
            if self.frame_index == 120:
                raise MediaStreamError
            frame = av.VideoFrame(32, 24, "yuv420p")
            frame.pts = self.frame_index * 3_000
            frame.time_base = Fraction(1, 90_000)
            self.frame_index += 1
            return frame

    async def scenario() -> None:
        missing_pts = {42 * 3_000, 78 * 3_000}
        filtered = MetadataMatchedVideoTrack(
            Track(),
            lambda pts: asyncio.sleep(0, result=pts not in missing_pts),
        )
        recorder = PyAvH264Mp4Recorder(
            tmp_path / "metadata-filtered.mp4",
            filtered,
            width=32,
            height=24,
        )

        await recorder.start()
        await recorder.wait()
        await recorder.stop()

        assert recorder.frames_received == 118
        assert filtered.skipped_frame_count == 2
        assert {record.source_frame_pts for record in recorder.frame_records}.isdisjoint(
            missing_pts
        )

    asyncio.run(scenario())
