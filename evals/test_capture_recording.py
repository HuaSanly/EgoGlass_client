from __future__ import annotations

import asyncio
import json
import threading
from fractions import Fraction
from pathlib import Path

import av
from aiortc import MediaStreamError

from schemas.recording import (
    FrameMetadataMatchStatus,
    ImuSensorType,
    RecordingFrameRow,
    RecordingImuRow,
)
from ui.gateway.adapters.mp4_recorder import PyAvH264Mp4Recorder
from ui.gateway.capture_recording import CaptureRecordingReader, CaptureRecordingWriter


def test_recording_artifacts_are_traceable_and_time_aligned(tmp_path: Path) -> None:
    recording_id = "c" * 32
    writer = CaptureRecordingWriter.create(
        tmp_path,
        recording_id=recording_id,
        countdown_started_at_unix_ns=1_000,
        countdown_started_at_client_monotonic_ns=10_000,
    )
    writer.video_path.write_bytes(b"eval-video")
    for index, frame_time in enumerate((1_000_000, 2_000_000, 3_000_000)):
        writer.append_frame(
            RecordingFrameRow(
                frame_index=index,
                recording_time_ns=frame_time,
                mp4_pts=index * 3_000,
                mp4_time_base_num=1,
                mp4_time_base_den=90_000,
                connection_session_id="eval-connection",
                received_at_client_monotonic_ns=10_000 + frame_time,
                metadata_match_status=FrameMetadataMatchStatus.UNMATCHED,
            )
        )
    for index, sample_time in enumerate((100_000, 1_500_000, 2_500_000, 3_500_000)):
        writer.append_imu(
            RecordingImuRow(
                sample_index=index,
                recording_time_ns=sample_time,
                connection_session_id="eval-connection",
                sensor_type=(
                    ImuSensorType.ACCELEROMETER if index % 2 == 0 else ImuSensorType.GYROSCOPE
                ),
                sequence_number=index // 2,
                sensor_event_monotonic_ns=20_000 + sample_time,
                received_at_elapsed_realtime_ns=21_000 + sample_time,
                received_at_client_monotonic_ns=22_000 + sample_time,
                accuracy=3,
                x=0.1,
                y=0.2,
                z=0.3,
                inside_video_span=False,
            )
        )
    reader = writer.finalize(ended_at_unix_ns=4_000_000)

    reopened = CaptureRecordingReader.open(reader.directory)
    imu_rows = tuple(reopened.iter_imu_samples())
    manifest_payload = json.loads((reader.directory / "manifest.json").read_text(encoding="utf-8"))

    assert reopened.hashes_verified
    assert [row.recording_time_ns for row in imu_rows] == [100_000, 1_500_000, 2_500_000]
    assert [row.inside_video_span for row in imu_rows] == [False, True, True]
    assert manifest_payload["recording_id"] == recording_id
    assert manifest_payload["frame_count"] == 3
    assert manifest_payload["imu_sample_count"] == 3
    assert set(manifest_payload["artifacts"]) == {"video", "imu", "frames", "quality"}


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
