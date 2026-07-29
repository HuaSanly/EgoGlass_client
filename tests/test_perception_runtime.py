from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from av import VideoFrame

from perception.runtime import (
    HandTrackingRuntime,
    LiveHandTrackingFrame,
    _H264ReplayWriter,
)
from perception.sensor_preprocessing import (
    ClockId,
    TimeObservation,
    TimestampSemantic,
    client_perf_source_instance_id,
)


@dataclass(frozen=True)
class FakeResult:
    frame_index: int

    def to_json_dict(self) -> dict[str, object]:
        return {"schema_version": "1.0", "frame_index": self.frame_index, "hands": []}


def _runtime_config(path: Path) -> Path:
    path.write_text(
        """schema_version: \"1.0\"
enabled: true
max_live_inference_fps: 60.0
replay:
  inference_stride_frames: 5
""",
        encoding="utf-8",
    )
    return path


def _frame(index: int, received_at_ns: int) -> LiveHandTrackingFrame:
    return LiveHandTrackingFrame(
        session_id="a" * 32,
        connection_session_id="a" * 32,
        frame_index=index,
        received_at_client_monotonic_ns=received_at_ns,
        decoded_frame=VideoFrame(8, 6, "yuv420p"),
    )


def test_replay_writer_creates_fast_start_h264_mp4(tmp_path: Path) -> None:
    path = tmp_path / "annotated.mp4"
    writer = _H264ReplayWriter(path, 30.0, 8, 6)
    for value, presentation_time_ns in (
        (20, 1_000_000_000),
        (120, 1_040_000_000),
        (220, 1_090_000_000),
    ):
        writer.write(
            np.full((6, 8, 3), value, dtype=np.uint8),
            presentation_time_ns,
        )
    writer.close()
    writer.close()

    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        frames = list(container.decode(stream))

    encoded = path.read_bytes()
    assert stream.codec_context.name == "h264"
    assert len(frames) == 3
    assert [Fraction(frame.pts) * frame.time_base for frame in frames] == [
        Fraction(0),
        Fraction(1, 25),
        Fraction(9, 100),
    ]
    assert encoded.index(b"moov") < encoded.index(b"mdat")


def test_live_runtime_keeps_only_newest_pending_frame(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()
    processed: list[int] = []

    async def scenario() -> None:
        runtime = HandTrackingRuntime(
            recordings_root=tmp_path / "recordings",
            runtime_config_path=_runtime_config(tmp_path / "runtime.yaml"),
        )

        def process(frame: LiveHandTrackingFrame) -> FakeResult:
            processed.append(frame.frame_index)
            if frame.frame_index == 0:
                started.set()
                assert release.wait(timeout=2)
            return FakeResult(frame.frame_index)

        runtime._process_live_frame = process  # type: ignore[method-assign]
        await runtime.submit_live_frame(_frame(0, 0))
        assert await asyncio.to_thread(started.wait, 2)
        await runtime.submit_live_frame(_frame(1, 20_000_000))
        await runtime.submit_live_frame(_frame(2, 40_000_000))
        release.set()
        await runtime.close()
        status = await runtime.status()

        assert processed == [0, 2]
        assert status["live_frames_received"] == 3
        assert status["live_frames_dropped"] == 1
        assert status["live_inferences"] == 2
        assert status["latest_result"] == {
            "schema_version": "1.0",
            "frame_index": 2,
            "hands": [],
        }

    asyncio.run(scenario())


def test_status_events_push_initial_snapshot_and_completed_inference(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = HandTrackingRuntime(
            recordings_root=tmp_path / "recordings",
            runtime_config_path=_runtime_config(tmp_path / "runtime.yaml"),
        )
        runtime._process_live_frame = (  # type: ignore[method-assign]
            lambda frame: FakeResult(frame.frame_index)
        )
        events = runtime.status_events(heartbeat_seconds=0.05)
        try:
            initial = await anext(events)
            assert initial is not None
            assert initial["status_revision"] == 0
            assert initial["latest_result"] is None

            await runtime.submit_live_frame(_frame(9, 0))
            pushed = await asyncio.wait_for(anext(events), timeout=1)
            while pushed is None or pushed["latest_result"] is None:
                pushed = await asyncio.wait_for(anext(events), timeout=1)

            assert pushed["status_revision"] > initial["status_revision"]
            assert pushed["live_inferences"] == 1
            assert pushed["latest_result"] == {
                "schema_version": "1.0",
                "frame_index": 9,
                "hands": [],
            }
            assert await asyncio.wait_for(anext(events), timeout=0.2) is None
        finally:
            await events.aclose()
            await runtime.close()

    asyncio.run(scenario())


def test_live_preprocessing_anchors_session_time_at_first_received_frame(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    runtime = HandTrackingRuntime(
        recordings_root=tmp_path / "recordings",
        runtime_config_path=_runtime_config(tmp_path / "runtime.yaml"),
        sensor_config_path=repository / "config" / "sensor-preprocessing.yaml",
    )
    first_received_ns = 8_724_913_800_000
    frame = _frame(0, first_received_ns)

    try:
        preprocessor = runtime._live_preprocessing_for(frame)
        estimate = preprocessor.clock_mapper.map(
            TimeObservation(
                session_id=frame.session_id,
                source_clock_id=ClockId.CLIENT_PERF_COUNTER_NS,
                source_instance_id=client_perf_source_instance_id(
                    frame.session_id,
                    frame.connection_session_id,
                ),
                source_timestamp=first_received_ns,
                timestamp_semantic=TimestampSemantic.CLIENT_RECEIPT,
            )
        )

        assert preprocessor.clock_mapper.segments[0].source_from == first_received_ns
        assert estimate.session_time_ns == 0
    finally:
        asyncio.run(runtime.close())


def test_runtime_rejects_recording_path_escape(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    runtime = HandTrackingRuntime(
        recordings_root=recordings,
        runtime_config_path=_runtime_config(tmp_path / "runtime.yaml"),
    )

    async def scenario() -> None:
        try:
            try:
                await runtime.start_replay("../outside")
            except Exception as error:
                assert "escapes recordings root" in str(error)
            else:
                raise AssertionError("path escape was accepted")
        finally:
            await runtime.close()

    asyncio.run(scenario())
