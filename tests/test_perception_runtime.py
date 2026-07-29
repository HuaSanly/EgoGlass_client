from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path

from av import VideoFrame

from perception.runtime import HandTrackingRuntime, LiveHandTrackingFrame
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
