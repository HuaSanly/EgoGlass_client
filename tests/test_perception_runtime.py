from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from av import VideoFrame

from perception.runtime import (
    HandTrackingRuntime,
    LiveHandTrackingFrame,
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


def test_live_runtime_accepts_an_immutable_gateway_rgb_frame(tmp_path: Path) -> None:
    image_rgb = np.full((6, 8, 3), 41, dtype=np.uint8)
    image_rgb.setflags(write=False)
    processed: list[LiveHandTrackingFrame] = []

    async def scenario() -> None:
        runtime = HandTrackingRuntime(
            runtime_config_path=_runtime_config(tmp_path / "runtime.yaml"),
        )

        def process(frame: LiveHandTrackingFrame) -> FakeResult:
            processed.append(frame)
            return FakeResult(frame.frame_index)

        runtime._process_live_frame = process  # type: ignore[method-assign]
        await runtime.submit_rgb_frame(
            session_id="a" * 32,
            connection_session_id="b" * 32,
            frame_index=12,
            received_at_client_monotonic_ns=500,
            image_rgb=image_rgb,
        )
        await runtime.close()

    asyncio.run(scenario())
    assert len(processed) == 1
    assert processed[0].decoded_frame is None
    assert processed[0].image_rgb is image_rgb


def test_offline_gpu_claim_waits_for_inflight_live_inference(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()

    async def scenario() -> None:
        runtime = HandTrackingRuntime(
            runtime_config_path=_runtime_config(tmp_path / "runtime.yaml"),
        )

        def process(frame: LiveHandTrackingFrame) -> FakeResult:
            started.set()
            assert release.wait(timeout=2)
            return FakeResult(frame.frame_index)

        runtime._process_live_frame = process  # type: ignore[method-assign]
        released = threading.Event()

        def release_tracker() -> None:
            runtime._tracker = None
            released.set()

        runtime._release_tracker_for_current_thread = release_tracker  # type: ignore[method-assign]
        await runtime.submit_live_frame(_frame(0, 0))
        assert await asyncio.to_thread(started.wait, 2)
        claim = asyncio.create_task(runtime.set_offline_processing(True))
        await asyncio.sleep(0)
        assert not claim.done()

        release.set()
        await asyncio.wait_for(claim, timeout=2)
        status = await runtime.status()
        assert status["offline_processing"] is True
        assert status["state"] == "disabled"
        assert status["live_inferences"] == 0
        assert status["latest_result"] is None
        assert released.is_set()
        await runtime.close()

    asyncio.run(scenario())


def test_status_events_push_initial_snapshot_and_completed_inference(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = HandTrackingRuntime(
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
