import asyncio
from pathlib import Path

from av import VideoFrame

from perception.runtime import HandTrackingRuntime, LiveHandTrackingFrame
from perception.sensor_preprocessing import (
    ClockId,
    TimeObservation,
    TimestampSemantic,
    client_perf_source_instance_id,
)

REPOSITORY = Path(__file__).parents[1]


def _observation(frame: LiveHandTrackingFrame) -> TimeObservation:
    return TimeObservation(
        session_id=frame.session_id,
        source_clock_id=ClockId.CLIENT_PERF_COUNTER_NS,
        source_instance_id=client_perf_source_instance_id(
            frame.session_id,
            frame.connection_session_id,
        ),
        source_timestamp=frame.received_at_client_monotonic_ns,
        timestamp_semantic=TimestampSemantic.CLIENT_RECEIPT,
    )


def _frame(connection_id: str, received_at_ns: int) -> LiveHandTrackingFrame:
    return LiveHandTrackingFrame(
        session_id="a" * 32,
        connection_session_id=connection_id,
        frame_index=0,
        received_at_client_monotonic_ns=received_at_ns,
        decoded_frame=VideoFrame(8, 6, "yuv420p"),
    )


def test_each_live_connection_starts_a_nonnegative_session_timeline(tmp_path: Path) -> None:
    runtime = HandTrackingRuntime(
        recordings_root=tmp_path / "recordings",
        runtime_config_path=REPOSITORY / "config" / "perception-runtime.yaml",
        sensor_config_path=REPOSITORY / "config" / "sensor-preprocessing.yaml",
    )
    first = _frame("b" * 32, 9_000_000_000_000)
    later = _frame("b" * 32, first.received_at_client_monotonic_ns + 16_666_667)
    reconnected = _frame("c" * 32, 4_000_000_000)

    try:
        first_pipeline = runtime._live_preprocessing_for(first)
        assert first_pipeline.clock_mapper.map(_observation(first)).session_time_ns == 0
        assert first_pipeline.clock_mapper.map(_observation(later)).session_time_ns == 16_666_667

        reconnected_pipeline = runtime._live_preprocessing_for(reconnected)
        assert reconnected_pipeline is not first_pipeline
        assert reconnected_pipeline.clock_mapper.map(_observation(reconnected)).session_time_ns == 0
    finally:
        asyncio.run(runtime.close())
