import asyncio

import pytest

from egoglass_operator_console.models import RuntimeSettings, SessionPhase
from egoglass_operator_console.runtime import ConsoleRuntime, build_telemetry


def test_simulated_trajectory_is_deterministic_and_in_camera_optical_frame() -> None:
    settings = RuntimeSettings()
    first = build_telemetry(
        tick=12,
        settings=settings,
        session_id="test",
        session_phase=SessionPhase.LIVE,
        recording=False,
        calibration=asyncio.run(ConsoleRuntime().state()).calibration,
        now_unix_ns=123,
    )
    second = build_telemetry(
        tick=12,
        settings=settings,
        session_id="test",
        session_phase=SessionPhase.LIVE,
        recording=False,
        calibration=asyncio.run(ConsoleRuntime().state()).calibration,
        now_unix_ns=123,
    )

    assert first == second
    assert first.calibration.coordinate_frame == "camera_optical"
    assert {hand.side for hand in first.hands} == {"left", "right"}
    assert all(len(hand.waypoints) == settings.prediction_steps for hand in first.hands)
    assert all(point.z_m > 0 for hand in first.hands for point in hand.waypoints)


@pytest.mark.parametrize("active", [True, False])
def test_session_transition_controls_recording(active: bool) -> None:
    runtime = ConsoleRuntime()
    state = asyncio.run(runtime.set_session_active(active))

    assert (state.session_phase is SessionPhase.LIVE) is active
    if not active:
        assert state.recording is False


def test_recording_requires_live_session() -> None:
    runtime = ConsoleRuntime()
    asyncio.run(runtime.set_session_active(False))

    with pytest.raises(RuntimeError, match="recording requires a live session"):
        asyncio.run(runtime.set_recording(True))


def test_non_integer_sampling_ratio_preserves_source_frame_rate() -> None:
    settings = RuntimeSettings(capture_fps=15, inference_fps=10)
    snapshot = asyncio.run(ConsoleRuntime().telemetry(0))
    snapshot = build_telemetry(
        tick=10,
        settings=settings,
        session_id="ratio-test",
        session_phase=SessionPhase.LIVE,
        recording=False,
        calibration=snapshot.calibration,
        now_unix_ns=123,
    )

    assert snapshot.frame_seq == 15
    assert snapshot.captured_at_sdk_ms == 1000
    assert snapshot.received_at_perf_counter_ns == 1_000_000_000
