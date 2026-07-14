from egoglass_operator_console.models import RuntimeSettings, SessionPhase
from egoglass_operator_console.runtime import ConsoleRuntime, build_telemetry


def test_default_simulated_session_meets_operator_console_quality_budget() -> None:
    settings = RuntimeSettings()
    calibration = __import__("asyncio").run(ConsoleRuntime().state()).calibration
    snapshots = [
        build_telemetry(
            tick=tick,
            settings=settings,
            session_id="eval-session",
            session_phase=SessionPhase.LIVE,
            recording=tick >= 10,
            calibration=calibration,
            now_unix_ns=1_000_000_000 + tick,
        )
        for tick in range(120)
    ]

    assert all(len(snapshot.hands) == 2 for snapshot in snapshots)
    assert all(hand.present for snapshot in snapshots for hand in snapshot.hands)
    assert all(
        len(hand.waypoints) == settings.prediction_steps
        for snapshot in snapshots
        for hand in snapshot.hands
    )
    assert all(
        point.z_m > 0.4 and abs(point.x_m) < 0.5 and abs(point.y_m) < 0.5
        for snapshot in snapshots
        for hand in snapshot.hands
        for point in hand.waypoints
    )
    assert all(
        [point.t_offset_ms for point in hand.waypoints]
        == sorted({point.t_offset_ms for point in hand.waypoints})
        for snapshot in snapshots
        for hand in snapshot.hands
    )
    assert max(snapshot.metrics.feedback_latency_ms for snapshot in snapshots) < 500
    assert max(snapshot.metrics.queue_depth for snapshot in snapshots) == 0
    assert snapshots[-1].metrics.dropped_frames / snapshots[-1].frame_seq < 0.02


def test_configurable_sampling_rate_keeps_frame_and_clock_alignment() -> None:
    settings = RuntimeSettings(capture_fps=15, inference_fps=10)
    calibration = __import__("asyncio").run(ConsoleRuntime().state()).calibration
    snapshots = [
        build_telemetry(
            tick=tick,
            settings=settings,
            session_id="sampling-eval",
            session_phase=SessionPhase.LIVE,
            recording=False,
            calibration=calibration,
            now_unix_ns=1_000_000_000 + tick,
        )
        for tick in range(11)
    ]

    assert snapshots[-1].frame_seq == 15
    assert snapshots[-1].captured_at_sdk_ms == 1000
    assert all(
        current.frame_seq >= previous.frame_seq
        for previous, current in zip(snapshots[:-1], snapshots[1:], strict=True)
    )
