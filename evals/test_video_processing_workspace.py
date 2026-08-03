from __future__ import annotations

import os
import time
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QRect
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QWidget

from perception.video_processing import (
    ProcessingJob,
    ProcessingJobState,
    ProcessingPreset,
    ProcessingRunInfo,
    ProcessingRunState,
)
from ui.app import MainWindow
from ui.replay.player import PlaybackClipSpan, PlaybackFrame, ReplaySnapshot, ReplayState
from ui.state import RuntimeSnapshot
from ui.video_processing.hall import VideoHall
from ui.views.video_processing import _processing_states, _Selection
from ui.widgets.spatial_sync_canvas import SpatialSyncCanvas
from ui.widgets.video_canvas import VideoCanvas


def _runtime() -> SimpleNamespace:
    empty_runs: Future[tuple[ProcessingRunInfo, ...]] = Future()
    empty_runs.set_result(())

    def no_result(*_values: object) -> Future[dict[str, object] | None]:
        future: Future[dict[str, object] | None] = Future()
        future.set_result(None)
        return future

    return SimpleNamespace(
        snapshot=lambda: RuntimeSnapshot(),
        latest_frame=lambda: None,
        take_latest_perception_result=lambda: None,
        command_results=lambda: (),
        request_library_refresh=lambda: None,
        request_session=lambda _action: None,
        request_stream=lambda _action: None,
        request_recording=lambda _action: None,
        request_imu_pose_reset=lambda: None,
        request_processing=lambda *_args, **_kwargs: None,
        request_processing_cancel=lambda _job_id: None,
        request_processing_retry=lambda _job_id: None,
        request_processing_export=lambda *_args: None,
        request_processing_auto_queue=lambda _enabled: None,
        request_processing_default_preset=lambda _preset_id: None,
        request_live_inference=lambda _enabled: None,
        processing_runs=lambda _session_id: empty_runs,
        processing_result=no_result,
        session_directory=lambda _session_id: Path("missing-session"),
        stop=lambda: None,
    )


def _run(run_id: str, completed_at: int, *, clip_id: str | None = None) -> ProcessingRunInfo:
    return ProcessingRunInfo(
        run_id=run_id,
        session_id="session",
        clip_id=clip_id,
        preset=ProcessingPreset(),
        state=ProcessingRunState.COMPLETED,
        input_frame_count=50,
        inferred_frame_count=48,
        detected_hand_count=30,
        started_at_unix_ns=1,
        completed_at_unix_ns=completed_at,
        results_path=Path("results.sqlite"),
        is_viewable=True,
    )


def test_workbench_keeps_video_and_space_visible_at_supported_sizes(
    qt_application: QApplication,
) -> None:
    window = MainWindow(_runtime())  # type: ignore[arg-type]
    try:
        view = window.processing_view
        view.stack.setCurrentWidget(view.workbench)
        view.workbench.set_context("session", "布局评估", "clip")
        for width, height in ((1280, 800), (1440, 900), (1920, 1080)):
            window.resize(width, height)
            window.show()
            qt_application.processEvents()
            video_rect = QRect(view.canvas.mapTo(window, QPoint(0, 0)), view.canvas.size())
            spatial_rect = QRect(
                view.spatial_canvas.mapTo(window, QPoint(0, 0)),
                view.spatial_canvas.size(),
            )
            geometry = view.canvas.canvas_geometry()

            assert view.canvas.isVisible()
            assert view.spatial_canvas.isVisible()
            assert not video_rect.intersects(spatial_rect)
            assert abs(geometry.width / geometry.height - 4 / 3) < 1e-9
            assert len(view.workbench.findChildren(QWidget, "readonlySlicePlaceholder")) == 2
            assert not window.grab().isNull()
    finally:
        window.close()
        qt_application.processEvents()


def test_video_hall_cards_fill_rows_before_wrapping(
    qt_application: QApplication,
) -> None:
    from ingest_gateway.recording_models import CaptureSessionState

    def clip(index: int) -> SimpleNamespace:
        return SimpleNamespace(
            clip_id=f"clip-{index}",
            recorded_at_unix_ms=1_700_000_000_000 + index,
            duration_ms=3_000,
            width=640,
            height=480,
            fps=30,
            frame_count=90,
            file_size_bytes=4096,
        )

    hall = VideoHall()
    hall.resize(1100, 760)
    hall.set_library(
        SimpleNamespace(
            sessions=[
                SimpleNamespace(
                    session_id="session-a",
                    display_name="会话 A",
                    state=CaptureSessionState.COMPLETE,
                    clips=[clip(1)],
                ),
                SimpleNamespace(
                    session_id="session-b",
                    display_name="会话 B",
                    state=CaptureSessionState.COMPLETE,
                    clips=[clip(2)],
                ),
            ]
        )
    )
    hall.show()
    qt_application.processEvents()
    first, second = tuple(hall.cards.values())

    assert first.y() == second.y()
    assert second.x() > first.x()
    assert first.thumbnail.geometry().bottom() < first.session_label.geometry().top()
    assert first.session_label.geometry().bottom() < first.primary_meta.geometry().top()
    assert "#f7f9fc" in hall.scroll.viewport().styleSheet()
    original_position = first.pos()
    QTest.mouseMove(first, first.rect().center())
    QTest.qWait(150)
    assert first.pos() == original_position
    hall.resize(600, 760)
    qt_application.processEvents()
    assert second.y() > first.y()
    hall.close()


def test_result_switching_does_not_reopen_or_duplicate_video_decoder() -> None:
    window = MainWindow(_runtime())  # type: ignore[arg-type]
    calls: list[tuple[object, ...]] = []
    try:
        view = window.processing_view
        view.replay.open_session = (  # type: ignore[method-assign]
            lambda *values: calls.append(values)
        )
        view.workbench.set_runs((_run("new", 3), _run("old", 2)), "clip")
        view.workbench.result_combo.setCurrentIndex(0)
        view.workbench.result_combo.setCurrentIndex(1)
        view.workbench.comparison_combo.setCurrentIndex(1)

        assert calls == []
        assert view.findChildren(VideoCanvas) == [view.canvas]
        assert view.findChildren(SpatialSyncCanvas) == [view.spatial_canvas]
    finally:
        window.close()


def test_slow_result_store_queries_do_not_create_a_per_frame_backlog() -> None:
    runtime = _runtime()
    pending: Future[dict[str, object] | None] = Future()
    calls: list[tuple[object, ...]] = []

    def slow_query(*values: object) -> Future[dict[str, object] | None]:
        calls.append(values)
        return pending

    runtime.processing_result = slow_query
    window = MainWindow(runtime)  # type: ignore[arg-type]
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image.setflags(write=False)
    try:
        view = window.processing_view
        view.stack.setCurrentWidget(view.workbench)
        view.workbench.set_runs((_run("run", 2),), "clip")
        for index in range(60):
            view._query_results(
                PlaybackFrame("session", "clip", index, index, index, image)
            )

        assert len(calls) == 1
        assert len(view._result_futures) == 1
        pending.set_result(None)
    finally:
        window.close()


def test_session_clip_timeline_uses_real_spans_and_emits_seek() -> None:
    window = MainWindow(_runtime())  # type: ignore[arg-type]
    seeks: list[float] = []
    try:
        timeline = window.processing_view.workbench.clip_timeline
        timeline.seekRequested.connect(seeks.append)
        timeline.set_clips(
            (
                PlaybackClipSpan("clip-a", 0.0, 2.0, 60),
                PlaybackClipSpan("clip-b", 2.0, 5.5, 105),
            )
        )
        timeline._buttons["clip-b"].click()

        assert seeks == [2.0]
        assert timeline._buttons["clip-b"].toolTip().startswith("105 帧")
    finally:
        window.close()


def test_session_wide_pipeline_state_is_visible_on_every_clip_card() -> None:
    session = SimpleNamespace(
        session_id="session",
        clips=[SimpleNamespace(clip_id="a"), SimpleNamespace(clip_id="b")],
    )
    job = ProcessingJob(
        "job",
        "session",
        None,
        ProcessingPreset(),
        ProcessingJobState.QUEUED,
        1,
        2,
    )

    assert _processing_states((job,), session) == {"a": "等待", "b": "等待"}


def test_processing_ab_layers_share_one_frame_identity_and_spatial_result(
    qt_application: QApplication,
) -> None:
    runtime = _runtime()
    calls: list[tuple[object, ...]] = []
    result = {
        "session_id": "session",
        "sequence_id": "clip",
        "frame_index": 42,
        "session_time_ns": 2_000_000_000,
        "source_image_width_px": 640,
        "source_image_height_px": 480,
        "hands": [
            {
                "handedness": "left",
                "source_keypoints_2d_px": [
                    [160 + index * 4, 120 + index * 3] for index in range(21)
                ],
                "keypoints_3d_camera_m": [
                    [index * 0.001, index * 0.002, 0.4] for index in range(21)
                ],
            }
        ],
    }

    def processing_result(*values: object) -> Future[dict[str, object]]:
        calls.append(values)
        future: Future[dict[str, object]] = Future()
        future.set_result(
            {
                **result,
                "frame_index": values[3],
                "session_time_ns": values[4],
            }
        )
        return future

    runtime.processing_result = processing_result
    window = MainWindow(runtime)  # type: ignore[arg-type]
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image.setflags(write=False)
    frame = PlaybackFrame("session", "clip", 42, 1_000_000_000, 2_000_000_000, image)
    try:
        view = window.processing_view
        view._selection = _Selection("session", "clip")
        view.stack.setCurrentWidget(view.workbench)
        view.workbench.set_runs((_run("run-a", 3), _run("run-b", 2)), "clip")
        view.workbench.comparison_combo.setCurrentIndex(1)
        view.replay._update(  # type: ignore[attr-defined]
            state=ReplayState.PAUSED,
            frame=frame,
            duration_seconds=3.0,
            position_seconds=2.0,
        )
        view._update_frame()
        view._update_frame()
        qt_application.processEvents()

        expected_key = ("session", "clip", 42, 2_000_000_000)
        assert [tuple(call[index] for index in (0, 2, 3, 4)) for call in calls] == [
            expected_key,
            expected_key,
        ]
        assert view.canvas.status().overlay_visible
        assert view.spatial_canvas.status().latest_frame_index == 42
        assert view.spatial_canvas.status().has_left_hand
    finally:
        window.close()
        qt_application.processEvents()


def test_slice_placeholders_have_no_annotation_or_file_write_path() -> None:
    source_root = Path(__file__).parents[1] / "ui"
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            source_root / "video_processing" / "workbench.py",
            source_root / "views" / "video_processing.py",
        )
    )

    assert "\nfrom annotation" not in sources
    assert "\nimport annotation" not in sources
    assert "write_text" not in sources
    assert "write_bytes" not in sources
    assert "只读占位" in sources


def test_replay_unload_contract_clears_media_identity() -> None:
    from ui.replay.player import ReplayPlayer

    player = ReplayPlayer()
    try:
        player._update(  # type: ignore[attr-defined]
            state=ReplayState.PAUSED,
            path=Path("session"),
            session_id="session",
            clip_id="clip",
            duration_seconds=3.0,
        )
        player.unload()
        deadline = time.monotonic() + 1.0
        while player.snapshot().state is not ReplayState.EMPTY:
            if time.monotonic() >= deadline:
                raise AssertionError("replay unload did not become empty")
            time.sleep(0.005)

        assert player.snapshot() == ReplaySnapshot(revision=player.snapshot().revision)
    finally:
        player.close()


def test_shutdown_failure_still_stops_runtime(
    qt_application: QApplication,
) -> None:
    runtime = _runtime()
    stopped: list[bool] = []
    runtime.stop = lambda: stopped.append(True)
    window = MainWindow(runtime)  # type: ignore[arg-type]
    original = window.processing_view.close_resources

    def fail_after_close() -> None:
        original()
        raise RuntimeError("simulated replay close failure")

    window.processing_view.close_resources = fail_after_close  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="simulated replay close failure"):
            window.shutdown()
        assert stopped == [True]
    finally:
        window.close()
        qt_application.processEvents()


@pytest.mark.skipif(
    os.environ.get("EGOGLASS_RUN_REPLAY_SOAK") != "1",
    reason="set EGOGLASS_RUN_REPLAY_SOAK=1 for the 30-second recording soak",
)
def test_existing_recording_plays_continuously_for_thirty_seconds(
    qt_application: QApplication,
) -> None:
    from ui.replay.player import ReplayPlayer

    root = Path(os.environ.get("EGOGLASS_RECORDINGS_ROOT", "local-data/recordings"))
    candidates = tuple(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".") and (path / "session.json").is_file()
    )
    if not candidates:
        pytest.skip("no local capture sessions are available")

    player: ReplayPlayer | None = None
    for candidate in candidates:
        current = ReplayPlayer()
        current.open_session(candidate)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            state = current.snapshot().state
            if state in {ReplayState.PAUSED, ReplayState.ERROR}:
                break
            time.sleep(0.01)
        if current.snapshot().state is ReplayState.PAUSED:
            player = current
            break
        current.close()
    if player is None:
        pytest.fail("none of the local recording sessions could be opened")

    canvas = VideoCanvas()
    canvas.resize(960, 720)
    canvas.show()
    indexed_duration = player.snapshot().duration_seconds
    player.play()
    started = time.monotonic()
    accepted_frames = 0
    loops = 0
    last_revision = -1
    try:
        while time.monotonic() - started < 30.0:
            snapshot = player.snapshot()
            if snapshot.revision != last_revision:
                last_revision = snapshot.revision
                if snapshot.frame is not None and canvas.set_frame(snapshot.frame):
                    accepted_frames += 1
            if snapshot.state is ReplayState.ENDED:
                loops += 1
                player.seek(0.0)
                player.play()
            if snapshot.state is ReplayState.ERROR:
                pytest.fail(snapshot.error or "recording playback failed")
            qt_application.processEvents()
            time.sleep(0.002)
    finally:
        player.close()
        canvas.close()

    assert time.monotonic() - started >= 30.0
    assert accepted_frames >= 300
    assert canvas.status().presented_frames >= 300
    assert loops >= 1 or indexed_duration >= 30.0
