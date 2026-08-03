from __future__ import annotations

import time
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QRect
from PyQt6.QtWidgets import QApplication, QWidget

from ui.app import MainWindow
from ui.replay.player import PlaybackFrame, ReplayState
from ui.state import RuntimeSnapshot
from ui.views.video_processing import _Selection
from ui.widgets.spatial_sync_canvas import SpatialSyncCanvas
from ui.widgets.video_canvas import VideoCanvas


def _runtime() -> SimpleNamespace:
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
        request_live_inference=lambda _enabled: None,
        processing_runs=lambda _session_id: None,
        processing_result=lambda *_args: None,
        session_directory=lambda _session_id: None,
        stop=lambda: None,
    )


def test_processing_workspace_keeps_video_and_space_visible_at_supported_sizes(
    qt_application: QApplication,
) -> None:
    window = MainWindow(_runtime())  # type: ignore[arg-type]
    try:
        for width, height in ((1280, 800), (1440, 900), (1920, 1080)):
            window.resize(width, height)
            window.show()
            qt_application.processEvents()
            view = window.processing_view
            video_rect = QRect(view.canvas.mapTo(window, QPoint(0, 0)), view.canvas.size())
            spatial_rect = QRect(
                view.spatial_canvas.mapTo(window, QPoint(0, 0)),
                view.spatial_canvas.size(),
            )
            inspector_rect = QRect(
                view.inspector_pivot.mapTo(window, QPoint(0, 0)),
                view.inspector_pivot.size(),
            )
            geometry = view.canvas.canvas_geometry()

            assert view.isVisible()
            assert view.canvas.isVisible()
            assert view.spatial_canvas.isVisible()
            assert not video_rect.intersects(spatial_rect)
            assert spatial_rect.top() < inspector_rect.top()
            assert abs(geometry.width / geometry.height - 4 / 3) < 1e-9
            assert not window.grab().isNull()
            assert view.findChild(QWidget, "processingTimeline") is None
            assert not hasattr(view, "timeline_bars")
    finally:
        window.close()
        qt_application.processEvents()


def test_processing_workspace_has_no_video_space_pivot() -> None:
    source = Path(__file__).parents[1] / "ui" / "views" / "video_processing.py"
    text = source.read_text(encoding="utf-8")

    assert 'addItem("video"' not in text
    assert 'addItem("space"' not in text
    assert "self.canvas" in text
    assert "self.spatial_canvas" in text
    assert "self.spatial_canvas.set_pose(snapshot.imu_pose)" in text


def test_replay_unload_contract_clears_media_identity() -> None:
    from ui.replay.player import ReplayPlayer, ReplaySnapshot

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


def test_processing_inspector_pages_remain_rendered_after_real_pivot_clicks(
    qt_application: QApplication,
) -> None:
    window = MainWindow(_runtime())  # type: ignore[arg-type]
    try:
        for width, height in ((1280, 800), (1440, 900), (1920, 1080)):
            window.resize(width, height)
            window.show()
            qt_application.processEvents()
            view = window.processing_view

            for key in ("preset", "layers", "frame"):
                view.inspector_pivot.items[key].click()
                qt_application.processEvents()
                page = view.inspector_pages[key]

                assert view.inspector_stack.currentWidget() is page
                assert page.isVisible()
                assert page.width() > 0
                assert page.height() > 0
                assert not page.grab().isNull()
    finally:
        window.close()
        qt_application.processEvents()


def test_processing_workspace_shutdown_failure_does_not_leave_runtime_running(
    qt_application: QApplication,
) -> None:
    runtime = _runtime()
    stopped: list[bool] = []
    runtime.stop = lambda: stopped.append(True)
    window = MainWindow(runtime)  # type: ignore[arg-type]
    processing_close = window.processing_view.close_resources

    def fail_after_replay_close() -> None:
        processing_close()
        raise RuntimeError("simulated replay close failure")

    window.processing_view.close_resources = fail_after_replay_close  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="simulated replay close failure"):
            window.shutdown()
        assert stopped == [True]
    finally:
        window.close()
        qt_application.processEvents()


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
                    [160 + index * 4, 120 + index * 3]
                    for index in range(21)
                ],
                "keypoints_3d_camera_m": [
                    [index * 0.001, index * 0.002, 0.4]
                    for index in range(21)
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
        view._selection = _Selection("session", None)
        view._run_ids = {"主结果": "run-a", "对比结果": "run-b"}
        view.primary_run_combo.addItem("主结果")
        view.primary_run_combo.setCurrentText("主结果")
        view.comparison_run_combo.addItem("对比结果")
        view.comparison_run_combo.setCurrentText("对比结果")
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
        assert view.canvas.status().latest_frame_index == 42
        assert view.canvas.status().overlay_visible
        assert view.spatial_canvas.status().latest_frame_index == 42
        assert view.spatial_canvas.status().has_left_hand
        assert view.findChildren(VideoCanvas) == [view.canvas]
        assert view.findChildren(SpatialSyncCanvas) == [view.spatial_canvas]

        backward_frame = PlaybackFrame(
            "session", "clip", 7, 200_000_000, 1_200_000_000, image
        )
        view.replay._update(  # type: ignore[attr-defined]
            frame=backward_frame,
            position_seconds=1.2,
        )
        view._update_frame()
        view._update_frame()
        qt_application.processEvents()

        backward_key = ("session", "clip", 7, 1_200_000_000)
        assert [tuple(call[index] for index in (0, 2, 3, 4)) for call in calls] == [
            expected_key,
            expected_key,
            backward_key,
            backward_key,
        ]
        assert view.canvas.status().latest_overlay_frame_index == 7
        assert view.canvas.status().overlay_visible
        assert view.spatial_canvas.status().latest_frame_index == 7
    finally:
        window.close()
        qt_application.processEvents()
