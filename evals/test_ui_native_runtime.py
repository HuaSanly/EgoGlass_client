import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QRect
from PyQt6.QtWidgets import QApplication
from pyqtgraph.opengl import GLViewWidget
from qfluentwidgets import BodyLabel, IconInfoBadge

from ui.app import MainWindow
from ui.application.runtime_state import RuntimeSnapshot
from ui.gateway.live_frames import LiveFrame, LiveFramePacer
from ui.widgets.spatial_sync_canvas import SpatialSyncCanvas
from ui.widgets.status_indicator import StatusIndicator
from ui.widgets.video_canvas import VideoCanvas


def test_native_runtime_uses_direct_frames_and_one_process() -> None:
    repository = Path(__file__).parents[1]
    runtime = (repository / "ui" / "application" / "runtime_host.py").read_text(
        encoding="utf-8"
    )
    video = (repository / "ui" / "widgets" / "video_canvas.py").read_text(
        encoding="utf-8"
    )
    home = (repository / "ui" / "views" / "home.py").read_text(encoding="utf-8")
    processing = (repository / "ui" / "views" / "video_processing.py").read_text(
        encoding="utf-8"
    )

    assert "threading.Thread" in runtime
    assert "uvicorn.Server" in runtime
    assert "create_app(" in runtime
    assert "run_coroutine_threadsafe" in runtime
    assert "requests" not in runtime
    assert "httpx" not in runtime
    assert "subprocess" not in runtime
    assert "multiprocessing" not in runtime
    assert "QImage.Format.Format_RGB888" in video
    assert "QPainter" in video
    assert "source.subscribe(buffered=False)" in (
        repository / "ui" / "gateway" / "adapters" / "aiortc_peer.py"
    ).read_text(encoding="utf-8")
    assert "jpeg" not in video.lower()
    assert "mjpg" not in video.lower()
    assert "library_refresh_at_ns" not in runtime
    assert "library_task = asyncio.create_task(self._initial_library_refresh())" in runtime
    assert "request_library_refresh" in processing
    assert "VideoHall" in processing
    assert "VideoThumbnailService" in processing
    assert "self.replay.open_session" in processing
    assert "SegmentedWidget" not in home
    assert "HeaderCardWidget" in home
    assert 'TitleLabel("EgoGlass")' in home
    assert "感知工作台" not in home
    assert "眼镜实时感知、采集与离线回放" not in home


def test_native_video_path_has_rgb_fanout_and_bounded_latest_overlay() -> None:
    repository = Path(__file__).parents[1]
    live_frames = (repository / "ui" / "gateway" / "live_frames.py").read_text(
        encoding="utf-8"
    )
    video = (repository / "ui" / "widgets" / "video_canvas.py").read_text(
        encoding="utf-8"
    )
    home = (repository / "ui" / "views" / "home.py").read_text(encoding="utf-8")

    assert "submit_rgb_frame" in live_frames
    assert "self._frame = frame" in video
    assert "maximum_overlay_age_frames: int = 18" in video
    assert "frame_age > self._maximum_overlay_age_frames" in video
    assert "frame_index" in home
    assert "recent_presentation_fps" in video
    assert "LiveFramePacer" in live_frames
    assert "maximum_queue_frames: int = 4" in live_frames
    assert "video_pts_ns" in live_frames
    assert "next_for_display" in (
        repository / "ui" / "application" / "runtime_host.py"
    ).read_text(encoding="utf-8")
    assert "_forward_perception_results" in (
        repository / "ui" / "application" / "runtime_host.py"
    ).read_text(encoding="utf-8")
    assert "take_latest_perception_result" in home
    assert "self._frame_timer.setInterval(16)" in home


def test_live_overlay_survives_normal_inference_frame_delay(
    qt_application: QApplication,
) -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image.setflags(write=False)
    canvas = VideoCanvas()
    canvas.set_frame(LiveFrame("session", "connection", 100, 0, 0, image))
    canvas.set_overlay(
        {
            "session_id": "session",
            "sequence_id": "connection",
            "frame_index": 92,
            "source_image_width_px": 640,
            "source_image_height_px": 480,
            "hands": [
                {
                    "handedness": "left",
                    "source_keypoints_2d_px": [],
                    "source_bbox_xyxy_px": [100, 100, 200, 200],
                }
            ],
        }
    )

    assert canvas.status().overlay_visible
    assert canvas.status().overlay_frame_age == 8
    canvas.set_frame(LiveFrame("session", "connection", 110, 0, 0, image))
    assert canvas.status().overlay_visible
    canvas.set_frame(LiveFrame("session", "connection", 111, 0, 0, image))
    assert not canvas.status().overlay_visible
    qt_application.processEvents()


def test_spatial_sync_is_a_flat_opengl_sidebar_surface() -> None:
    repository = Path(__file__).parents[1]
    source = (repository / "ui" / "views" / "home.py").read_text(encoding="utf-8")
    spatial_source = (
        repository / "ui" / "widgets" / "spatial_sync_canvas.py"
    ).read_text(encoding="utf-8")
    indicator_source = (
        repository / "ui" / "widgets" / "status_indicator.py"
    ).read_text(encoding="utf-8")
    project = (repository / "pyproject.toml").read_text(encoding="utf-8")

    assert 'HeaderCardWidget("空间同步"' not in source
    assert 'HeaderCardWidget("同步数据"' in source
    assert "SimpleCardWidget" in source
    assert "frameLinkStrip" in source
    assert "syncMetricBlock" not in source
    assert "_sync_metric_block" not in source
    assert "link_sync_badge" not in source
    assert "InfoBadge(" not in source
    assert "font-size:" not in source
    assert "IconInfoBadge" in indicator_source
    assert "BodyLabel" in indicator_source
    assert "QLabel" not in indicator_source
    assert "scroll.viewport().setStyleSheet(\"background: transparent;\")" in source
    assert "GLViewWidget" in spatial_source
    assert "GLMeshItem" not in spatial_source
    assert "_camera_points_to_scene" in spatial_source
    assert "_hand_to_scene" not in spatial_source
    presentation_source = (
        repository / "ui" / "presentation" / "spatial_scene.py"
    ).read_text(encoding="utf-8")
    assert "keypoints_3d_camera_m" in presentation_source
    assert "imuPoseResetButton" in spatial_source
    assert "QColor(\"#f8fafc\")" in spatial_source
    assert "rgba(255, 255, 255, 0.14)" in spatial_source
    assert "reset_pose_requested.connect(self.runtime.request_imu_pose_reset)" in source
    assert "QPainter" not in spatial_source
    assert "StrongBodyLabel" in spatial_source
    assert "CaptionLabel" in spatial_source
    assert "QLabel" not in spatial_source
    assert '"PyOpenGL==3.1.0"' in project
    assert '"pyqtgraph==0.14.0"' in project


def test_fluent_home_renders_without_overlap_at_supported_sizes(
    qt_application: QApplication,
) -> None:
    runtime = SimpleNamespace(
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
    window = MainWindow(runtime)  # type: ignore[arg-type]
    try:
        for width, height in ((1280, 800), (1440, 900), (1920, 1080)):
            window.resize(width, height)
            window.show()
            window.switchTo(window.home_view)
            qt_application.processEvents()
            canvas = window.home_view.canvas.canvas_geometry()
            assert canvas.width > 0
            assert canvas.height > 0
            assert abs(canvas.width / canvas.height - 4 / 3) < 1e-9
            canvas_rect = QRect(
                window.home_view.canvas.mapTo(window, QPoint(0, 0)),
                window.home_view.canvas.size(),
            )
            sidebar_rect = QRect(
                window.home_view.sidebar.mapTo(window, QPoint(0, 0)),
                window.home_view.sidebar.size(),
            )
            frame_strip_rect = QRect(
                window.home_view.frame_link_strip.mapTo(window, QPoint(0, 0)),
                window.home_view.frame_link_strip.size(),
            )
            assert not canvas_rect.intersects(sidebar_rect)
            assert not frame_strip_rect.intersects(sidebar_rect)
            assert frame_strip_rect.left() == canvas_rect.left()
            assert frame_strip_rect.right() == canvas_rect.right()
            assert frame_strip_rect.top() > canvas_rect.bottom()
            spatial = window.home_view.findChild(SpatialSyncCanvas)
            assert spatial is not None
            assert spatial.findChild(GLViewWidget, "spatialSyncViewport") is not None
            indicators = window.home_view.findChildren(StatusIndicator)
            assert len(indicators) == 11
            assert all(indicator.findChild(IconInfoBadge) for indicator in indicators)
            assert all(indicator.findChild(BodyLabel) for indicator in indicators)
            spatial_rect = QRect(
                spatial.mapTo(window, QPoint(0, 0)),
                spatial.size(),
            )
            sync_card = window.home_view.sync_data_card
            sync_card_rect = QRect(
                sync_card.mapTo(window, QPoint(0, 0)),
                sync_card.size(),
            )
            assert spatial_rect.left() == sync_card_rect.left()
            assert spatial_rect.right() == sync_card_rect.right()
            assert not window.grab().isNull()
    finally:
        window.close()
        qt_application.processEvents()


@pytest.mark.skipif(
    os.environ.get("EGOGLASS_RUN_UI_SOAK") != "1",
    reason="set EGOGLASS_RUN_UI_SOAK=1 for the 30-second Qt presentation eval",
)
def test_four_by_three_canvas_sustains_thirty_fps_for_thirty_seconds(
    qt_application: QApplication,
) -> None:
    canvas = VideoCanvas()
    canvas.resize(960, 720)
    canvas.show()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image.setflags(write=False)
    started = time.perf_counter()
    frame_index = 0
    presentation_times_ns: list[int] = []
    while time.perf_counter() - started < 30:
        target = started + frame_index / 30
        remaining = target - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        now_ns = time.perf_counter_ns()
        presentation_times_ns.append(now_ns)
        canvas.set_frame(
            LiveFrame("soak", "soak", frame_index, now_ns, now_ns, image)
        )
        canvas.grab()
        qt_application.processEvents()
        frame_index += 1

    status = canvas.status()
    presentation_gaps_ms = np.diff(presentation_times_ns) / 1_000_000
    assert status.recent_presentation_fps >= 28
    assert status.latest_paint_ms is not None and status.latest_paint_ms < 100
    assert np.max(presentation_gaps_ms) < 100


def test_pts_pacer_smooths_bursty_lan_arrivals_without_unbounded_latency() -> None:
    frame_interval_ns = 33_333_333
    render_interval_ns = 16_666_667
    arrival_gaps_ns = (12_000_000, 54_666_666, 29_000_000, 37_666_666)
    arrival_ns = 0
    arrivals: list[tuple[int, LiveFrame]] = []
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    image.setflags(write=False)
    for index in range(180):
        if index:
            arrival_ns += arrival_gaps_ns[(index - 1) % len(arrival_gaps_ns)]
        arrivals.append(
            (
                arrival_ns,
                LiveFrame(
                    session_id="session",
                    connection_session_id="connection",
                    frame_index=index,
                    received_at_client_monotonic_ns=arrival_ns,
                    converted_at_client_monotonic_ns=arrival_ns,
                    image_rgb=image,
                    video_pts_ns=index * frame_interval_ns,
                ),
            )
        )

    pacer = LiveFramePacer()
    presented: list[tuple[int, LiveFrame]] = []
    arrival_index = 0
    previous_index = -1
    now_ns = 0
    deadline_ns = arrivals[-1][0] + 500_000_000
    maximum_queue_depth = 0
    while now_ns <= deadline_ns:
        while arrival_index < len(arrivals) and arrivals[arrival_index][0] <= now_ns:
            pacer.enqueue(arrivals[arrival_index][1])
            arrival_index += 1
        frame = pacer.next_frame(now_ns)
        status = pacer.status()
        maximum_queue_depth = max(maximum_queue_depth, status.queue_depth)
        if frame is not None and frame.frame_index != previous_index:
            presented.append((now_ns, frame))
            previous_index = frame.frame_index
        now_ns += render_interval_ns

    presentation_gaps_ms = np.diff([time_ns for time_ns, _ in presented]) / 1_000_000
    presentation_latency_ms = np.asarray(
        [
            (time_ns - frame.video_pts_ns) / 1_000_000
            for time_ns, frame in presented
            if frame.video_pts_ns is not None
        ]
    )
    assert len(presented) >= 170
    assert maximum_queue_depth <= 4
    assert np.percentile(presentation_gaps_ms, 95) <= 50.1
    assert np.max(presentation_latency_ms) <= 120.0


def test_pts_pacer_does_not_systematically_drop_batched_rtp_frames() -> None:
    frame_interval_ns = 33_333_333
    render_interval_ns = 10_000_000
    arrival_gaps_ns = (100_000_000, 0, 0, 0, 66_666_665)
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    image.setflags(write=False)
    arrivals: list[tuple[int, LiveFrame]] = []
    arrival_ns = 0
    for index in range(180):
        if index:
            arrival_ns += arrival_gaps_ns[(index - 1) % len(arrival_gaps_ns)]
        arrivals.append(
            (
                arrival_ns,
                LiveFrame(
                    session_id="session",
                    connection_session_id="connection",
                    frame_index=index,
                    received_at_client_monotonic_ns=arrival_ns,
                    converted_at_client_monotonic_ns=arrival_ns,
                    image_rgb=image,
                    video_pts_ns=index * frame_interval_ns,
                ),
            ),
        )

    pacer = LiveFramePacer()
    presented_indices: list[int] = []
    arrival_index = 0
    previous_index = -1
    now_ns = 0
    deadline_ns = arrivals[-1][0] + 500_000_000
    while now_ns <= deadline_ns:
        while arrival_index < len(arrivals) and arrivals[arrival_index][0] <= now_ns:
            pacer.enqueue(arrivals[arrival_index][1])
            arrival_index += 1
        frame = pacer.next_frame(now_ns)
        if frame is not None and frame.frame_index != previous_index:
            presented_indices.append(frame.frame_index)
            previous_index = frame.frame_index
        now_ns += render_interval_ns

    status = pacer.status(now_ns)
    assert len(presented_indices) >= 178
    assert status.frames_dropped <= 2
    assert status.starvations <= 1
