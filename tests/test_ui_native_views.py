from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import Future
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication
from pyqtgraph.opengl import GLViewWidget
from qfluentwidgets import HeaderCardWidget

from ingest_gateway.imu_preview import ImuPoseSnapshot
from ingest_gateway.live_frames import LiveFrame
from ingest_gateway.webrtc_models import StreamControlAction
from ui.app import MainWindow
from ui.state import RuntimeSnapshot
from ui.views.home import HomeView, ViewerMode, _confidence_text
from ui.widgets.spatial_sync_canvas import (
    SpatialSyncCanvas,
    _camera_points_to_scene,
    _glasses_frame_mesh,
)
from ui.widgets.video_canvas import VideoCanvas, fit_image_geometry


class RuntimeStub:
    def __init__(self) -> None:
        self.snapshot_value = RuntimeSnapshot()
        self.frame: LiveFrame | None = None
        self.stop_calls = 0
        self.refresh_calls = 0
        self.session_actions: list[str] = []
        self.stream_actions: list[StreamControlAction] = []
        self.recording_actions: list[str] = []
        self.replay_sessions: list[str] = []
        self.perception_result: dict[str, object] | None = None
        self.imu_pose_reset_calls = 0

    def snapshot(self) -> RuntimeSnapshot:
        return self.snapshot_value

    def latest_frame(self) -> LiveFrame | None:
        return self.frame

    def take_latest_perception_result(self) -> dict[str, object] | None:
        result, self.perception_result = self.perception_result, None
        return result

    def command_results(self) -> tuple[object, ...]:
        return ()

    def request_library_refresh(self) -> None:
        self.refresh_calls += 1

    def request_session(self, action: str) -> None:
        self.session_actions.append(action)

    def request_stream(self, action: StreamControlAction) -> None:
        self.stream_actions.append(action)

    def request_recording(self, action: str) -> None:
        self.recording_actions.append(action)

    def request_replay_generation(self, session_id: str) -> None:
        self.replay_sessions.append(session_id)

    def request_imu_pose_reset(self) -> None:
        self.imu_pose_reset_calls += 1

    def media_path(self, _session_id: str, _clip_id: str) -> Future[Path | None]:
        future: Future[Path | None] = Future()
        future.set_result(None)
        return future

    def replay_video_path(self, *_values: str) -> Future[Path]:
        future: Future[Path] = Future()
        future.set_exception(FileNotFoundError("missing replay"))
        return future

    def stop(self) -> None:
        self.stop_calls += 1


def _frame(
    index: int,
    *,
    width: int = 640,
    height: int = 480,
    connection_session_id: str = "connection",
) -> LiveFrame:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image.setflags(write=False)
    return LiveFrame(
        "session",
        connection_session_id,
        index,
        index,
        index,
        image,
    )


def _imu_pose() -> ImuPoseSnapshot:
    return ImuPoseSnapshot(
        session_id="session",
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        roll_degrees=0.0,
        pitch_degrees=0.0,
        yaw_degrees=0.0,
        accelerometer_mps2=(0.0, 9.8, 0.0),
        gyroscope_radps=(0.0, 0.0, 0.0),
        samples_received=120,
        samples_processed=120,
        queue_overflow_count=0,
        recent_rate_hz=100.0,
        latest_sample_age_ms=8.0,
    )


def _hand_result(
    *,
    include_right: bool = True,
    frame_index: int = 12,
    sequence_id: str = "connection",
) -> dict[str, object]:
    points = [[index * 0.01, index * 0.003, 0.35 + index * 0.002] for index in range(21)]
    hands: list[dict[str, object]] = [
        {
            "handedness": "left",
            "keypoints_3d_camera_m": points,
            "source_keypoints_2d_px": [],
            "source_bbox_xyxy_px": [100, 100, 200, 200],
            "detector_confidence": 0.9,
            "reconstruction_quality": 0.8,
            "depth_score": 0.7,
            "coverage_score": 0.6,
            "compactness_score": 0.5,
            "final_confidence": 0.4,
        }
    ]
    if include_right:
        hands.append({**hands[0], "handedness": "right"})
    return {
        "session_id": "session",
        "sequence_id": sequence_id,
        "frame_index": frame_index,
        "source_image_width_px": 640,
        "source_image_height_px": 480,
        "hands": hands,
    }


def test_fluent_window_registers_only_home_and_one_video_canvas(
    qt_application: QApplication,
) -> None:
    runtime = RuntimeStub()
    window = MainWindow(runtime)  # type: ignore[arg-type]
    try:
        window.show()
        qt_application.processEvents()

        assert window.stackedWidget.count() == 1
        assert window.stackedWidget.widget(0) is window.home_view
        assert len(window.findChildren(VideoCanvas)) == 1
        assert len(window.findChildren(SpatialSyncCanvas)) == 1
        assert not (Path(__file__).parents[1] / "ui/views/library.py").exists()
        assert not (Path(__file__).parents[1] / "ui/views/annotation.py").exists()
        assert not (Path(__file__).parents[1] / "ui/views/diagnostics.py").exists()
    finally:
        window.close()
        qt_application.processEvents()
    assert runtime.stop_calls == 1


def test_live_and_replay_modes_share_the_same_canvas(
    qt_application: QApplication,
) -> None:
    home = HomeView(RuntimeStub())  # type: ignore[arg-type]
    try:
        original_canvas = home.canvas
        home.set_viewer_mode(ViewerMode.REPLAY)
        assert home.canvas is original_canvas
        assert home.replay_controls.isVisible() is False
        home.show()
        qt_application.processEvents()
        assert home.replay_controls.isVisible()
        assert home.mode_badge.text() == "来源 · 回放"
        home.set_viewer_mode(ViewerMode.LIVE)
        assert home.canvas is original_canvas
        assert not home.replay_controls.isVisible()
        assert home.mode_badge.text() == "来源 · 实时"
    finally:
        home.close_resources()


def test_four_by_three_canvas_fills_a_four_by_three_area() -> None:
    geometry = fit_image_geometry(960, 720, 640, 480)

    assert geometry.minimum == (0.0, 0.0)
    assert geometry.maximum == (960.0, 720.0)
    assert geometry.scale == 1.5


def test_non_four_by_three_source_is_letterboxed_without_crop_or_stretch() -> None:
    geometry = fit_image_geometry(960, 720, 1280, 720)

    assert geometry.minimum == (0.0, 90.0)
    assert geometry.maximum == (960.0, 630.0)
    assert geometry.scale == 0.75


def test_video_canvas_keeps_rgb_frame_alive_and_paints_it(
    qt_application: QApplication,
) -> None:
    canvas = VideoCanvas()
    canvas.resize(960, 720)
    frame = _frame(3)

    assert canvas.set_frame(frame)
    pixmap = canvas.grab()
    qt_application.processEvents()

    assert pixmap.size().width() == 960
    assert pixmap.size().height() == 720
    assert canvas._frame is frame
    assert canvas.status().presented_frames == 1
    assert canvas.status().latest_frame_index == 3


def test_overlay_reuses_a_recent_same_stream_result_with_a_finite_age(
    qt_application: QApplication,
) -> None:
    canvas = VideoCanvas(maximum_overlay_age_frames=3)
    canvas.resize(960, 720)
    canvas.set_frame(_frame(10))
    assert canvas.set_overlay(_hand_result(include_right=False, frame_index=9))
    recent = canvas.grab().toImage().pixelColor(150, 150)
    recent_status = canvas.status()

    canvas.set_frame(_frame(13))
    expired = canvas.grab().toImage().pixelColor(150, 150)
    expired_status = canvas.status()
    qt_application.processEvents()

    assert recent.green() > recent.red()
    assert recent_status.overlay_visible
    assert recent_status.overlay_frame_age == 1
    assert expired != recent
    assert not expired_status.overlay_visible
    assert expired_status.overlay_frame_age is None


def test_overlay_rejects_other_streams_and_clears_on_empty_result(
    qt_application: QApplication,
) -> None:
    canvas = VideoCanvas()
    canvas.resize(960, 720)
    canvas.set_frame(_frame(20))
    assert canvas.set_overlay(_hand_result(include_right=False, frame_index=19))
    visible = canvas.grab().toImage().pixelColor(150, 150)

    assert not canvas.set_overlay(
        _hand_result(
            include_right=False,
            frame_index=20,
            sequence_id="previous-connection",
        )
    )
    assert canvas.status().overlay_visible
    empty = _hand_result(include_right=False, frame_index=20)
    empty["hands"] = []
    assert canvas.set_overlay(empty)
    cleared = canvas.grab().toImage().pixelColor(150, 150)
    qt_application.processEvents()

    assert visible.green() > visible.red()
    assert cleared != visible
    assert not canvas.status().overlay_visible


def test_overlay_waits_for_a_future_result_frame_and_clears_on_reconnect(
    qt_application: QApplication,
) -> None:
    canvas = VideoCanvas()
    canvas.set_frame(_frame(30))
    assert canvas.set_overlay(_hand_result(include_right=False, frame_index=31))
    assert not canvas.status().overlay_visible

    canvas.set_frame(_frame(31))
    assert canvas.status().overlay_visible

    canvas.set_frame(_frame(1, connection_session_id="new-connection"))
    assert not canvas.status().overlay_visible
    assert canvas.status().latest_overlay_frame_index is None
    qt_application.processEvents()


def test_home_controls_call_runtime_commands() -> None:
    runtime = RuntimeStub()
    home = HomeView(runtime)  # type: ignore[arg-type]
    try:
        home.stream_button.setProperty("action", "stop")
        home.recording_button.setProperty("action", "start")
        home._toggle_stream()
        home._toggle_recording()
        home.refresh_button.click()
        home.session_button.click()
        home.spatial_canvas.reset_pose_button.click()

        assert runtime.stream_actions == [StreamControlAction.STOP]
        assert runtime.recording_actions == ["start"]
        assert runtime.refresh_calls == 1
        assert runtime.session_actions == ["new"]
        assert runtime.imu_pose_reset_calls == 1
    finally:
        home.close_resources()


def test_home_frame_loop_consumes_pushed_perception_result() -> None:
    runtime = RuntimeStub()
    runtime.frame = _frame(100)
    runtime.perception_result = _hand_result(include_right=False, frame_index=92)
    home = HomeView(runtime)  # type: ignore[arg-type]
    try:
        home._update_frame()

        status = home.canvas.status()
        assert status.latest_frame_index == 100
        assert status.latest_overlay_frame_index == 92
        assert status.overlay_frame_age == 8
        assert status.overlay_visible
        assert runtime.perception_result is None
    finally:
        home.close_resources()


def test_capture_controls_share_the_mode_bar_not_the_live_sidebar() -> None:
    home = HomeView(RuntimeStub())  # type: ignore[arg-type]
    try:
        sidebar_widgets = set(home.sidebar.findChildren(object))
        assert home.stream_button not in sidebar_widgets
        assert home.recording_button not in sidebar_widgets
        assert home.session_button not in sidebar_widgets
        assert home.findChild(type(home.stream_button), "streamControlButton") is home.stream_button
        assert len(home.findChildren(SpatialSyncCanvas)) == 1
        assert home.spatial_canvas.reset_pose_button.objectName() == "imuPoseResetButton"
        assert not isinstance(home.spatial_canvas.parent(), HeaderCardWidget)
        assert home.spatial_canvas.findChild(GLViewWidget, "spatialSyncViewport") is not None
        assert "background: transparent" in home.sidebar.viewport().styleSheet()

        source = (Path(__file__).parents[1] / "ui" / "views" / "home.py").read_text(
            encoding="utf-8"
        )
        assert 'HeaderCardWidget("采集控制"' not in source
        assert 'HeaderCardWidget("空间同步"' not in source
        assert 'HeaderCardWidget("同步数据"' in source
    finally:
        home.close_resources()


def test_spatial_sync_canvas_renders_imu_and_hand_pose(
    qt_application: QApplication,
) -> None:
    canvas = SpatialSyncCanvas()
    canvas.resize(340, 250)
    canvas.set_pose(_imu_pose())
    canvas.set_hand_result(_hand_result())

    pixmap = canvas.grab()
    qt_application.processEvents()
    status = canvas.status()

    assert not pixmap.isNull()
    assert status.has_imu_pose
    assert status.has_left_hand
    assert status.has_right_hand
    assert status.latest_frame_index == 12
    assert canvas.findChild(GLViewWidget, "spatialSyncViewport") is not None


def test_spatial_sync_canvas_uses_camera_frame_without_side_offsets() -> None:
    camera_points = np.asarray(
        [
            [0.10, 0.00, 0.00],
            [0.00, 0.20, 0.00],
            [0.00, 0.00, 0.30],
        ],
        dtype=np.float64,
    )

    scene_points = _camera_points_to_scene(camera_points, (1.0, 0.0, 0.0, 0.0))

    np.testing.assert_allclose(scene_points[0], [0.10, 0.00, 0.00], atol=1e-6)
    np.testing.assert_allclose(scene_points[1], [0.00, 0.00, -0.20], atol=1e-6)
    np.testing.assert_allclose(scene_points[2], [0.00, 0.30, 0.00], atol=1e-6)


def test_spatial_sync_canvas_draws_both_hands_in_the_same_camera_reference() -> None:
    canvas = SpatialSyncCanvas()
    repeated_points = [[0.01 * index, 0.02, 0.40] for index in range(21)]
    canvas.set_hand_result(
        {
            "frame_index": 14,
            "hands": [
                {"handedness": "left", "keypoints_3d_camera_m": repeated_points},
                {"handedness": "right", "keypoints_3d_camera_m": repeated_points},
            ],
        }
    )

    left_points = canvas._left_points.pos
    right_points = canvas._right_points.pos

    np.testing.assert_allclose(left_points, right_points, atol=1e-6)
    assert canvas.status().has_left_hand
    assert canvas.status().has_right_hand


def test_spatial_sync_canvas_uses_mesh_glasses_model() -> None:
    vertices, faces = _glasses_frame_mesh()

    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3
    assert len(vertices) > 300
    assert len(faces) > 300


def test_spatial_sync_canvas_ignores_bad_hand_pose_data(
    qt_application: QApplication,
) -> None:
    canvas = SpatialSyncCanvas()
    canvas.resize(340, 250)
    canvas.set_hand_result(
        {
            "frame_index": 13,
            "hands": [{"handedness": "left", "keypoints_3d_camera_m": [[0.0, 1.0]]}],
        }
    )

    pixmap = canvas.grab()
    qt_application.processEvents()
    status = canvas.status()

    assert not pixmap.isNull()
    assert not status.has_left_hand
    assert not status.has_right_hand
    assert status.latest_frame_index == 13


def test_spatial_sync_canvas_uses_opengl_not_painter_projection() -> None:
    source = (Path(__file__).parents[1] / "ui" / "widgets" / "spatial_sync_canvas.py").read_text(
        encoding="utf-8"
    )

    assert "GLViewWidget" in source
    assert "QPainter" not in source
    assert "paintEvent" not in source
    assert "StrongBodyLabel" in source
    assert "CaptionLabel" in source


def test_ui_uses_fluent_labels_instead_of_q_label() -> None:
    ui_root = Path(__file__).parents[1] / "ui"
    sources = "\n".join(path.read_text(encoding="utf-8") for path in ui_root.rglob("*.py"))

    assert "QLabel" not in sources


def test_top_context_badges_show_the_active_source_and_session() -> None:
    home = HomeView(RuntimeStub())  # type: ignore[arg-type]
    try:
        home._live_session_id = "live-session-1234"
        home._sync_context_badges()
        assert home.mode_badge.text() == "来源 · 实时"
        assert home.session_badge.text() == "会话 · live-ses"

        home._session_labels = {"记录会话": "replay-session-5678"}
        home.session_combo.addItem("记录会话")
        home.session_combo.setCurrentText("记录会话")
        home.set_viewer_mode(ViewerMode.REPLAY)
        assert home.mode_badge.text() == "来源 · 回放"
        assert home.session_badge.text() == "会话 · replay-s"
    finally:
        home.close_resources()


def test_replay_generation_tooltip_tracks_actual_job_completion(
    qt_application: QApplication,
) -> None:
    home = HomeView(RuntimeStub())  # type: ignore[arg-type]
    try:
        home.show()
        running = RuntimeSnapshot(
            perception={
                "replay": {
                    "state": "running",
                    "detail": "processing 3/10 frames",
                    "frames_processed": 3,
                    "frame_total": 10,
                    "report": None,
                }
            }
        )
        home._update_replay_job(running)
        qt_application.processEvents()
        assert home._replay_state_tooltip is not None
        assert home._replay_state_tooltip.isVisible()

        complete = RuntimeSnapshot(
            perception={
                "replay": {
                    "state": "complete",
                    "detail": "annotated replay is ready",
                    "frames_processed": 10,
                    "frame_total": 10,
                    "report": {},
                }
            }
        )
        home._update_replay_job(complete)
        assert home._replay_state_tooltip is None
        assert home.open_result_button.isEnabled()
    finally:
        home.close_resources()


def test_confidence_text_includes_every_score() -> None:
    text = _confidence_text(
        "左手",
        {
            "detector_confidence": 0.91,
            "reconstruction_quality": 0.82,
            "depth_score": 0.73,
            "coverage_score": 0.64,
            "compactness_score": 0.55,
            "final_confidence": 0.46,
        },
    )

    assert text == (
        "左手\n检测 0.91  ·  重建 0.82  ·  深度 0.73  ·  "
        "覆盖 0.64  ·  紧致 0.55  ·  最终 0.46"
    )


def test_ui_source_has_no_dearpygui_compatibility_layer() -> None:
    repository = Path(__file__).parents[1]
    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in (repository / "ui").rglob("*.py")
    )

    assert "dearpygui" not in sources.lower()
    assert "VideoSurface" not in sources


def test_canvas_benchmark_runs_directly_from_its_documented_path() -> None:
    repository = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "benchmark_native_texture.py"),
            "--frames",
            "20",
        ],
        cwd=repository,
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout[result.stdout.index("{") :])
    assert report["width"] == 640
    assert report["height"] == 480
    assert report["frames"] == 10
    assert report["effective_fps"] > 0
