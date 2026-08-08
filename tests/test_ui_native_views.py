from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtGui import QVector3D
from PyQt6.QtWidgets import QApplication, QWidget
from pyqtgraph.opengl import GLViewWidget
from qfluentwidgets import HeaderCardWidget, SimpleCardWidget, TitleLabel

from schemas import VioPose, VioTrajectory
from sensor_preprocessing import RecordedImuPose
from ui.app import MainWindow
from ui.application.runtime_state import RuntimeSnapshot
from ui.gateway.imu_preview import ImuPoseSnapshot
from ui.gateway.live_frames import LiveFrame
from ui.gateway.recording_models import (
    CaptureSessionState,
    RecordingClip,
    RecordingLibrary,
    RecordingSession,
)
from ui.gateway.webrtc_models import StreamControlAction
from ui.processing import (
    ProcessingJob,
    ProcessingJobState,
    ProcessingPreset,
    ProcessingRunInfo,
    ProcessingRunState,
    ProcessingServiceSnapshot,
    VioRunInfo,
    VioRunState,
)
from ui.replay.player import PlaybackFrame, ReplayState
from ui.views.home import HomeView, _confidence_body
from ui.views.video_processing import _processing_states, _result_counts, _Selection
from ui.widgets.spatial_sync_canvas import (
    SpatialReferenceFrame,
    SpatialSyncCanvas,
    _camera_points_to_scene,
    _floor_grid,
    _glasses_frame_mesh,
)
from ui.widgets.status_indicator import StatusIndicator
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
        self.perception_result: dict[str, object] | None = None
        self.imu_pose_reset_calls = 0
        self.processing_requests: list[tuple[str, str | None, str | None]] = []
        self.processing_cancels: list[str] = []
        self.processing_retries: list[str] = []
        self.processing_exports: list[tuple[str, str, str]] = []
        self.processing_auto_queue: list[bool] = []
        self.processing_default_presets: list[str] = []
        self.runs_by_session: dict[str, tuple[ProcessingRunInfo, ...]] = {}
        self.session_paths: dict[str, Path] = {}

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

    def request_imu_pose_reset(self) -> None:
        self.imu_pose_reset_calls += 1

    def request_processing(
        self,
        session_id: str,
        *,
        clip_id: str | None = None,
        preset_id: str | None,
    ) -> None:
        self.processing_requests.append((session_id, clip_id, preset_id))

    def request_processing_cancel(self, job_id: str) -> None:
        self.processing_cancels.append(job_id)

    def request_processing_retry(self, job_id: str) -> None:
        self.processing_retries.append(job_id)

    def request_processing_export(self, session_id: str, run_id: str, clip_id: str) -> None:
        self.processing_exports.append((session_id, run_id, clip_id))

    def request_processing_auto_queue(self, enabled: bool) -> None:
        self.processing_auto_queue.append(enabled)

    def request_processing_default_preset(self, preset_id: str) -> None:
        self.processing_default_presets.append(preset_id)

    def request_live_inference(self, _enabled: bool) -> None:
        return None

    def session_directory(self, session_id: str) -> Path:
        return self.session_paths.get(session_id, Path("missing-session"))

    def processing_runs(self, session_id: str) -> Future[tuple[ProcessingRunInfo, ...]]:
        future: Future[tuple[ProcessingRunInfo, ...]] = Future()
        future.set_result(self.runs_by_session.get(session_id, ()))
        return future

    def processing_result(self, *_values: object) -> Future[dict[str, object] | None]:
        future: Future[dict[str, object] | None] = Future()
        future.set_result(None)
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


def _spatial_view_parameters(
    canvas: SpatialSyncCanvas,
) -> tuple[tuple[float, float, float], float, float, float, float]:
    parameters = canvas.view.cameraParams()
    center = parameters["center"]
    return (
        tuple(round(float(value), 6) for value in (center.x(), center.y(), center.z())),
        round(float(parameters["distance"]), 6),
        round(float(parameters["elevation"]), 6),
        round(float(parameters["azimuth"]), 6),
        round(float(parameters["fov"]), 6),
    )


def _recording_library(
    *,
    complete: bool = True,
    include_incomplete: bool = False,
) -> RecordingLibrary:
    session_id = "1" * 32
    clip_id = "2" * 32
    clip = RecordingClip(
        clip_id=clip_id,
        recorded_at_unix_ms=1_700_000_000_000,
        ended_at_unix_ms=1_700_000_003_000,
        duration_ms=3_000,
        width=640,
        height=480,
        fps=30,
        file_size_bytes=4096,
        frame_count=90,
        media_url=f"/api/v1/recordings/media/{session_id}/{clip_id}",
    )
    sessions = [
        RecordingSession(
            session_id=session_id,
            display_name="示例会话",
            started_at_unix_ms=1_700_000_000_000,
            ended_at_unix_ms=1_700_000_003_000,
            state=(
                CaptureSessionState.COMPLETE
                if complete
                else CaptureSessionState.INCOMPLETE
            ),
            clips=[clip],
        )
    ]
    if include_incomplete:
        other_session_id = "3" * 32
        other_clip_id = "4" * 32
        sessions.append(
            RecordingSession(
                session_id=other_session_id,
                display_name="未完成会话",
                started_at_unix_ms=1_700_000_010_000,
                state=CaptureSessionState.INCOMPLETE,
                clips=[
                    RecordingClip(
                        clip_id=other_clip_id,
                        recorded_at_unix_ms=1_700_000_010_000,
                        ended_at_unix_ms=1_700_000_011_000,
                        duration_ms=1_000,
                        width=640,
                        height=480,
                        fps=30,
                        file_size_bytes=2048,
                        frame_count=30,
                        media_url=(
                            f"/api/v1/recordings/media/{other_session_id}/"
                            f"{other_clip_id}"
                        ),
                    )
                ],
            )
        )
    return RecordingLibrary(sessions=sessions)


def _processing_run(
    run_id: str,
    *,
    clip_id: str | None = None,
    completed_at_unix_ns: int = 2_000_000_000,
    vio_run_id: str | None = None,
) -> ProcessingRunInfo:
    return ProcessingRunInfo(
        run_id=run_id,
        session_id="session",
        clip_id=clip_id,
        preset=ProcessingPreset(),
        state=ProcessingRunState.COMPLETED,
        input_frame_count=100,
        inferred_frame_count=90,
        detected_hand_count=72,
        started_at_unix_ns=1_000_000_000,
        completed_at_unix_ns=completed_at_unix_ns,
        results_path=Path("results.sqlite"),
        is_viewable=True,
        vio_run_id=vio_run_id,
    )


def test_fluent_window_registers_dataset_processing_first_routes(
    qt_application: QApplication,
) -> None:
    runtime = RuntimeStub()
    window = MainWindow(runtime)  # type: ignore[arg-type]
    try:
        window.show()
        qt_application.processEvents()

        assert window.stackedWidget.count() == 5
        assert window.stackedWidget.widget(0) is window.processing_view
        assert window.stackedWidget.widget(1) is window.dataset_view
        assert window.stackedWidget.widget(2) is window.pipeline_view
        assert window.stackedWidget.widget(3) is window.home_view
        assert window.stackedWidget.widget(4) is window.settings_view
        assert window.stackedWidget.currentWidget() is window.processing_view
        assert window.processing_view.showing_hall
        assert len(window.processing_view.findChildren(VideoCanvas)) == 1
        assert len(window.processing_view.findChildren(SpatialSyncCanvas)) == 1
        assert not window.processing_view.canvas.isVisible()
        assert window.processing_view.findChild(QWidget, "processingInspector") is None
        headers = [
            window.pipeline_view.table.horizontalHeaderItem(column).text()
            for column in range(window.pipeline_view.table.columnCount())
        ]
        assert headers == [
            "任务",
            "会话",
            "范围",
            "处理方案",
            "状态",
            "进度",
            "耗时",
            "说明 / 失败原因",
        ]
        assert not (Path(__file__).parents[1] / "ui/views/library.py").exists()
        assert not (Path(__file__).parents[1] / "ui/views/annotation.py").exists()
        assert not (Path(__file__).parents[1] / "ui/views/diagnostics.py").exists()
    finally:
        window.close()
        qt_application.processEvents()
    assert runtime.stop_calls == 1


def test_video_processing_opens_in_hall_without_inspector_or_task_table(
    qt_application: QApplication,
) -> None:
    window = MainWindow(RuntimeStub())  # type: ignore[arg-type]
    try:
        window.resize(1280, 800)
        window.show()
        qt_application.processEvents()
        view = window.processing_view
        source = (Path(__file__).parents[1] / "ui" / "views" / "video_processing.py").read_text(
            encoding="utf-8"
        )

        assert view.stack.currentWidget() is view.hall
        assert view.hall.isVisible()
        assert "处理检查器" not in source
        assert not hasattr(view, "task_table")
    finally:
        window.close()
        qt_application.processEvents()


def test_window_shutdown_releases_runtime_after_a_view_close_failure(
    qt_application: QApplication,
) -> None:
    runtime = RuntimeStub()
    window = MainWindow(runtime)  # type: ignore[arg-type]
    processing_close = window.processing_view.close_resources
    home_close = window.home_view.close_resources
    closed: list[str] = []

    def fail_processing_close() -> None:
        processing_close()
        closed.append("processing")
        raise RuntimeError("replay close failed")

    def track_home_close() -> None:
        home_close()
        closed.append("home")

    window.processing_view.close_resources = fail_processing_close  # type: ignore[method-assign]
    window.home_view.close_resources = track_home_close  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="replay close failed"):
        window.shutdown()

    assert closed == ["processing", "home"]
    assert runtime.stop_calls == 1
    window.close()


def test_processing_video_and_space_views_consume_the_same_playback_frame(
    qt_application: QApplication,
) -> None:
    runtime = RuntimeStub()
    window = MainWindow(runtime)  # type: ignore[arg-type]
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image.setflags(write=False)
    frame = PlaybackFrame("session", "connection", 12, 30_000_000, 1_030_000_000, image)
    pose = RecordedImuPose(
        session_id="session",
        session_time_ns=frame.session_time_ns,
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        roll_degrees=0.0,
        pitch_degrees=0.0,
        yaw_degrees=0.0,
        accelerometer_mps2=(0.0, 0.0, 9.81),
        gyroscope_radps=(0.0, 0.0, 0.0),
        samples_received=100,
        samples_processed=100,
        recent_rate_hz=100.0,
    )
    result = _hand_result(frame_index=12)
    try:
        view = window.processing_view
        view._selection = _Selection("session", "connection")
        view.stack.setCurrentWidget(view.workbench)
        view.workbench.set_context("session", "同步测试", "connection")
        view.spatial_canvas.set_reference_frame(SpatialReferenceFrame.CAMERA)
        view.workbench.set_runs(
            (
                _processing_run("run-primary", completed_at_unix_ns=3_000_000_000),
                _processing_run("run-older"),
            ),
            "connection",
        )

        def completed_result(*_values: object) -> Future[dict[str, object]]:
            future: Future[dict[str, object]] = Future()
            future.set_result(result)
            return future

        runtime.processing_result = completed_result  # type: ignore[method-assign]
        view.replay._update(  # type: ignore[attr-defined]
            state=ReplayState.PAUSED,
            duration_seconds=2.0,
            position_seconds=1.03,
            frame=frame,
            imu_pose=pose,
        )
        window.show()
        qt_application.processEvents()
        view._update_frame()
        view._update_frame()

        assert view.canvas.status().latest_frame_index == 12
        assert view.canvas.status().overlay_visible
        assert view.spatial_canvas.status().latest_frame_index == 12
        assert view.spatial_canvas.status().has_imu_pose
        assert view.spatial_canvas.status().has_left_hand
        assert view.findChildren(VideoCanvas) == [view.canvas]
        assert view.findChildren(SpatialSyncCanvas) == [view.spatial_canvas]
    finally:
        window.close()
        qt_application.processEvents()


def test_workbench_processes_current_clip_and_exports_selected_result() -> None:
    runtime = RuntimeStub()
    window = MainWindow(runtime)  # type: ignore[arg-type]
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image.setflags(write=False)
    frame = PlaybackFrame("session", "clip", 7, 7_000_000, 107_000_000, image)
    view = window.processing_view
    try:
        view._selection = _Selection("session", "clip")
        view.stack.setCurrentWidget(view.workbench)
        view.workbench.set_context("session", "测试会话", "clip")
        view.workbench.set_runs((_processing_run("run-complete"),), "clip")
        view.workbench.process_button.click()
        view.replay._update(frame=frame)  # type: ignore[attr-defined]
        view.workbench.export_button.click()

        assert runtime.processing_requests == [("session", "clip", None)]
        assert runtime.processing_exports == [("session", "run-complete", "clip")]
    finally:
        window.close()


def test_pipeline_page_owns_cancel_and_retry_commands() -> None:
    runtime = RuntimeStub()
    window = MainWindow(runtime)  # type: ignore[arg-type]
    running = ProcessingJob(
        "running-job",
        "session",
        "clip",
        ProcessingPreset(),
        ProcessingJobState.RUNNING,
        1,
        2,
        run_id="run-active",
        started_at_unix_ns=1,
    )
    failed = ProcessingJob(
        "failed-job",
        "session",
        "clip",
        ProcessingPreset(),
        ProcessingJobState.FAILED,
        1,
        3,
        detail="model failed",
        started_at_unix_ns=1,
        finished_at_unix_ns=3,
    )
    try:
        runtime.snapshot_value = RuntimeSnapshot(
            processing=ProcessingServiceSnapshot(1, running.job_id, False, (running, failed))
        )
        window.pipeline_view._update_status()
        window.pipeline_view.table.selectRow(0)
        window.pipeline_view.cancel_button.click()
        window.pipeline_view.table.selectRow(1)
        window.pipeline_view.retry_button.click()

        assert runtime.processing_cancels == ["running-job"]
        assert runtime.processing_retries == ["failed-job"]
    finally:
        window.close()


def test_hall_keeps_incomplete_sessions_visible_but_disabled(
    qt_application: QApplication,
) -> None:
    window = MainWindow(RuntimeStub())  # type: ignore[arg-type]
    view = window.processing_view
    library = _recording_library(include_incomplete=True)
    try:
        view.hall.set_library(library)
        complete = view.hall.cards[("1" * 32, "2" * 32)]
        incomplete = view.hall.cards[("3" * 32, "4" * 32)]

        assert complete.isEnabled()
        assert not incomplete.isEnabled()
        assert "暂不可用" in incomplete.availability_badge.text()
    finally:
        window.close()
        qt_application.processEvents()


def test_hall_result_counts_include_session_and_matching_clip_runs() -> None:
    library = _recording_library()
    session = library.sessions[0]
    clip_id = session.clips[0].clip_id
    runs = (
        _processing_run("session-run"),
        _processing_run("clip-run", clip_id=clip_id),
        _processing_run("other-run", clip_id="9" * 32),
    )

    assert _result_counts(runs, session) == {clip_id: 2}


def test_hall_processing_state_maps_the_latest_applicable_job() -> None:
    session = _recording_library().sessions[0]
    clip_id = session.clips[0].clip_id
    running = ProcessingJob(
        "job",
        session.session_id,
        clip_id,
        ProcessingPreset(),
        ProcessingJobState.RUNNING,
        1,
        2,
    )

    assert _processing_states((running,), session) == {clip_id: "处理中"}

    partial = ProcessingJob(
        "partial-job",
        session.session_id,
        clip_id,
        ProcessingPreset(),
        ProcessingJobState.PARTIAL,
        3,
        4,
    )
    assert _processing_states((partial,), session) == {clip_id: "部分完成"}


def test_workbench_selects_latest_valid_result_and_filters_runs_by_clip() -> None:
    window = MainWindow(RuntimeStub())  # type: ignore[arg-type]
    try:
        workbench = window.processing_view.workbench
        runs = (
            _processing_run("older", completed_at_unix_ns=2_000_000_000),
            _processing_run(
                "other-clip",
                clip_id="other",
                completed_at_unix_ns=5_000_000_000,
            ),
            _processing_run(
                "newer",
                clip_id="clip",
                completed_at_unix_ns=4_000_000_000,
            ),
        )

        workbench.set_runs(runs, "clip")

        assert workbench.result_combo.count() == 3
        assert workbench.primary_run_id == "newer"
        workbench.result_combo.setCurrentIndex(0)
        assert workbench.primary_run_id is None
    finally:
        window.close()


def test_workbench_uses_vio_run_bound_to_selected_processing_result() -> None:
    window = MainWindow(RuntimeStub())  # type: ignore[arg-type]
    pose = VioPose(1, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    try:
        workbench = window.processing_view.workbench
        workbench.set_runs(
            (_processing_run("processed", clip_id="clip", vio_run_id="bound"),),
            "clip",
        )
        workbench.set_vio_runs(
            (
                VioRunInfo(
                    "newest-unrelated",
                    "session",
                    "clip",
                    VioRunState.COMPLETED,
                    Path("newest"),
                    2,
                    3,
                    VioTrajectory((pose,)),
                ),
                VioRunInfo(
                    "bound",
                    "session",
                    "clip",
                    VioRunState.COMPLETED,
                    Path("bound"),
                    1,
                    2,
                    VioTrajectory((pose,)),
                ),
            )
        )

        assert workbench.selected_vio_run is not None
        assert workbench.selected_vio_run.run_id == "bound"
    finally:
        window.close()


def test_entering_and_leaving_workbench_opens_once_then_unloads_decoder() -> None:
    runtime = RuntimeStub()
    window = MainWindow(runtime)  # type: ignore[arg-type]
    view = window.processing_view
    library = _recording_library()
    session = library.sessions[0]
    clip = session.clips[0]
    calls: list[tuple[object, ...]] = []
    try:
        view._sessions = {session.session_id: session}
        view.replay.open_session = (  # type: ignore[method-assign]
            lambda path, clip_id=None: calls.append(("open", path, clip_id))
        )
        view.replay.pause = lambda: calls.append(("pause",))  # type: ignore[method-assign]
        view.replay.unload = lambda: calls.append(("unload",))  # type: ignore[method-assign]

        view.open_workbench(session.session_id, clip.clip_id)
        assert view.stack.currentWidget() is view.workbench
        view.show_hall()

        assert calls == [
            ("open", Path("missing-session"), clip.clip_id),
            ("pause",),
            ("unload",),
        ]
        assert view.stack.currentWidget() is view.hall
        assert view.canvas.status().latest_frame_index is None
    finally:
        window.close()


def test_run_index_refresh_is_not_lost_while_an_older_query_is_in_flight() -> None:
    runtime = RuntimeStub()
    window = MainWindow(runtime)  # type: ignore[arg-type]
    view = window.processing_view
    session = _recording_library().sessions[0]
    first: Future[tuple[ProcessingRunInfo, ...]] = Future()
    second: Future[tuple[ProcessingRunInfo, ...]] = Future()
    futures = iter((first, second))
    runtime.processing_runs = lambda _session_id: next(futures)  # type: ignore[method-assign]
    try:
        view._sessions = {session.session_id: session}
        view._request_runs(session.session_id)
        view._request_runs(session.session_id, force=True)
        first.set_result((_processing_run("old"),))
        view._resolve_run_futures()
        second.set_result((_processing_run("new"),))
        view._resolve_run_futures()

        assert [run.run_id for run in view._runs[session.session_id]] == ["new"]
        assert not view._pending_run_refreshes
    finally:
        window.close()


def test_result_queries_keep_at_most_one_in_flight_request_per_layer() -> None:
    runtime = RuntimeStub()
    window = MainWindow(runtime)  # type: ignore[arg-type]
    view = window.processing_view
    calls: list[tuple[object, ...]] = []
    first: Future[dict[str, object] | None] = Future()
    second: Future[dict[str, object] | None] = Future()
    futures = iter((first, second))

    def query(*values: object) -> Future[dict[str, object] | None]:
        calls.append(values)
        return next(futures)

    runtime.processing_result = query  # type: ignore[method-assign]
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image.setflags(write=False)
    frame_one = PlaybackFrame("session", "clip", 1, 1, 1, image)
    frame_two = PlaybackFrame("session", "clip", 2, 2, 2, image)
    try:
        view.stack.setCurrentWidget(view.workbench)
        view.workbench.set_runs((_processing_run("run"),), "clip")
        view._query_results(frame_one)
        view._query_results(frame_two)
        assert len(calls) == 1

        view.replay._update(frame=frame_two)  # type: ignore[attr-defined]
        first.set_result(None)
        view._resolve_result_queries(("session", "clip", 2, 2))

        assert len(calls) == 2
        assert calls[-1][3:] == (2, 2)
        second.set_result(None)
    finally:
        window.close()


def test_home_header_only_shows_egoglass_title() -> None:
    home = HomeView(RuntimeStub())  # type: ignore[arg-type]
    try:
        titles = home.findChildren(TitleLabel)

        assert [title.text() for title in titles] == ["EgoGlass"]
    finally:
        home.close_resources()


def test_live_capture_has_one_canvas_and_no_legacy_replay_controls(
    qt_application: QApplication,
) -> None:
    home = HomeView(RuntimeStub())  # type: ignore[arg-type]
    try:
        home.show()
        qt_application.processEvents()

        assert home.findChildren(VideoCanvas) == [home.canvas]
        assert home.frame_link_strip.isVisible()
        assert home.mode_badge.text() == "来源 · 实时"
        assert not hasattr(home, "replay_controls")
        assert not hasattr(home, "right_stack")
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


def test_primary_hand_overlay_never_draws_a_center_divider(
    qt_application: QApplication,
) -> None:
    canvas = VideoCanvas()
    canvas.resize(960, 720)
    canvas.set_frame(_frame(3))
    assert canvas.set_overlay(_hand_result(frame_index=3))

    image = canvas.grab().toImage()
    qt_application.processEvents()
    midpoint_x = canvas.width() // 2
    center_pixels = [image.pixelColor(midpoint_x, y) for y in (60, 240, 360, 600)]

    assert all(max(color.red(), color.green(), color.blue()) < 8 for color in center_pixels)
    assert canvas.status().overlay_visible


def test_video_workbench_has_no_ab_comparison_control_or_overlay_path() -> None:
    repository = Path(__file__).parents[1]
    sources = "\n".join(
        (repository / path).read_text(encoding="utf-8")
        for path in (
            "ui/widgets/video_canvas.py",
            "ui/video_processing/workbench.py",
            "ui/views/video_processing.py",
        )
    )

    for removed_symbol in (
        "comparison_combo",
        "comparison_run_id",
        "comparisonSelectionChanged",
        "set_comparison_overlay",
        "_comparison_overlay",
        'CaptionLabel("A/B"',
    ):
        assert removed_symbol not in sources


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


def test_playback_seek_backwards_replaces_the_primary_overlay(
    qt_application: QApplication,
) -> None:
    canvas = VideoCanvas()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image.setflags(write=False)
    canvas.set_frame(
        PlaybackFrame("session", "clip", 20, 20_000_000, 120_000_000, image)
    )
    assert canvas.set_overlay(
        _hand_result(include_right=False, frame_index=20, sequence_id="clip")
    )

    canvas.set_frame(
        PlaybackFrame("session", "clip", 5, 5_000_000, 105_000_000, image)
    )

    assert canvas.status().latest_overlay_frame_index is None
    assert canvas.set_overlay(
        _hand_result(include_right=False, frame_index=5, sequence_id="clip")
    )
    assert canvas.status().latest_overlay_frame_index == 5
    assert canvas.status().overlay_visible


def test_home_controls_call_runtime_commands() -> None:
    runtime = RuntimeStub()
    home = HomeView(runtime)  # type: ignore[arg-type]
    try:
        home.stream_button.setProperty("action", "stop")
        home.recording_button.setProperty("action", "start")
        home._toggle_stream()
        home._toggle_recording()
        home.session_button.click()
        home.spatial_canvas.reset_pose_button.click()

        assert runtime.stream_actions == [StreamControlAction.STOP]
        assert runtime.recording_actions == ["start"]
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
        assert home.frame_link_strip not in sidebar_widgets
        assert isinstance(home.frame_link_strip, SimpleCardWidget)
        assert home.frame_link_strip.parent() is home.canvas.parent()
        assert home.findChild(type(home.stream_button), "streamControlButton") is home.stream_button
        assert len(home.findChildren(SpatialSyncCanvas)) == 1
        assert home.spatial_canvas.reset_pose_button.objectName() == "imuPoseResetButton"
        assert home.spatial_canvas.reset_pose_button.size().width() == 28
        assert home.spatial_canvas.reset_pose_button.iconSize().width() == 15
        assert "rgba(255, 255, 255, 0.14)" in home.spatial_canvas.styleSheet()
        assert not isinstance(home.spatial_canvas.parent(), HeaderCardWidget)
        assert home.spatial_canvas.findChild(GLViewWidget, "spatialSyncViewport") is not None
        assert len(home.sync_data_card.findChildren(QWidget, "syncStatusRow")) == 3
        assert len(home.findChildren(StatusIndicator)) == 11
        assert "background: transparent" in home.sidebar.viewport().styleSheet()

        source = (Path(__file__).parents[1] / "ui" / "views" / "home.py").read_text(
            encoding="utf-8"
        )
        assert 'HeaderCardWidget("采集控制"' not in source
        assert 'HeaderCardWidget("空间同步"' not in source
        assert 'HeaderCardWidget("同步数据"' in source
        assert "syncMetricBlock" not in source
        assert "_sync_metric_block" not in source
        assert "link_sync_badge" not in source
        assert "InfoBadge(" not in source
        assert "font-size:" not in source
    finally:
        home.close_resources()


def test_frame_link_strip_maps_live_display_metrics() -> None:
    home = HomeView(RuntimeStub())  # type: ignore[arg-type]
    try:
        display = SimpleNamespace(
            frames_received=120,
            frames_converted=118,
            presentation_fps=29.5,
            latest_conversion_ms=0.42,
            presentation_queue_depth=1,
            presentation_frames_dropped=2,
            pending_frames_overwritten=3,
        )

        home._update_metrics(RuntimeSnapshot(display=display))  # type: ignore[arg-type]

        assert home.frame_input_value.text() == "120 / 118"
        assert home.frame_input_detail.text() == "接收帧 / RGB 帧"
        assert home.frame_present_value.text() == "29.5 FPS"
        assert home.frame_present_detail.text() == "呈现 0 帧"
        assert home.frame_latency_value.text() == "0.42 / 0.00 ms"
        assert home.frame_buffer_value.text() == "1 / 2 / 3"
        assert home.frame_buffer_detail.text() == "队列 / 丢帧 / 跳帧 · 覆盖 --"
    finally:
        home.close_resources()


def test_imu_row_separates_sync_state_from_gyroscope_rate() -> None:
    home = HomeView(RuntimeStub())  # type: ignore[arg-type]
    try:
        home._update_imu(RuntimeSnapshot(imu_pose=_imu_pose()))

        assert home.imu_sync_badge.text() == "已同步"
        assert home.imu_detail.text().startswith("100.0 Hz")
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


def test_spatial_sync_scene_updates_do_not_move_the_observer_camera(
    qt_application: QApplication,
) -> None:
    canvas = SpatialSyncCanvas()
    canvas.resize(640, 480)
    expected = _spatial_view_parameters(canvas)
    canvas.set_hand_result(_hand_result())
    far_result = _hand_result(frame_index=13)
    hands = far_result["hands"]
    assert isinstance(hands, list)
    for hand in hands:
        assert isinstance(hand, dict)
        hand["keypoints_3d_camera_m"] = [
            [8.0 + index, -5.0, 12.0] for index in range(21)
        ]
    canvas.set_hand_result(far_result)
    qt_application.processEvents()

    center = canvas.view.opts["center"]
    assert isinstance(center, QVector3D)
    assert _spatial_view_parameters(canvas) == expected
    assert not canvas.view.viewMatrix().isIdentity()
    canvas.close()


def test_spatial_sync_preserves_manual_view_until_reference_frame_switch() -> None:
    canvas = SpatialSyncCanvas()
    try:
        canvas.view.opts["fov"] = 58.0
        canvas.view.setCameraPosition(
            pos=QVector3D(2.0, 3.0, 4.0),
            distance=2.4,
            elevation=31.0,
            azimuth=-42.0,
        )
        manual = _spatial_view_parameters(canvas)

        canvas.set_hand_result(_hand_result())

        assert _spatial_view_parameters(canvas) == manual
    finally:
        canvas.close()


def test_spatial_sync_reference_switch_restores_exact_mode_presets() -> None:
    canvas = SpatialSyncCanvas()
    view = canvas.view
    try:
        assert _spatial_view_parameters(canvas) == (
            (0.0, 0.4, -0.1),
            1.1,
            5.0,
            -90.0,
            43.0,
        )

        canvas.set_reference_frame(SpatialReferenceFrame.WORLD)
        assert canvas.view is view
        assert _spatial_view_parameters(canvas) == (
            (0.0, 0.3, -0.1),
            1.65,
            12.0,
            -90.0,
            43.0,
        )

        canvas.view.setCameraPosition(distance=2.8, elevation=40.0, azimuth=15.0)
        canvas.view.opts["fov"] = 60.0
        canvas.set_reference_frame(SpatialReferenceFrame.CAMERA)
        assert _spatial_view_parameters(canvas) == (
            (0.0, 0.4, -0.1),
            1.1,
            5.0,
            -90.0,
            43.0,
        )

        canvas.view.setCameraPosition(distance=3.0, elevation=-10.0, azimuth=80.0)
        canvas.set_reference_frame(SpatialReferenceFrame.WORLD)
        assert _spatial_view_parameters(canvas) == (
            (0.0, 0.3, -0.1),
            1.65,
            12.0,
            -90.0,
            43.0,
        )
    finally:
        canvas.close()


def test_spatial_sync_starts_from_an_upright_front_view_with_horizontal_grid() -> None:
    canvas = SpatialSyncCanvas()
    try:
        assert canvas.view.opts["azimuth"] == -90
        assert canvas.view.opts["elevation"] == 5
        matrix = canvas.view.viewMatrix()
        origin = matrix.map(QVector3D(0.0, 0.0, 0.0))
        horizontal = matrix.map(QVector3D(1.0, 0.0, 0.0))
        vertical = matrix.map(QVector3D(0.0, 0.0, 1.0))
        assert horizontal.x() - origin.x() > 0.99
        assert abs(horizontal.y() - origin.y()) < 1e-6
        assert abs(vertical.x() - origin.x()) < 1e-6
        assert vertical.y() - origin.y() > 0.95
        grid = _floor_grid()
        assert np.allclose(grid[:, 2], -0.42)
        assert np.ptp(grid[:, 0]) > 0
        assert np.ptp(grid[:, 1]) > 0
    finally:
        canvas.close()


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


def test_spatial_sync_canvas_does_not_render_a_glasses_model() -> None:
    vertices, faces = _glasses_frame_mesh()

    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3
    assert len(vertices) == 0
    assert len(faces) == 0


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
    finally:
        home.close_resources()


def test_confidence_body_includes_every_score() -> None:
    text = _confidence_body(
        {
            "detector_confidence": 0.91,
            "reconstruction_quality": 0.82,
            "depth_score": 0.73,
            "coverage_score": 0.64,
            "compactness_score": 0.55,
            "final_confidence": 0.46,
        }
    )

    assert text == (
        "检测 0.91 · 重建 0.82 · 深度 0.73\n"
        "覆盖 0.64 · 紧致 0.55 · 最终 0.46"
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
