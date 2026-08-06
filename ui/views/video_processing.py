from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass

from PyQt6.QtCore import QEvent, QTimer
from PyQt6.QtWidgets import QStackedWidget, QVBoxLayout, QWidget
from qfluentwidgets import InfoBar

from ui.application.runtime_host import UnifiedRuntimeHost
from ui.gateway.recording_models import RecordingLibrary, RecordingSession
from ui.processing import ProcessingJob, ProcessingJobState, ProcessingRunInfo, VioRunInfo
from ui.replay.player import PlaybackFrame, ReplayPlayer, ReplayState
from ui.video_processing import ProcessingWorkbench, VideoHall, VideoThumbnailService


@dataclass(frozen=True, slots=True)
class _Selection:
    session_id: str
    clip_id: str


class VideoProcessingView(QWidget):
    """Coordinate the video hall, one reusable decoder, and result overlays."""

    def __init__(
        self,
        runtime: UnifiedRuntimeHost,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("videoProcessingView")
        self.runtime = runtime
        configuration_service = getattr(runtime, "configuration_service", None)
        sensor_config_path = (
            configuration_service.config_directory / "sensor-preprocessing.yaml"
            if configuration_service is not None
            else None
        )
        self.replay = ReplayPlayer(sensor_config_path)
        self.thumbnails = VideoThumbnailService(workers=2)
        self._selection: _Selection | None = None
        self._library: RecordingLibrary | None = None
        self._library_signature: tuple[object, ...] = ()
        self._sessions: dict[str, RecordingSession] = {}
        self._last_frame_key: tuple[str, str, int, int] | None = None
        self._last_processing_revision = -1
        self._last_replay_error: str | None = None
        self._run_futures: dict[
            str, concurrent.futures.Future[tuple[ProcessingRunInfo, ...]]
        ] = {}
        self._pending_run_refreshes: set[str] = set()
        self._runs: dict[str, tuple[ProcessingRunInfo, ...]] = {}
        self._vio_runs: dict[str, tuple[VioRunInfo, ...]] = {}
        self._vio_run_futures: dict[
            str, concurrent.futures.Future[tuple[VioRunInfo, ...]]
        ] = {}
        self._pending_vio_refreshes: set[str] = set()
        self._result_futures: dict[
            str,
            tuple[
                tuple[str, str, int, int],
                concurrent.futures.Future[dict[str, object] | None],
            ],
        ] = {}
        self._build_ui()

        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(16)
        self._frame_timer.timeout.connect(self._update_frame)
        self._frame_timer.start()
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(100)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget(self)
        self.hall = VideoHall(self.stack)
        self.workbench = ProcessingWorkbench(self.replay, self.stack)
        self.stack.addWidget(self.hall)
        self.stack.addWidget(self.workbench)
        self.stack.setCurrentWidget(self.hall)
        root.addWidget(self.stack)

        self.hall.refreshRequested.connect(self._refresh_library)
        self.hall.clipActivated.connect(self.open_workbench)
        self.workbench.backRequested.connect(self.show_hall)
        self.workbench.processRequested.connect(self._process_current_clip)
        self.workbench.exportRequested.connect(self._export_current_result)
        self.workbench.resultSelectionChanged.connect(self._clear_result_queries)
        self.workbench.comparisonSelectionChanged.connect(self._clear_result_queries)

        # Stable compatibility handles for render tests and canvas consumers.
        self.canvas = self.workbench.canvas
        self.spatial_canvas = self.workbench.spatial_canvas

    @property
    def showing_hall(self) -> bool:
        return self.stack.currentWidget() is self.hall

    def close_resources(self) -> None:
        self._frame_timer.stop()
        self._status_timer.stop()
        self.thumbnails.close()
        self.replay.close()

    def show_hall(self) -> None:
        self.replay.pause()
        self.replay.unload()
        self._selection = None
        self._last_frame_key = None
        self._last_replay_error = None
        self._result_futures.clear()
        self._vio_run_futures.clear()
        self._pending_vio_refreshes.clear()
        self.workbench.clear_media()
        self.stack.setCurrentWidget(self.hall)

    def open_workbench(self, session_id: str, clip_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            self._show_error("录像库中找不到所选会话")
            return
        self._selection = _Selection(session_id, clip_id)
        self._last_frame_key = None
        self._last_replay_error = None
        self.workbench.set_context(
            session_id,
            session.display_name or session_id[:8],
            clip_id,
        )
        self.workbench.set_runs(self._runs.get(session_id, ()), clip_id)
        self.workbench.set_vio_runs(self._vio_runs.get(session_id, ()))
        self.stack.setCurrentWidget(self.workbench)
        try:
            self.replay.open_session(self.runtime.session_directory(session_id), clip_id)
        except Exception as error:
            self.show_hall()
            self._show_error(str(error))
            return
        self._request_runs(session_id, force=True)
        self._request_vio_runs(session_id, force=True)
        self._clear_result_queries()

    def hideEvent(self, event: QEvent) -> None:
        if not self.showing_hall:
            self.show_hall()
        super().hideEvent(event)

    def _refresh_library(self) -> None:
        self.thumbnails.invalidate()
        self.runtime.request_library_refresh()

    def _update_status(self) -> None:
        snapshot = self.runtime.snapshot()
        if snapshot.library is not None:
            self._apply_library(snapshot.library)
        processing = snapshot.processing
        if processing is not None and processing.revision != self._last_processing_revision:
            self._last_processing_revision = processing.revision
            affected = {
                job.session_id
                for job in processing.jobs
                if job.run_id is not None or job.state.value in {"completed", "failed"}
            }
            for session_id in affected:
                if session_id in self._sessions:
                    self._request_runs(session_id, force=True)
                    self._request_vio_runs(session_id, force=True)
            for session_id, session in self._sessions.items():
                self.hall.set_processing_states(
                    session_id,
                    _processing_states(processing.jobs, session),
                )
        if processing is not None and self._selection is not None:
            active_job = next(
                (
                    job
                    for job in processing.jobs
                    if job.session_id == self._selection.session_id
                    and (job.clip_id is None or job.clip_id == self._selection.clip_id)
                    and job.state
                    in {
                        ProcessingJobState.QUEUED,
                        ProcessingJobState.PREPARING,
                        ProcessingJobState.RUNNING,
                        ProcessingJobState.CANCELING,
                    }
                ),
                None,
            )
            if active_job is not None:
                self.workbench.set_vio_status("SLAM/VIO：随当前离线处理任务执行")
        self._resolve_run_futures()
        self._resolve_vio_futures()
        self._resolve_thumbnails()
        if self.isVisible():
            self._drain_command_results()

    def _apply_library(self, library: RecordingLibrary) -> None:
        signature = tuple(
            (
                session.session_id,
                session.state.value,
                tuple(
                    (
                        clip.clip_id,
                        clip.file_size_bytes,
                        clip.frame_count,
                        clip.duration_ms,
                    )
                    for clip in session.clips
                ),
            )
            for session in library.sessions
        )
        if signature == self._library_signature:
            return
        self._library_signature = signature
        self._library = library
        self._sessions = {session.session_id: session for session in library.sessions}
        self.hall.set_library(library)
        for session in library.sessions:
            if not session.clips:
                continue
            try:
                session_path = self.runtime.session_directory(session.session_id)
            except Exception:
                continue
            for clip in session.clips:
                self.thumbnails.request(
                    session.session_id,
                    clip.clip_id,
                    session_path,
                )
            self._request_runs(session.session_id)
            self._request_vio_runs(session.session_id)

    def _request_runs(self, session_id: str, *, force: bool = False) -> None:
        if session_id in self._run_futures:
            if force:
                self._pending_run_refreshes.add(session_id)
            return
        if not force and session_id in self._runs:
            self.hall.set_result_counts(
                session_id,
                _result_counts(self._runs[session_id], self._sessions[session_id]),
            )
            return
        future = self.runtime.processing_runs(session_id)
        if isinstance(future, concurrent.futures.Future):
            self._run_futures[session_id] = future

    def _resolve_run_futures(self) -> None:
        for session_id, future in tuple(self._run_futures.items()):
            if not future.done():
                continue
            del self._run_futures[session_id]
            try:
                runs = future.result()
            except Exception as error:
                if self._selection is not None and self._selection.session_id == session_id:
                    self._show_error(str(error))
                if session_id in self._pending_run_refreshes:
                    self._pending_run_refreshes.remove(session_id)
                    self._request_runs(session_id, force=True)
                continue
            self._runs[session_id] = runs
            session = self._sessions.get(session_id)
            if session is not None:
                self.hall.set_result_counts(session_id, _result_counts(runs, session))
            selection = self._selection
            if (
                selection is not None
                and selection.session_id == session_id
                and not self.showing_hall
            ):
                self.workbench.set_runs(runs, selection.clip_id)
            if session_id in self._pending_run_refreshes:
                self._pending_run_refreshes.remove(session_id)
                self._request_runs(session_id, force=True)

    def _request_vio_runs(self, session_id: str, *, force: bool = False) -> None:
        if session_id in self._vio_run_futures:
            if force:
                self._pending_vio_refreshes.add(session_id)
            return
        if not force and session_id in self._vio_runs:
            return
        request = getattr(self.runtime, "vio_runs", None)
        if not callable(request):
            return
        future = request(session_id)
        if isinstance(future, concurrent.futures.Future):
            self._vio_run_futures[session_id] = future

    def _resolve_vio_futures(self) -> None:
        for session_id, future in tuple(self._vio_run_futures.items()):
            if not future.done():
                continue
            del self._vio_run_futures[session_id]
            try:
                runs = future.result()
            except Exception as error:
                if self._selection is not None and self._selection.session_id == session_id:
                    self._show_error(str(error))
                continue
            self._vio_runs[session_id] = runs
            selection = self._selection
            if (
                selection is not None
                and selection.session_id == session_id
                and not self.showing_hall
            ):
                self.workbench.set_vio_runs(runs)
            if session_id in self._pending_vio_refreshes:
                self._pending_vio_refreshes.remove(session_id)
                self._request_vio_runs(session_id, force=True)

    def _resolve_thumbnails(self) -> None:
        for result in self.thumbnails.take_completed():
            self.hall.set_thumbnail(result.session_id, result.clip_id, result.image)

    def _update_frame(self) -> None:
        if self.showing_hall:
            return
        snapshot = self.replay.snapshot()
        self.workbench.set_replay(snapshot)
        if (
            snapshot.state is ReplayState.ERROR
            and snapshot.error
            and snapshot.error != self._last_replay_error
        ):
            self._last_replay_error = snapshot.error
            self._show_error(snapshot.error)
        frame = snapshot.frame
        if frame is None:
            return
        key = (frame.session_id, frame.clip_id, frame.frame_index, frame.session_time_ns)
        if key != self._last_frame_key:
            self._last_frame_key = key
            selection = self._selection
            if selection is None or selection.clip_id != frame.clip_id:
                session = self._sessions.get(frame.session_id)
                self._selection = _Selection(frame.session_id, frame.clip_id)
                if session is not None:
                    self.workbench.set_context(
                        frame.session_id,
                        session.display_name or frame.session_id[:8],
                        frame.clip_id,
                    )
                self.workbench.set_runs(self._runs.get(frame.session_id, ()), frame.clip_id)
            self.canvas.set_frame(frame)
            self.workbench.set_imu_pose(snapshot.imu_pose)
            selected_vio = self.workbench.selected_vio_run
            self.workbench.set_vio_pose(
                selected_vio.pose_at(frame.session_time_ns) if selected_vio is not None else None
            )
            self._query_results(frame)
        self._resolve_result_queries(key)

    def _query_results(self, frame: PlaybackFrame) -> None:
        for layer, run_id in (
            ("primary", self.workbench.primary_run_id),
            ("comparison", self.workbench.comparison_run_id),
        ):
            if run_id is None:
                continue
            existing = self._result_futures.get(layer)
            if existing is not None and not existing[1].done():
                continue
            key = (frame.session_id, frame.clip_id, frame.frame_index, frame.session_time_ns)
            self._result_futures[layer] = (
                key,
                self.runtime.processing_result(
                    frame.session_id,
                    run_id,
                    frame.clip_id,
                    frame.frame_index,
                    frame.session_time_ns,
                ),
            )

    def _resolve_result_queries(self, frame_key: tuple[str, str, int, int]) -> None:
        refresh_current_frame = False
        for layer, (key, future) in tuple(self._result_futures.items()):
            if not future.done():
                continue
            del self._result_futures[layer]
            try:
                result = future.result()
            except Exception as error:
                self._show_error(str(error))
                continue
            if key != frame_key:
                refresh_current_frame = True
                continue
            if layer == "primary":
                self.canvas.set_overlay(result)
                self.workbench.set_hand_result(result)
            else:
                self.canvas.set_comparison_overlay(result)
        if refresh_current_frame:
            frame = self.replay.snapshot().frame
            if frame is not None:
                self._query_results(frame)

    def _clear_result_queries(self) -> None:
        self._result_futures.clear()
        self.canvas.set_overlay(None)
        self.canvas.set_comparison_overlay(None)
        self.workbench.set_hand_result(None)
        frame = self.replay.snapshot().frame
        if frame is not None and not self.showing_hall:
            self._query_results(frame)

    def _process_current_clip(self) -> None:
        selection = self._selection
        if selection is None:
            self._show_error("当前没有可处理的视频片段")
            return
        self.runtime.request_processing(
            selection.session_id,
            clip_id=selection.clip_id,
            preset_id=None,
        )

    def _export_current_result(self) -> None:
        selection = self._selection
        frame = self.replay.snapshot().frame
        run_id = self.workbench.primary_run_id
        if selection is None or frame is None or run_id is None:
            self._show_error("请先选择可查看的处理结果")
            return
        self.runtime.request_processing_export(
            selection.session_id,
            run_id,
            frame.clip_id,
        )

    def _drain_command_results(self) -> None:
        for result in self.runtime.command_results():
            if result.succeeded:
                InfoBar.success("操作完成", result.detail, duration=3000, parent=self.window())
            else:
                self._show_error(result.detail)

    def _show_error(self, detail: str) -> None:
        InfoBar.error("操作失败", detail, duration=4500, parent=self.window())


def _result_counts(
    runs: tuple[ProcessingRunInfo, ...],
    session: RecordingSession,
) -> dict[str, int]:
    return {
        clip.clip_id: sum(run.covers_clip(clip.clip_id) for run in runs)
        for clip in session.clips
    }


def _processing_states(
    jobs: tuple[ProcessingJob, ...],
    session: RecordingSession,
) -> dict[str, str]:
    labels = {
        ProcessingJobState.QUEUED: "等待",
        ProcessingJobState.PREPARING: "校验",
        ProcessingJobState.RUNNING: "处理中",
        ProcessingJobState.CANCELING: "取消中",
        ProcessingJobState.COMPLETED: "完成",
        ProcessingJobState.FAILED: "失败",
        ProcessingJobState.INTERRUPTED: "中断",
        ProcessingJobState.CANCELED: "已取消",
    }
    states: dict[str, str] = {}
    for clip in session.clips:
        job = next(
            (
                item
                for item in jobs
                if item.session_id == session.session_id
                and (item.clip_id is None or item.clip_id == clip.clip_id)
            ),
            None,
        )
        if job is not None:
            states[clip.clip_id] = labels[job.state]
    return states
