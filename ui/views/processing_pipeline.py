from __future__ import annotations

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QHBoxLayout, QHeaderView, QTableWidgetItem, QVBoxLayout, QWidget
from qfluentwidgets import (
    CaptionLabel,
    FluentIcon,
    InfoBar,
    PrimaryPushButton,
    PushButton,
    TableWidget,
    TitleLabel,
)

from perception.video_processing import ProcessingJob, ProcessingJobState
from ui.runtime import UnifiedRuntimeHost


class ProcessingPipelineView(QWidget):
    """Present persistent processing jobs and their lifecycle commands."""

    def __init__(
        self,
        runtime: UnifiedRuntimeHost,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("processingPipelineView")
        self.runtime = runtime
        self._last_revision = -1
        self._job_ids: list[str] = []
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._update_status)
        self._timer.start()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 18)
        root.setSpacing(12)
        header = QHBoxLayout()
        title = QVBoxLayout()
        title.setSpacing(2)
        title.addWidget(TitleLabel("流水线", self))
        self.summary = CaptionLabel("暂无处理任务", self)
        title.addWidget(self.summary)
        header.addLayout(title)
        header.addStretch(1)
        self.cancel_button = PushButton("取消", self)
        self.cancel_button.setIcon(FluentIcon.CANCEL)
        self.cancel_button.clicked.connect(self._cancel_selected)
        header.addWidget(self.cancel_button)
        self.retry_button = PrimaryPushButton("重试", self)
        self.retry_button.setIcon(FluentIcon.SYNC)
        self.retry_button.clicked.connect(self._retry_selected)
        header.addWidget(self.retry_button)
        root.addLayout(header)

        self.table = TableWidget(self)
        self.table.setObjectName("processingJobTable")
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(
            ["任务", "会话", "范围", "处理方案", "状态", "进度", "耗时", "说明 / 失败原因"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(TableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(TableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._sync_actions)
        root.addWidget(self.table, 1)
        self._sync_actions()

    def close_resources(self) -> None:
        self._timer.stop()

    def _update_status(self) -> None:
        processing = self.runtime.snapshot().processing
        if processing is not None and processing.revision != self._last_revision:
            self._last_revision = processing.revision
            self._set_jobs(processing.jobs)
        if self.isVisible():
            for result in self.runtime.command_results():
                if result.succeeded:
                    InfoBar.success("操作完成", result.detail, duration=2200, parent=self.window())
                else:
                    InfoBar.error("操作失败", result.detail, duration=4500, parent=self.window())

    def _set_jobs(self, jobs: tuple[ProcessingJob, ...]) -> None:
        self._job_ids = [job.job_id for job in jobs]
        self.table.setRowCount(len(jobs))
        for row, job in enumerate(jobs):
            values = (
                job.job_id[-8:],
                job.session_id[:8],
                "完整会话" if job.clip_id is None else job.clip_id[:8],
                job.preset.display_name,
                _job_state_text(job.state),
                f"{job.progress_current}/{job.progress_total}",
                _duration_text(job.elapsed_seconds),
                job.detail,
            )
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        active = next(
            (
                job
                for job in jobs
                if job.state
                in {
                    ProcessingJobState.PREPARING,
                    ProcessingJobState.RUNNING,
                    ProcessingJobState.CANCELING,
                }
            ),
            None,
        )
        self.summary.setText(
            "暂无处理任务"
            if not jobs
            else (
                f"{_job_state_text(active.state)} · "
                f"{active.progress_current}/{active.progress_total}"
                if active is not None
                else f"共 {len(jobs)} 个历史任务"
            )
        )
        self._sync_actions()

    def _selected_job(self) -> ProcessingJob | None:
        row = self.table.currentRow()
        processing = self.runtime.snapshot().processing
        if processing is None or not 0 <= row < len(self._job_ids):
            return None
        job_id = self._job_ids[row]
        return next((job for job in processing.jobs if job.job_id == job_id), None)

    def _sync_actions(self) -> None:
        job = self._selected_job()
        self.cancel_button.setEnabled(
            job is not None
            and job.state
            in {
                ProcessingJobState.QUEUED,
                ProcessingJobState.PREPARING,
                ProcessingJobState.RUNNING,
            }
        )
        self.retry_button.setEnabled(
            job is not None
            and job.state
            in {
                ProcessingJobState.FAILED,
                ProcessingJobState.INTERRUPTED,
                ProcessingJobState.CANCELED,
            }
        )

    def _cancel_selected(self) -> None:
        job = self._selected_job()
        if job is not None:
            self.runtime.request_processing_cancel(job.job_id)

    def _retry_selected(self) -> None:
        job = self._selected_job()
        if job is not None:
            self.runtime.request_processing_retry(job.job_id)


def _job_state_text(state: ProcessingJobState) -> str:
    return {
        ProcessingJobState.QUEUED: "等待",
        ProcessingJobState.PREPARING: "校验",
        ProcessingJobState.RUNNING: "处理中",
        ProcessingJobState.CANCELING: "取消中",
        ProcessingJobState.COMPLETED: "完成",
        ProcessingJobState.FAILED: "失败",
        ProcessingJobState.INTERRUPTED: "中断",
        ProcessingJobState.CANCELED: "已取消",
    }[state]


def _duration_text(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(round(seconds), 60)
    return f"{minutes:d}m {remainder:02d}s"
