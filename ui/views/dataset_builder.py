from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QHBoxLayout, QHeaderView, QTableWidgetItem, QVBoxLayout, QWidget
from qfluentwidgets import (
    CaptionLabel,
    ComboBox,
    FluentIcon,
    InfoBar,
    LineEdit,
    PrimaryPushButton,
    PushButton,
    SpinBox,
    TableWidget,
    TitleLabel,
)

from schemas import QualityGate
from ui.application.runtime_host import UnifiedRuntimeHost
from ui.dataset_builder import (
    DatasetBuilder,
    DatasetBuildResult,
    DatasetCandidate,
    DatasetCandidateSummary,
    DatasetCatalogService,
    DatasetQualityChecker,
)


class DatasetBuildWizard(QWidget):
    """Compact review and publication controls for the selected candidate."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("datasetBuildWizard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)
        command_row = QHBoxLayout()
        command_row.setSpacing(10)
        text = QVBoxLayout()
        text.setSpacing(2)
        self.step_title = TitleLabel("Dataset build", self)
        self.detail = CaptionLabel("Select a verified offline run to inspect.", self)
        text.addWidget(self.step_title)
        text.addWidget(self.detail)
        command_row.addLayout(text, 1)
        self.dataset_id = LineEdit(self)
        self.dataset_id.setObjectName("datasetVersionInput")
        self.dataset_id.setPlaceholderText("dataset version id")
        self.dataset_id.setFixedWidth(220)
        command_row.addWidget(self.dataset_id)
        self.inspect_button = PushButton("Inspect", self)
        self.inspect_button.setIcon(FluentIcon.SEARCH)
        command_row.addWidget(self.inspect_button)
        self.publish_button = PrimaryPushButton("Publish", self)
        self.publish_button.setIcon(FluentIcon.SAVE)
        self.publish_button.setEnabled(False)
        command_row.addWidget(self.publish_button)
        layout.addLayout(command_row)

        self.issue_table = TableWidget(self)
        self.issue_table.setObjectName("datasetQualityIssueTable")
        self.issue_table.setColumnCount(5)
        self.issue_table.setHorizontalHeaderLabels(
            ["Gate", "Issue", "Interval", "Status", "Message"]
        )
        self.issue_table.verticalHeader().setVisible(False)
        self.issue_table.setSelectionBehavior(TableWidget.SelectionBehavior.SelectRows)
        self.issue_table.setEditTriggers(TableWidget.EditTrigger.NoEditTriggers)
        self.issue_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.issue_table.setMinimumHeight(86)
        self.issue_table.setMaximumHeight(150)
        layout.addWidget(self.issue_table)

        review_row = QHBoxLayout()
        review_row.setSpacing(8)
        self.operator = LineEdit(self)
        self.operator.setObjectName("qualityReviewOperator")
        self.operator.setPlaceholderText("reviewer")
        self.operator.setFixedWidth(150)
        self.reason = LineEdit(self)
        self.reason.setObjectName("qualityReviewReason")
        self.reason.setPlaceholderText("reason for restoring a soft gate")
        self.restore_button = PushButton("Restore soft gate", self)
        self.restore_button.setIcon(FluentIcon.ACCEPT)
        self.restore_button.setEnabled(False)
        review_row.addWidget(self.operator)
        review_row.addWidget(self.reason, 1)
        review_row.addWidget(self.restore_button)
        layout.addLayout(review_row)
        self.issue_table.itemSelectionChanged.connect(self._issue_selection_changed)

    def set_restore_handler(self, handler: Callable[[], None]) -> None:
        self.restore_button.clicked.connect(handler)

    def set_candidate(self, candidate: DatasetCandidate | None) -> None:
        if candidate is None:
            self.detail.setText("Select a verified offline run to inspect.")
            self.publish_button.setEnabled(False)
            self.issue_table.setRowCount(0)
            self.restore_button.setEnabled(False)
            return
        self.detail.setText(
            f"{len(candidate.episodes)} episodes | "
            f"{candidate.quality.issue_count} quality issues | "
            f"annotation {candidate.annotation_revision_id[:8]}"
        )
        self.publish_button.setEnabled(candidate.publishable)
        issues = candidate.quality.all_issues()
        self.issue_table.setRowCount(len(issues))
        for row_index, issue in enumerate(issues):
            interval = (
                f"{issue.clip_id}:{issue.start_frame_index}-"
                f"{issue.end_frame_index_exclusive}"
            )
            status = (
                "restored"
                if issue.restored
                else ("blocked" if issue.gate is QualityGate.HARD else "review")
            )
            values = (issue.gate.value, issue.issue_id, interval, status, issue.message)
            for column, value in enumerate(values):
                self.issue_table.setItem(row_index, column, QTableWidgetItem(str(value)))
        self._issue_selection_changed()

    def selected_issue_id(self) -> str | None:
        candidate = getattr(self.parent(), "_candidate", None)
        row = self.issue_table.currentRow()
        issues = candidate.quality.all_issues() if candidate is not None else ()
        if 0 <= row < len(issues):
            return issues[row].issue_id
        return None

    def _issue_selection_changed(self) -> None:
        candidate = getattr(self.parent(), "_candidate", None)
        row = self.issue_table.currentRow()
        issues = candidate.quality.all_issues() if candidate is not None else ()
        selected = issues[row] if 0 <= row < len(issues) else None
        self.restore_button.setEnabled(
            selected is not None
            and selected.gate is QualityGate.SOFT
            and not selected.restored
        )


class DatasetView(QWidget):
    """Dataset candidate hall and immutable publication entry point."""

    def __init__(self, runtime: UnifiedRuntimeHost, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("datasetView")
        self.runtime = runtime
        runtime_config = getattr(runtime, "config", None)
        configured_root = getattr(runtime_config, "recordings_root", Path("local-data/recordings"))
        root = Path(configured_root).expanduser().resolve()
        self.catalog = DatasetCatalogService(root)
        self.builder = DatasetBuilder(root)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="egoglass-dataset",
        )
        self._scan_future: concurrent.futures.Future[tuple[DatasetCandidateSummary, ...]] | None = (
            None
        )
        self._candidate_future: concurrent.futures.Future[DatasetCandidate] | None = None
        self._publish_future: concurrent.futures.Future[DatasetBuildResult] | None = None
        self._rows: tuple[DatasetCandidateSummary, ...] = ()
        self._all_rows: tuple[DatasetCandidateSummary, ...] = ()
        self._candidate: DatasetCandidate | None = None
        self._closed = False
        self._build_ui()
        self.wizard.set_restore_handler(self.restore_selected_issue)
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll)
        self._timer.start()
        QTimer.singleShot(0, self.refresh)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(12)
        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(TitleLabel("Dataset", self))
        self.summary = CaptionLabel("Scanning immutable offline runs...", self)
        titles.addWidget(self.summary)
        header.addLayout(titles)
        header.addStretch(1)
        self.refresh_button = PushButton("Refresh", self)
        self.refresh_button.setIcon(FluentIcon.SYNC)
        self.refresh_button.clicked.connect(self.refresh)
        header.addWidget(self.refresh_button)
        layout.addLayout(header)
        layout.addWidget(self._build_filters())

        self.table = TableWidget(self)
        self.table.setObjectName("datasetCandidateTable")
        self.table.setColumnCount(12)
        self.table.setHorizontalHeaderLabels(
            [
                "Session",
                "Clip",
                "Run",
                "Processing",
                "VIO",
                "Hands",
                "Objects",
                "Interpolation",
                "Phases",
                "Quality",
                "Annotation",
                "Candidate",
            ]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(TableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(TableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        layout.addWidget(self.table, 1)

        self.wizard = DatasetBuildWizard(self)
        self.wizard.inspect_button.clicked.connect(self.inspect_selected)
        self.wizard.publish_button.clicked.connect(self.publish_selected)
        layout.addWidget(self.wizard)

    def _build_filters(self) -> QWidget:
        panel = QWidget(self)
        panel.setObjectName("datasetFilters")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        selectors = QHBoxLayout()
        selectors.setSpacing(8)
        self.run_filter = LineEdit(panel)
        self.run_filter.setPlaceholderText("processing run")
        self.run_filter.setClearButtonEnabled(True)
        selectors.addWidget(self.run_filter, 2)
        self.profile_filter = _filter_combo(panel, "All task profiles")
        self.calibration_filter = _filter_combo(panel, "All calibrations")
        self.quality_filter = _filter_combo(
            panel,
            "All quality states",
            ("ready", "review", "blocked"),
        )
        self.annotation_filter = _filter_combo(
            panel,
            "All annotation states",
            ("published", "missing"),
        )
        self.candidate_filter = _filter_combo(
            panel,
            "All candidate states",
            ("candidate", "excluded"),
        )
        for widget in (
            self.profile_filter,
            self.calibration_filter,
            self.quality_filter,
            self.annotation_filter,
            self.candidate_filter,
        ):
            selectors.addWidget(widget, 1)
        layout.addLayout(selectors)

        thresholds = QHBoxLayout()
        thresholds.setSpacing(8)
        thresholds.addWidget(CaptionLabel("Minimum coverage", panel))
        self.vio_coverage_filter = _coverage_spin(panel, "VIO")
        self.hand_coverage_filter = _coverage_spin(panel, "Hands")
        self.object_coverage_filter = _coverage_spin(panel, "Objects")
        for widget in (
            self.vio_coverage_filter,
            self.hand_coverage_filter,
            self.object_coverage_filter,
        ):
            thresholds.addWidget(widget)
        thresholds.addStretch(1)
        layout.addLayout(thresholds)
        self.run_filter.textChanged.connect(self._apply_filters)
        for widget in (
            self.profile_filter,
            self.calibration_filter,
            self.quality_filter,
            self.annotation_filter,
            self.candidate_filter,
        ):
            widget.currentIndexChanged.connect(self._apply_filters)
        for widget in (
            self.vio_coverage_filter,
            self.hand_coverage_filter,
            self.object_coverage_filter,
        ):
            widget.valueChanged.connect(self._apply_filters)
        return panel

    def close_resources(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def refresh(self) -> None:
        if self._closed:
            return
        if self._scan_future is not None and not self._scan_future.done():
            return
        self.refresh_button.setEnabled(False)
        self.summary.setText("Scanning immutable offline runs...")
        self._scan_future = self._executor.submit(self.catalog.scan)

    def inspect_selected(self) -> None:
        row = self.table.currentRow()
        if not 0 <= row < len(self._rows):
            return
        selected = self._rows[row]
        if selected.annotation_revision_id is None:
            self._error("Publish an annotation revision before building this candidate.")
            return
        if self._candidate_future is not None and not self._candidate_future.done():
            return
        self.wizard.detail.setText("Running quality gates and assembling virtual episodes...")
        self._candidate_future = self._executor.submit(
            self.builder.candidate,
            selected.session_id,
            selected.run_id,
            annotation_revision_id=selected.annotation_revision_id,
        )

    def publish_selected(self) -> None:
        candidate = self._candidate
        if candidate is None or not candidate.publishable:
            return
        dataset_id = self.wizard.dataset_id.text().strip()
        if not dataset_id:
            dataset_id = time.strftime("egoglass-%Y%m%d-%H%M%S", time.localtime())
            self.wizard.dataset_id.setText(dataset_id)
        if self._publish_future is not None and not self._publish_future.done():
            return
        self.wizard.publish_button.setEnabled(False)
        self.wizard.detail.setText("Publishing immutable JSONL and manifest artifacts...")
        self._publish_future = self._executor.submit(
            self.builder.publish,
            dataset_id,
            (candidate,),
        )

    def restore_selected_issue(self) -> None:
        candidate = self._candidate
        issue_id = self.wizard.selected_issue_id()
        operator = self.wizard.operator.text().strip()
        reason = self.wizard.reason.text().strip()
        if candidate is None or issue_id is None:
            return
        if not operator or not reason:
            self._error("Reviewer and restore reason are required.")
            return
        try:
            report = DatasetQualityChecker.restore_soft_issue(
                candidate.quality,
                issue_id,
                operator=operator,
                reason=reason,
            )
        except (KeyError, ValueError) as error:
            self._error(str(error))
            return
        self.wizard.publish_button.setEnabled(False)
        self.wizard.restore_button.setEnabled(False)
        self.wizard.detail.setText("Applying review and rebuilding virtual episodes...")
        self._candidate_future = self._executor.submit(
            self.builder.candidate,
            candidate.session_id,
            candidate.run_id,
            annotation_revision_id=candidate.annotation_revision_id,
            quality_report=report,
        )

    def _poll(self) -> None:
        if self._scan_future is not None and self._scan_future.done():
            future, self._scan_future = self._scan_future, None
            self.refresh_button.setEnabled(True)
            try:
                self._set_rows(future.result())
            except Exception as error:
                self._error(str(error))
        if self._candidate_future is not None and self._candidate_future.done():
            future, self._candidate_future = self._candidate_future, None
            try:
                self._candidate = future.result()
                self.wizard.set_candidate(self._candidate)
            except Exception as error:
                self._candidate = None
                self.wizard.set_candidate(None)
                self._error(str(error))
        if self._publish_future is not None and self._publish_future.done():
            future, self._publish_future = self._publish_future, None
            try:
                result = future.result()
            except Exception as error:
                self.wizard.set_candidate(self._candidate)
                self._error(str(error))
            else:
                self.wizard.detail.setText(
                    f"Published {result.episode_count} episodes and "
                    f"{result.sample_count} samples to {result.output_directory}"
                )
                InfoBar.success(
                    "Dataset published",
                    str(result.output_directory),
                    duration=5000,
                    parent=self.window(),
                )

    def _set_rows(self, rows: tuple[DatasetCandidateSummary, ...]) -> None:
        self._all_rows = rows
        self._set_filter_options(rows)
        self._apply_filters()

    def _render_rows(self, rows: tuple[DatasetCandidateSummary, ...]) -> None:
        self._rows = rows
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = (
                row.session_id[:8],
                row.clip_id[:8] if row.clip_id else "all clips",
                row.run_id[-8:],
                row.processing_state,
                row.vio_state,
                f"{row.hand_coverage:.0%}",
                f"{row.object_coverage:.0%}",
                f"{row.interpolation_ratio:.0%}",
                str(row.phase_count),
                row.quality_state,
                row.annotation_revision_id[:8] if row.annotation_revision_id else "missing",
                "ready" if row.candidate else "no",
            )
            for column, value in enumerate(values):
                self.table.setItem(row_index, column, QTableWidgetItem(value))
        self.summary.setText(f"{len(rows)} offline processing runs")
        self._selection_changed()

    def _set_filter_options(self, rows: tuple[DatasetCandidateSummary, ...]) -> None:
        _replace_filter_values(
            self.profile_filter,
            "All task profiles",
            tuple(sorted({row.task_profile_id for row in rows if row.task_profile_id})),
        )
        _replace_filter_values(
            self.calibration_filter,
            "All calibrations",
            tuple(
                sorted(
                    {
                        row.calibration_profile_id
                        for row in rows
                        if row.calibration_profile_id
                    }
                )
            ),
        )

    def _apply_filters(self, _value: object = None) -> None:
        run_query = self.run_filter.text().strip().lower()
        profile = self.profile_filter.currentData()
        calibration = self.calibration_filter.currentData()
        quality = self.quality_filter.currentData()
        annotation = self.annotation_filter.currentData()
        candidate_state = self.candidate_filter.currentData()
        minimum_vio = self.vio_coverage_filter.value() / 100.0
        minimum_hands = self.hand_coverage_filter.value() / 100.0
        minimum_objects = self.object_coverage_filter.value() / 100.0
        rows = tuple(
            row
            for row in self._all_rows
            if (not run_query or run_query in row.run_id.lower())
            and (profile is None or row.task_profile_id == profile)
            and (calibration is None or row.calibration_profile_id == calibration)
            and (quality is None or row.quality_state == quality)
            and (
                annotation is None
                or (annotation == "published") == (row.annotation_revision_id is not None)
            )
            and (
                candidate_state is None
                or (candidate_state == "candidate") == row.candidate
            )
            and row.vio_coverage >= minimum_vio
            and row.hand_coverage >= minimum_hands
            and row.object_coverage >= minimum_objects
        )
        self._render_rows(rows)

    def _selection_changed(self) -> None:
        self._candidate = None
        self.wizard.set_candidate(None)

    def _error(self, detail: str) -> None:
        InfoBar.error("Dataset operation failed", detail, duration=5000, parent=self.window())


def _filter_combo(
    parent: QWidget,
    all_text: str,
    values: tuple[str, ...] = (),
) -> ComboBox:
    combo = ComboBox(parent)
    combo.addItem(all_text, userData=None)
    for value in values:
        combo.addItem(value, userData=value)
    return combo


def _coverage_spin(parent: QWidget, prefix: str) -> SpinBox:
    spin = SpinBox(parent)
    spin.setRange(0, 100)
    spin.setPrefix(f"{prefix} ")
    spin.setSuffix("%")
    spin.setFixedWidth(120)
    return spin


def _replace_filter_values(
    combo: ComboBox,
    all_text: str,
    values: tuple[str, ...],
) -> None:
    selected = combo.currentData()
    combo.blockSignals(True)
    combo.clear()
    combo.addItem(all_text, userData=None)
    for value in values:
        combo.addItem(value, userData=value)
    index = combo.findData(selected)
    combo.setCurrentIndex(max(0, index))
    combo.blockSignals(False)
