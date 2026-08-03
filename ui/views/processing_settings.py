from __future__ import annotations

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from qfluentwidgets import (
    ComboBox,
    FluentIcon,
    InfoBar,
    SettingCard,
    SettingCardGroup,
    SwitchButton,
    TitleLabel,
)

from perception.video_processing import DEFAULT_PRESETS
from ui.runtime import UnifiedRuntimeHost


class ProcessingSettingsView(QWidget):
    """Persist the defaults consumed by newly submitted offline jobs."""

    def __init__(
        self,
        runtime: UnifiedRuntimeHost,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("processingSettingsView")
        self.runtime = runtime
        self._last_revision = -1
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._update_status)
        self._timer.start()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 18, 24, 18)
        root.setSpacing(16)
        root.addWidget(TitleLabel("系统设置", self))
        group = SettingCardGroup("离线视频处理", self)
        self.preset_card = SettingCard(
            FluentIcon.MOVIE,
            "默认处理方案",
            "从视频工作台提交任务时使用；切换不会修改历史运行。",
            group,
        )
        self.preset_combo = ComboBox(self.preset_card)
        for preset in DEFAULT_PRESETS:
            self.preset_combo.addItem(preset.display_name, userData=preset.preset_id)
        self.preset_combo.setMinimumWidth(220)
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        self.preset_card.hBoxLayout.addWidget(self.preset_combo)
        group.addSettingCard(self.preset_card)

        self.auto_card = SettingCard(
            FluentIcon.SYNC,
            "会话完成后自动入队",
            "默认关闭；开启后，新完成的完整会话会自动进入流水线。",
            group,
        )
        self.auto_switch = SwitchButton(self.auto_card)
        self.auto_switch.setOnText("开启")
        self.auto_switch.setOffText("关闭")
        self.auto_switch.checkedChanged.connect(self._auto_queue_changed)
        self.auto_card.hBoxLayout.addWidget(self.auto_switch)
        group.addSettingCard(self.auto_card)
        root.addWidget(group)
        root.addStretch(1)

    def close_resources(self) -> None:
        self._timer.stop()

    def _update_status(self) -> None:
        processing = self.runtime.snapshot().processing
        if processing is not None and processing.revision != self._last_revision:
            self._last_revision = processing.revision
            preset_index = self.preset_combo.findData(processing.default_preset_id)
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(max(0, preset_index))
            self.preset_combo.blockSignals(False)
            self.auto_switch.blockSignals(True)
            self.auto_switch.setChecked(processing.auto_enqueue_on_session_complete)
            self.auto_switch.blockSignals(False)
        if self.isVisible():
            for result in self.runtime.command_results():
                if result.succeeded:
                    InfoBar.success(
                        "设置已保存",
                        result.detail,
                        duration=2200,
                        parent=self.window(),
                    )
                else:
                    InfoBar.error("设置失败", result.detail, duration=4500, parent=self.window())

    def _preset_changed(self, _index: int) -> None:
        preset_id = self.preset_combo.currentData()
        if isinstance(preset_id, str):
            self.runtime.request_processing_default_preset(preset_id)

    def _auto_queue_changed(self, enabled: bool) -> None:
        self.runtime.request_processing_auto_queue(enabled)
