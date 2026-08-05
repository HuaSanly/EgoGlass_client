from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

from PyQt6.QtCore import QSignalBlocker, Qt, QTimer
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    ComboBox,
    DoubleSpinBox,
    FluentIcon,
    InfoBadge,
    InfoBar,
    InfoLevel,
    LineEdit,
    NavigationInterface,
    PrimaryPushButton,
    PushButton,
    SettingCard,
    SettingCardGroup,
    SmoothScrollArea,
    SpinBox,
    StrongBodyLabel,
    SwitchButton,
    TitleLabel,
    TransparentToolButton,
)

from ui.application.runtime_host import UnifiedRuntimeHost
from ui.configuration import ConfigApplyResult, ConfigImpact, ConfigSnapshot
from ui.processing import DEFAULT_PRESETS

# Display metadata is intentionally compact and may contain long localized descriptions.
# ruff: noqa: E501
FieldKind = Literal["bool", "choice", "int", "float", "text", "path", "file"]
_SETTINGS_BACKGROUND = "#f7f9fc"
_OFFLINE_FIXED_FIELDS = frozenset(
    {
        "detector",
        "fallback_detector",
        "device",
        "require_cuda",
        "enable_cuda_amp",
        "require_hamer",
        "allow_mediapipe_reconstruction_fallback",
        "vitpose_variant",
    }
)


@dataclass(frozen=True, slots=True)
class _FieldSpec:
    module_id: str
    group: str
    path: str
    title: str
    description: str
    kind: FieldKind
    options: tuple[tuple[str, object], ...] = ()
    minimum: float = 0
    maximum: float = 100
    step: float = 1
    decimals: int = 0
    suffix: str = ""


_MODULES = (
    ("client_runtime", "\u5ba2\u6237\u7aef\u4e0e\u7f51\u5173", FluentIcon.DEVELOPER_TOOLS, "Manage gateway, discovery, and recording storage"),
    ("sensor_preprocessing", "\u4f20\u611f\u5668\u9884\u5904\u7406", FluentIcon.CAMERA, "Manage calibration, decoding, undistortion, and IMU buffering"),
    ("live_hand_tracking", "\u5b9e\u65f6\u624b\u90e8\u8ffd\u8e2a", FluentIcon.ROBOT, "Low-latency MediaPipe hand tracking for the live preview"),
    ("offline_hand_tracking", "\u79bb\u7ebf\u624b\u90e8\u8ffd\u8e2a", FluentIcon.PHOTO, "Quality-first ViTPose-H and HaMeR processing for recorded video"),
    ("video_processing", "\u79bb\u7ebf\u89c6\u9891\u5904\u7406", FluentIcon.MOVIE, "Manage defaults for newly submitted offline jobs"),
)

_FIELD_DISPLAY: dict[str, tuple[str, str, str, str, tuple[tuple[str, object], ...]]] = {
    "host": ("Basic", "Gateway listen address", "HTTP and WebRTC gateway binding", "", ()),
    "port": ("Basic", "Gateway port", "TCP port for HTTP, signaling, and media", "", ()),
    "discovery_port": ("Basic", "Discovery port", "UDP port used for LAN discovery", "", ()),
    "enable_discovery": ("Basic", "Enable Discovery", "Allow glasses to discover this client", "", ()),
    "recordings_root": ("Basic", "Recording root", "Storage for sessions, jobs, and exports", "", ()),
    "calibration_file": ("Basic", "Sensor calibration", "Camera and IMU calibration JSON", "", ()),
    "recorded.verify_media_hashes": ("Basic", "Verify media SHA256", "Verify recorded media before offline processing", "", ()),
    "recorded.decode_threads": ("Basic", "Decode threads", "Zero lets FFmpeg choose automatically", " threads", ()),
    "image.undistort": ("Basic", "Enable undistortion", "Correct spatial-perception input with calibration", "", ()),
    "image.interpolation": ("Basic", "Remap interpolation", "Interpolation method for undistortion", "", (("Nearest", "nearest"), ("Linear", "linear"), ("Cubic", "cubic"), ("Area", "area"), ("Lanczos 4", "lanczos4"))),
    "image.border_mode": ("Basic", "Border mode", "Pixel fill strategy after undistortion", "", (("Constant", "constant"), ("Replicate", "replicate"), ("Reflect", "reflect"), ("Reflect 101", "reflect_101"))),
    "live.max_pending_imu_samples": ("Advanced", "Max pending IMU samples", "Upper bound for live IMU buffering", " samples", ()),
    "runtime.enabled": ("Basic", "Enable live inference", "Controls live display only; offline jobs are unaffected", "", ()),
    "runtime.max_live_inference_fps": ("Basic", "Max live inference FPS", "Limit live hand inference to keep display responsive", " FPS", ()),
    "algorithm.detector": ("Basic", "Primary detector", "Preferred 2D hand detector", "", (("MediaPipe", "mediapipe"), ("ViTPose", "vitpose"))),
    "algorithm.fallback_detector": ("Basic", "Fallback detector", "Used when the primary detector returns no result", "", (("Disabled", "none"), ("MediaPipe", "mediapipe"))),
    "algorithm.detector_keypoint_confidence": ("Basic", "Keypoint confidence", "Reject 2D keypoints below this threshold", "", ()),
    "algorithm.detector_min_valid_keypoints": ("Basic", "Minimum valid keypoints", "Minimum keypoints required before reconstruction", "", ()),
    "algorithm.minimum_hand_confidence": ("Basic", "Minimum hand confidence", "Reject hand results below this threshold", "", ()),
    "algorithm.device": ("Basic", "Inference device", "Device used by hand models", "", (("CUDA", "cuda"), ("CPU", "cpu"))),
    "algorithm.model_directory": ("Advanced", "Model directory", "HaMeR, ViTPose, MediaPipe, and MANO weights", "", ()),
    "algorithm.require_cuda": ("Advanced", "Require CUDA", "Block model startup when CUDA is unavailable", "", ()),
    "algorithm.enable_cuda_amp": ("Advanced", "Enable CUDA AMP", "Reduce GPU memory use and inference latency", "", ()),
    "algorithm.require_hamer": ("Advanced", "Require HaMeR", "Do not fall back to MediaPipe reconstruction", "", ()),
    "algorithm.download_models": ("Advanced", "Allow model downloads", "Download missing pinned model files at startup", "", ()),
    "algorithm.allow_mediapipe_reconstruction_fallback": ("Advanced", "Allow MediaPipe fallback", "Use MediaPipe 3D keypoints when HaMeR fails", "", ()),
    "algorithm.vitpose_variant": ("Advanced", "ViTPose variant", "Select the ViTPose model size", "", (("Small", "s"), ("Base", "b"), ("Large", "l"), ("Huge", "h"))),
    "algorithm.detector_bbox_padding_ratio": ("Advanced", "BBox padding ratio", "Padding around detector boxes before reconstruction", "", ()),
    "algorithm.detector_min_bbox_dimension_ratio": ("Advanced", "Minimum BBox dimension", "Reject boxes that are too small in the frame", "", ()),
    "algorithm.detector_max_bbox_area_ratio": ("Advanced", "Maximum BBox area", "Reject boxes that occupy too much of the frame", "", ()),
    "algorithm.physical_wrist_to_middle_mcp_m": ("Advanced", "Physical hand scale", "Reference wrist to middle-MCP distance", " m", ()),
    "algorithm.minimum_depth_m": ("Advanced", "Minimum depth", "Minimum hand depth in camera coordinates", " m", ()),
    "algorithm.maximum_depth_m": ("Advanced", "Maximum depth", "Maximum hand depth in camera coordinates", " m", ()),
    "algorithm.grasp_ratio_threshold": ("Advanced", "Grasp ratio threshold", "Normalized distance threshold for grasp state", "", ()),
    "algorithm.sources.hamer_code_revision": ("Model revisions", "HaMeR code revision", "Pinned upstream commit", "", ()),
    "algorithm.sources.hamer_weights_revision": ("Model revisions", "HaMeR weights revision", "Pinned model revision", "", ()),
    "algorithm.sources.vitpose_code_revision": ("Model revisions", "ViTPose code revision", "Pinned upstream commit", "", ()),
    "algorithm.sources.vitpose_weights_revision": ("Model revisions", "ViTPose weights revision", "Pinned model revision", "", ()),
    "algorithm.sources.mediapipe_weights_revision": ("Model revisions", "MediaPipe weights revision", "Pinned model revision", "", ()),
    "algorithm.sources.mano_weights_revision": ("Model revisions", "MANO weights revision", "Pinned model revision", "", ()),
    "default_preset_id": ("Basic", "Default processing preset", "Used for new workbench jobs", "", tuple((preset.display_name, preset.preset_id) for preset in DEFAULT_PRESETS)),
    "auto_enqueue_on_session_complete": ("Basic", "Auto-enqueue completed sessions", "Queue a completed session for offline processing", "", ()),
    "default_output_result_type": ("Basic", "Default result type", "Structured result format written by jobs", "", (("Structured results", "structured_results"),)),
}
def _field(
    module_id: str,
    path: str,
    kind: FieldKind,
    *,
    minimum: float = 0,
    maximum: float = 100,
    step: float = 1,
    decimals: int = 0,
) -> _FieldSpec:
    display = _FIELD_DISPLAY.get(path) or _FIELD_DISPLAY[f"algorithm.{path}"]
    return _FieldSpec(
        module_id,
        display[0],
        path,
        display[1],
        display[2],
        kind,
        options=display[4],
        minimum=minimum,
        maximum=maximum,
        step=step,
        decimals=decimals,
        suffix=display[3],
    )


def _hand_algorithm_fields(module_id: str, prefix: str) -> tuple[_FieldSpec, ...]:
    return (
        _field(module_id, f"{prefix}detector", "choice"),
        _field(module_id, f"{prefix}fallback_detector", "choice"),
        _field(module_id, f"{prefix}detector_keypoint_confidence", "float", maximum=1, step=0.01, decimals=2),
        _field(module_id, f"{prefix}detector_min_valid_keypoints", "int", minimum=1, maximum=21),
        _field(module_id, f"{prefix}minimum_hand_confidence", "float", maximum=1, step=0.01, decimals=2),
        _field(module_id, f"{prefix}device", "choice"),
        _field(module_id, f"{prefix}model_directory", "path"),
        _field(module_id, f"{prefix}require_cuda", "bool"),
        _field(module_id, f"{prefix}enable_cuda_amp", "bool"),
        _field(module_id, f"{prefix}require_hamer", "bool"),
        _field(module_id, f"{prefix}download_models", "bool"),
        _field(module_id, f"{prefix}allow_mediapipe_reconstruction_fallback", "bool"),
        _field(module_id, f"{prefix}vitpose_variant", "choice"),
        _field(module_id, f"{prefix}detector_bbox_padding_ratio", "float", maximum=2, step=0.05, decimals=2),
        _field(module_id, f"{prefix}detector_min_bbox_dimension_ratio", "float", maximum=1, step=0.01, decimals=2),
        _field(module_id, f"{prefix}detector_max_bbox_area_ratio", "float", minimum=0.01, maximum=1, step=0.01, decimals=2),
        _field(module_id, f"{prefix}physical_wrist_to_middle_mcp_m", "float", minimum=0.001, maximum=0.5, step=0.001, decimals=3),
        _field(module_id, f"{prefix}minimum_depth_m", "float", minimum=0.001, maximum=10, step=0.01, decimals=3),
        _field(module_id, f"{prefix}maximum_depth_m", "float", minimum=0.01, maximum=20, step=0.1, decimals=2),
        _field(module_id, f"{prefix}grasp_ratio_threshold", "float", minimum=0.01, maximum=10, step=0.05, decimals=2),
        _field(module_id, f"{prefix}sources.hamer_code_revision", "text"),
        _field(module_id, f"{prefix}sources.hamer_weights_revision", "text"),
        _field(module_id, f"{prefix}sources.vitpose_code_revision", "text"),
        _field(module_id, f"{prefix}sources.vitpose_weights_revision", "text"),
        _field(module_id, f"{prefix}sources.mediapipe_weights_revision", "text"),
        _field(module_id, f"{prefix}sources.mano_weights_revision", "text"),
    )


_FIELDS = (
    _field("client_runtime", "host", "text"),
    _field("client_runtime", "port", "int", minimum=1, maximum=65_535),
    _field("client_runtime", "discovery_port", "int", minimum=1, maximum=65_535),
    _field("client_runtime", "enable_discovery", "bool"),
    _field("client_runtime", "recordings_root", "path"),
    _field("sensor_preprocessing", "calibration_file", "file"),
    _field("sensor_preprocessing", "recorded.verify_media_hashes", "bool"),
    _field("sensor_preprocessing", "recorded.decode_threads", "int", maximum=32),
    _field("sensor_preprocessing", "image.undistort", "bool"),
    _field("sensor_preprocessing", "image.interpolation", "choice"),
    _field("sensor_preprocessing", "image.border_mode", "choice"),
    _field("sensor_preprocessing", "live.max_pending_imu_samples", "int", minimum=1, maximum=32_768),
    _field("live_hand_tracking", "runtime.enabled", "bool"),
    _field("live_hand_tracking", "runtime.max_live_inference_fps", "float", minimum=0.1, maximum=60, step=0.5, decimals=1),
    *_hand_algorithm_fields("live_hand_tracking", "algorithm."),
    *_hand_algorithm_fields("offline_hand_tracking", ""),
    _field("video_processing", "default_preset_id", "choice"),
    _field("video_processing", "auto_enqueue_on_session_complete", "bool"),
    _field("video_processing", "default_output_result_type", "choice"),
)
_IMPACT_LABELS = {
    ConfigImpact.IMMEDIATE: "Immediate",
    ConfigImpact.NEXT_SESSION: "Next session",
    ConfigImpact.NEXT_TASK: "Next task",
    ConfigImpact.RESTART_CLIENT: "Restart client",
}


def _save_result_detail_clean(result: ConfigApplyResult) -> str:
    if not result.changed_modules:
        return "No configuration changes"
    details = [f"Saved {len(result.changed_modules)} module(s)"]
    if result.immediate_applied:
        details.append("Immediate values applied")
    if result.pending_next_session:
        details.append("Some values apply next session")
    if result.pending_next_task:
        details.append("Some values apply next task")
    if result.pending_restart:
        details.append("Some values require a client restart")
    return "; ".join(details)


class _ParameterCard(SettingCard):
    def __init__(
        self,
        spec: _FieldSpec,
        impact: ConfigImpact,
        changed_callback: object,
        parent: QWidget,
    ) -> None:
        super().__init__(FluentIcon.SETTING, spec.title, spec.description, parent)
        self.spec = spec
        self._loading = False
        self.impact_badge = InfoBadge.info(_IMPACT_LABELS[impact], self)
        self.hBoxLayout.addWidget(self.impact_badge, 0, Qt.AlignmentFlag.AlignVCenter)
        self.editor = self._create_editor()
        self.hBoxLayout.addWidget(self.editor, 0, Qt.AlignmentFlag.AlignVCenter)
        self._connect_changed(changed_callback)

    def _create_editor(self) -> QWidget:
        spec = self.spec
        if spec.kind == "bool":
            editor = SwitchButton(self)
            editor.setOnText("On")
            editor.setOffText("Off")
            return editor
        if spec.kind == "choice":
            editor = ComboBox(self)
            for label, value in spec.options:
                editor.addItem(label, userData=value)
            editor.setMinimumWidth(180)
            return editor
        if spec.kind == "int":
            editor = SpinBox(self)
            editor.setRange(int(spec.minimum), int(spec.maximum))
            editor.setSingleStep(max(1, int(spec.step)))
            editor.setSuffix(spec.suffix)
            editor.setMinimumWidth(150)
            return editor
        if spec.kind == "float":
            editor = DoubleSpinBox(self)
            editor.setRange(spec.minimum, spec.maximum)
            editor.setSingleStep(spec.step)
            editor.setDecimals(spec.decimals)
            editor.setSuffix(spec.suffix)
            editor.setMinimumWidth(150)
            return editor
        editor = LineEdit(self)
        editor.setClearButtonEnabled(True)
        editor.setMinimumWidth(280)
        if spec.kind in {"path", "file"}:
            editor.setReadOnly(True)
            button = TransparentToolButton(FluentIcon.FOLDER, self)
            button.setToolTip("Choose file" if spec.kind == "file" else "Choose folder")
            button.clicked.connect(self._choose_path)
            self.hBoxLayout.addWidget(button, 0, Qt.AlignmentFlag.AlignVCenter)
        return editor

    def _choose_path(self) -> None:
        current = self.editor.text() if isinstance(self.editor, LineEdit) else ""
        if self.spec.kind == "file":
            selected, _ = QFileDialog.getOpenFileName(
                self, "Choose sensor calibration", current, "JSON (*.json)"
            )
        else:
            selected = QFileDialog.getExistingDirectory(self, "Choose folder", current)
        if selected and isinstance(self.editor, LineEdit):
            self.editor.setText(selected)

    def _connect_changed(self, callback: object) -> None:
        if isinstance(self.editor, SwitchButton):
            self.editor.checkedChanged.connect(callback)  # type: ignore[arg-type]
        elif isinstance(self.editor, ComboBox):
            self.editor.currentIndexChanged.connect(callback)  # type: ignore[arg-type]
        elif isinstance(self.editor, (SpinBox, DoubleSpinBox)):
            self.editor.valueChanged.connect(callback)  # type: ignore[arg-type]
        elif isinstance(self.editor, LineEdit):
            self.editor.textChanged.connect(callback)  # type: ignore[arg-type]
    def value(self) -> object:
        if isinstance(self.editor, SwitchButton):
            return self.editor.isChecked()
        if isinstance(self.editor, ComboBox):
            return self.editor.currentData()
        if isinstance(self.editor, (SpinBox, DoubleSpinBox)):
            return self.editor.value()
        if isinstance(self.editor, LineEdit):
            return self.editor.text().strip()
        raise TypeError("unsupported parameter editor")

    def set_value(self, value: object) -> None:
        blocker = QSignalBlocker(self.editor)
        if isinstance(self.editor, SwitchButton):
            self.editor.setChecked(bool(value))
        elif isinstance(self.editor, ComboBox):
            index = self.editor.findData(value)
            self.editor.setCurrentIndex(max(0, index))
        elif isinstance(self.editor, SpinBox):
            self.editor.setValue(int(value or 0))
        elif isinstance(self.editor, DoubleSpinBox):
            self.editor.setValue(float(value or 0))
        elif isinstance(self.editor, LineEdit):
            self.editor.setText(str(value or ""))
        del blocker


class _ModulePage(QWidget):
    def __init__(
        self,
        module_id: str,
        impacts: Mapping[str, ConfigImpact],
        changed_callback: object,
        parent: QWidget,
    ) -> None:
        super().__init__(parent)
        self.module_id = module_id
        self.cards: dict[str, _ParameterCard] = {}
        self.setObjectName(f"{module_id}SettingsPage")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"#{self.objectName()} {{ background-color: {_SETTINGS_BACKGROUND}; }}"
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        scroll = SmoothScrollArea(self)
        scroll.setObjectName(f"{module_id}SettingsScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(SmoothScrollArea.Shape.NoFrame)
        scroll.setStyleSheet(
            f"QScrollArea#{scroll.objectName()} {{ "
            f"background-color: {_SETTINGS_BACKGROUND}; border: none; }}"
        )
        scroll.viewport().setStyleSheet(
            f"background-color: {_SETTINGS_BACKGROUND}; border: none;"
        )
        content = QWidget(scroll)
        content.setObjectName(f"{module_id}SettingsContent")
        content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        content.setStyleSheet(
            f"#{content.objectName()} {{ background-color: {_SETTINGS_BACKGROUND}; }}"
        )
        layout = QVBoxLayout(content)
        layout.setContentsMargins(2, 2, 12, 24)
        layout.setSpacing(18)
        groups: dict[str, SettingCardGroup] = {}
        for spec in (item for item in _FIELDS if item.module_id == module_id):
            group = groups.get(spec.group)
            if group is None:
                group = SettingCardGroup(spec.group, content)
                group.setObjectName(f"{module_id}{spec.group}SettingsGroup")
                group.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
                group.setStyleSheet(
                    group.styleSheet()
                    + f"\nSettingCardGroup {{ background-color: {_SETTINGS_BACKGROUND}; }}"
                )
                groups[spec.group] = group
                layout.addWidget(group)
            impact = _field_impact(impacts, spec.path)
            card = _ParameterCard(spec, impact, changed_callback, group)
            self.cards[spec.path] = card
            group.addSettingCard(card)
        if module_id == "client_runtime":
            self.runtime_group = SettingCardGroup("运行状态", content)
            self.token_card, self.token_value = _readonly_card(
                self.runtime_group,
                FluentIcon.CERTIFICATE,
                "配对令牌",
                "令牌不会以明文显示或在此页面修改",
            )
            self.thread_card, self.thread_value = _readonly_card(
                self.runtime_group,
                FluentIcon.SPEED_HIGH,
                "运行线程与端口",
                "客户端主运行时的当前状态",
            )
            self.device_card, self.device_value = _readonly_card(
                self.runtime_group,
                FluentIcon.CONNECT,
                "眼镜输入",
                "设备采集参数只读，由眼镜端管理",
            )
            for card in (self.token_card, self.thread_card, self.device_card):
                self.runtime_group.addSettingCard(card)
            layout.addWidget(self.runtime_group)
        elif module_id == "live_hand_tracking":
            self.runtime_group = SettingCardGroup("实时状态", content)
            self.tracker_card, self.tracker_value = _readonly_card(
                self.runtime_group,
                FluentIcon.ROBOT,
                "实时 Tracker",
                "当前模型状态、检测器和累计推理帧",
            )
            self.runtime_group.addSettingCard(self.tracker_card)
            layout.addWidget(self.runtime_group)
        layout.addStretch(1)
        scroll.setWidget(content)
        root.addWidget(scroll)


class ProcessingSettingsView(QWidget):
    """Edit all supported client configuration through one typed service."""

    def __init__(self, runtime: UnifiedRuntimeHost, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("processingSettingsView")
        self.runtime = runtime
        self._snapshot: ConfigSnapshot | None = None
        self._module_values: dict[str, dict[str, object]] = {}
        self._baseline_values: dict[str, dict[str, object]] = {}
        self._current_module_id = _MODULES[0][0]
        self._save_pending = False
        self._save_with_apply = False
        self._save_future: Future[ConfigApplyResult] | None = None
        self._pending_impact_text = "无待生效更改"
        self.pages: dict[str, _ModulePage] = {}
        self._build_ui()
        self._set_static_labels()
        self._load_snapshot()
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._update_status)
        self._timer.start()

    def _set_static_labels(self) -> None:
        """Apply stable Chinese labels to controls created by the view."""

        self.module_title.setText(_MODULES[0][1])
        self.module_description.setText(_MODULES[0][3])
        self.dirty_badge.setText("已保存")
        self.pending_badge.setText("无待生效更改")
        self.revision_label.setText("配置 revision --")
        self.discard_button.setText("放弃更改")
        self.reset_button.setText("恢复模块默认值")
        self.validate_button.setText("校验配置")
        self.save_button.setText("保存")
        self.apply_button.setText("保存并应用")
        client_page = self.pages.get("client_runtime")
        if client_page is not None and hasattr(client_page, "runtime_group"):
            client_page.runtime_group.titleLabel.setText("运行状态")
            client_page.token_card.setTitle("配对令牌")
            client_page.token_card.setContent("令牌不会以明文显示或在此页面修改")
            client_page.thread_card.setTitle("运行线程与端口")
            client_page.thread_card.setContent("客户端主运行时的当前状态")
            client_page.device_card.setTitle("设备输入")
            client_page.device_card.setContent("设备采集参数只读，由眼镜端管理")

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.module_navigation = NavigationInterface(
            self,
            showMenuButton=False,
            showReturnButton=False,
            collapsible=False,
        )
        self.module_navigation.setObjectName("settingsModuleNavigation")
        self.module_navigation.setFixedWidth(214)
        self.module_navigation.setExpandWidth(214)
        root.addWidget(self.module_navigation)

        body = QWidget(self)
        body.setObjectName("settingsBody")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        body.setStyleSheet(
            f"#settingsBody {{ background-color: {_SETTINGS_BACKGROUND}; }}"
        )
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(28, 20, 28, 18)
        body_layout.setSpacing(12)
        header = QHBoxLayout()
        title_column = QVBoxLayout()
        title_column.setSpacing(2)
        self.module_title = TitleLabel(self)
        self.module_description = CaptionLabel(self)
        title_column.addWidget(self.module_title)
        title_column.addWidget(self.module_description)
        header.addLayout(title_column)
        header.addStretch(1)
        self.dirty_badge = InfoBadge.success("已保存", self)
        self.pending_badge = InfoBadge.info("无待生效更改", self)
        header.addWidget(self.dirty_badge, 0, Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self.pending_badge, 0, Qt.AlignmentFlag.AlignVCenter)
        body_layout.addLayout(header)

        self.stack = QStackedWidget(self)
        self.stack.setObjectName("settingsPageStack")
        self.stack.setFrameShape(QFrame.Shape.NoFrame)
        self.stack.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.stack.setStyleSheet(
            f"#settingsPageStack {{ background-color: {_SETTINGS_BACKGROUND}; border: none; }}"
        )
        body_layout.addWidget(self.stack, 1)
        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        self.revision_label = CaptionLabel("配置 revision --", self)
        action_row.addWidget(self.revision_label)
        action_row.addStretch(1)
        self.discard_button = PushButton("放弃更改", self)
        self.reset_button = PushButton("恢复模块默认值", self)
        self.validate_button = PushButton("校验配置", self)
        self.save_button = PushButton("保存", self)
        self.apply_button = PrimaryPushButton("保存并应用", self)
        self.discard_button.clicked.connect(self._discard)
        self.reset_button.clicked.connect(self._restore_defaults)
        self.validate_button.clicked.connect(self._validate)
        self.save_button.clicked.connect(self._save)
        self.apply_button.clicked.connect(self._save_and_apply)
        for button in (
            self.discard_button,
            self.reset_button,
            self.validate_button,
            self.save_button,
            self.apply_button,
        ):
            action_row.addWidget(button)
        body_layout.addLayout(action_row)
        root.addWidget(body, 1)

        for module_id, name, icon, _description in _MODULES:
            self.module_navigation.addItem(
                module_id,
                icon,
                name,
                onClick=partial(self._select_module, module_id),
            )
            page = _ModulePage(module_id, {}, self._form_changed, self.stack)
            self.pages[module_id] = page
            self.stack.addWidget(page)
        self.module_navigation.expand(useAni=False)
        self.module_navigation.setCurrentItem(self._current_module_id)
        self._select_module(self._current_module_id)

    def close_resources(self) -> None:
        self._timer.stop()

    def _load_snapshot(
        self,
        snapshot: ConfigSnapshot | None = None,
        *,
        preserve_baseline: bool = False,
    ) -> None:
        snapshot_provider = getattr(self.runtime, "configuration_snapshot", None)
        if not callable(snapshot_provider):
            self._set_available(False)
            self.module_description.setText("配置服务不可用")
            return
        try:
            snapshot = snapshot or snapshot_provider()
        except Exception as error:
            self._set_available(False)
            InfoBar.error("无法加载配置", str(error), duration=5000, parent=self.window())
            return
        self._snapshot = snapshot
        self._module_values = {
            module.module_id: _mutable(module.values) for module in snapshot.modules
        }
        if not preserve_baseline:
            self._baseline_values = {
                module_id: _mutable(values) for module_id, values in self._module_values.items()
            }
        for module in snapshot.modules:
            page = self.pages[module.module_id]
            for path, card in page.cards.items():
                card.set_value(_nested_get(self._module_values[module.module_id], path))
                impact = _field_impact(module.field_impacts, path)
                card.impact_badge.setText(_IMPACT_LABELS[impact])
        self.revision_label.setText(f"配置 revision {snapshot.revision}")
        self._set_available(True)
        self._sync_dirty_state()

    def _set_available(self, available: bool) -> None:
        for button in (
            self.discard_button,
            self.reset_button,
            self.validate_button,
            self.save_button,
            self.apply_button,
        ):
            button.setEnabled(available)
        for page in self.pages.values():
            for path, card in page.cards.items():
                fixed_offline_policy = (
                    page.module_id == "offline_hand_tracking"
                    and path in _OFFLINE_FIXED_FIELDS
                )
                card.editor.setEnabled(available and not fixed_offline_policy)

    def _select_module(self, module_id: str) -> None:
        self._current_module_id = module_id
        self.stack.setCurrentWidget(self.pages[module_id])
        self.module_navigation.setCurrentItem(module_id)
        for item_id, name, _icon, description in _MODULES:
            if item_id == module_id:
                self.module_title.setText(name)
                self.module_description.setText(description)
                break

    def _form_changed(self, _value: object = None) -> None:
        if self._snapshot is None:
            return
        module_id = self._current_module_id
        self._collect_module(module_id)
        stage = getattr(self.runtime, "stage_configuration", None)
        if not callable(stage):
            return
        try:
            stage({module_id: self._module_values[module_id]})
        except (KeyError, TypeError) as error:
            InfoBar.error("无法暂存参数", str(error), duration=4000, parent=self.window())
            return
        self._sync_dirty_state()

    def _collect_module(self, module_id: str) -> None:
        values = self._module_values[module_id]
        for path, card in self.pages[module_id].cards.items():
            _nested_set(values, path, card.value())

    def _collect_all(self) -> dict[str, dict[str, object]]:
        for module_id in self.pages:
            self._collect_module(module_id)
        return self._module_values

    def _sync_dirty_state(self) -> None:
        snapshot_provider = getattr(self.runtime, "configuration_snapshot", None)
        dirty = bool(callable(snapshot_provider) and snapshot_provider().dirty)
        self.dirty_badge.setText("有未保存更改" if dirty else "已保存")
        self.dirty_badge.setLevel(InfoLevel.WARNING if dirty else InfoLevel.SUCCESS)
        self.discard_button.setEnabled(bool(dirty))
        self.save_button.setEnabled(bool(dirty))
        self.apply_button.setEnabled(bool(dirty))
        if not dirty and not self._save_pending:
            self.pending_badge.setText("无待生效更改")

    def _discard(self) -> None:
        discard = getattr(self.runtime, "discard_configuration", None)
        if not callable(discard):
            return
        self._load_snapshot(discard())
        InfoBar.info(
            "已放弃更改", "参数已恢复到最近一次保存状态", duration=2200, parent=self.window()
        )

    def _restore_defaults(self) -> None:
        restore = getattr(self.runtime, "restore_configuration_defaults", None)
        if not callable(restore):
            return
        try:
            self._load_snapshot(
                restore(self._current_module_id),
                preserve_baseline=True,
            )
        except KeyError as error:
            InfoBar.error("无法恢复默认值", str(error), duration=4000, parent=self.window())

    def _validate(self) -> None:
        stage = getattr(self.runtime, "stage_configuration", None)
        validate = getattr(self.runtime, "validate_configuration", None)
        if not callable(stage) or not callable(validate):
            return
        stage(self._collect_all())
        issues = validate()
        if issues:
            self._show_validation_issues(issues)
            return
        InfoBar.success(
            "配置有效", "所有模块参数均通过类型和边界校验", duration=2600, parent=self.window()
        )

    def _save(self) -> None:
        self._request_save(apply=False)

    def _save_and_apply(self) -> None:
        self._request_save(apply=True)

    def _request_save(self, *, apply: bool) -> None:
        stage = getattr(self.runtime, "stage_configuration", None)
        validate = getattr(self.runtime, "validate_configuration", None)
        request_save = getattr(self.runtime, "request_configuration_save", None)
        if not callable(stage) or not callable(validate) or not callable(request_save):
            return
        try:
            stage(self._collect_all())
            issues = validate()
            if issues:
                self._show_validation_issues(issues)
                return
            self._pending_impact_text = self._changed_impact_text()
            future = request_save(None, apply=apply)
            if not isinstance(future, Future):
                raise TypeError("configuration save must return a Future")
        except Exception as error:
            InfoBar.error("配置保存失败", str(error), duration=5000, parent=self.window())
            return
        self._save_pending = True
        self._save_future = future
        self._save_with_apply = apply
        self.pending_badge.setText("正在保存并应用" if apply else "正在保存")
        self.save_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.apply_button.setText("正在应用…" if apply else "保存并应用")

    def _show_validation_issues(self, issues: object) -> None:
        issue_values = tuple(issues)  # type: ignore[arg-type]
        if not issue_values:
            return
        first = issue_values[0]
        if first.module_id in self.pages:
            self._select_module(first.module_id)
        detail = "；".join(
            f"{issue.field_path or issue.module_id}: {issue.message}" for issue in issue_values[:3]
        )
        if len(issue_values) > 3:
            detail += f"；另有 {len(issue_values) - 3} 项"
        InfoBar.error("参数校验失败", detail, duration=6500, parent=self.window())

    def _update_status(self) -> None:
        self._update_client_runtime_status()
        self._update_live_tracker_status()
        future = self._save_future
        if not self._save_pending or future is None or not future.done():
            return
        self._save_pending = False
        self._save_future = None
        self.apply_button.setText("保存并应用")
        try:
            result = future.result()
        except Exception as error:
            InfoBar.error("配置保存失败", str(error), duration=5000, parent=self.window())
            self._sync_dirty_state()
            return
        self._load_snapshot()
        self.pending_badge.setText(self._pending_impact_text)
        title = "配置已保存并应用" if self._save_with_apply else "配置已保存"
        InfoBar.success(
            title,
            _save_result_detail_clean(result),
            duration=3500,
            parent=self.window(),
        )

    def _changed_impact_text(self) -> str:
        if self._snapshot is None:
            return "无待生效更改"
        impacts: set[ConfigImpact] = set()
        for module in self._snapshot.modules:
            current = self._module_values[module.module_id]
            original = self._baseline_values[module.module_id]
            for path in self.pages[module.module_id].cards:
                if _nested_get(current, path) != _nested_get(original, path):
                    impacts.add(_field_impact(module.field_impacts, path))
        labels = [
            _IMPACT_LABELS[impact]
            for impact in (
                ConfigImpact.IMMEDIATE,
                ConfigImpact.NEXT_SESSION,
                ConfigImpact.NEXT_TASK,
                ConfigImpact.RESTART_CLIENT,
            )
            if impact in impacts
        ]
        return "、".join(labels) if labels else "无待生效更改"

    def _update_client_runtime_status(self) -> None:
        page = self.pages.get("client_runtime")
        if page is None or not hasattr(page, "token_value"):
            return
        pairing_token = getattr(self.runtime, "pairing_token", None)
        page.token_value.setText("已设置" if pairing_token else "未设置")
        snapshot = self.runtime.snapshot()
        config = getattr(self.runtime, "config", None)
        port = getattr(config, "port", "--")
        page.thread_value.setText(f"{'运行中' if snapshot.server_ready else '未就绪'} · TCP {port}")
        webrtc = snapshot.webrtc
        if webrtc is None:
            page.device_value.setText("未连接")
        else:
            size = (
                f"{webrtc.width} × {webrtc.height}"
                if webrtc.width and webrtc.height
                else "分辨率 --"
            )
            codec = webrtc.video_codec or "编码 --"
            page.device_value.setText(f"{webrtc.phase.value} · {size} · {codec}")

    def _update_live_tracker_status(self) -> None:
        page = self.pages.get("live_hand_tracking")
        if page is None or not hasattr(page, "tracker_value"):
            return
        perception = self.runtime.snapshot().perception
        state = str(perception.get("state", "--"))
        count = int(perception.get("live_inferences", 0) or 0)
        detector = "--"
        latest = perception.get("latest_result")
        if isinstance(latest, Mapping):
            detector = str(latest.get("detector_backend", "--"))
        page.tracker_value.setText(f"{state} · {detector} · {count} 帧")

def _readonly_card(
    parent: SettingCardGroup,
    icon: FluentIcon,
    title: str,
    description: str,
) -> tuple[SettingCard, BodyLabel]:
    card = SettingCard(icon, title, description, parent)
    value = StrongBodyLabel("--", card)
    value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    card.hBoxLayout.addWidget(value)
    return card, value


def _field_impact(impacts: Mapping[str, ConfigImpact], path: str) -> ConfigImpact:
    if path in impacts:
        return impacts[path]
    prefixes = path.split(".")
    for index in range(len(prefixes), 0, -1):
        wildcard = ".".join(prefixes[:index]) + ".*"
        if wildcard in impacts:
            return impacts[wildcard]
    return impacts.get("*", ConfigImpact.NEXT_TASK)


def _nested_get(values: Mapping[str, object], path: str) -> object:
    current: object = values
    for part in path.split("."):
        if not isinstance(current, Mapping):
            raise KeyError(path)
        current = current[part]
    return current


def _nested_set(values: dict[str, object], path: str, value: object) -> None:
    parts = path.split(".")
    current = values
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _mutable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _mutable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
