from __future__ import annotations

from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import BodyLabel, FluentIconBase, IconInfoBadge, InfoLevel


class StatusIndicator(QWidget):
    """Fluent status icon paired with readable, unboxed text."""

    def __init__(
        self,
        text: str,
        icon: FluentIconBase,
        parent: QWidget | None = None,
        *,
        level: InfoLevel = InfoLevel.INFOAMTION,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("statusIndicator")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.icon_badge = IconInfoBadge(icon, self, level)
        self.icon_badge.setObjectName("statusIndicatorIcon")
        self.icon_badge.setFixedSize(18, 18)
        self.icon_badge.setIconSize(QSize(10, 10))
        layout.addWidget(self.icon_badge)

        self.text_label = BodyLabel(text, self)
        self.text_label.setObjectName("statusIndicatorText")
        layout.addWidget(self.text_label)

        self.setMinimumHeight(24)

    def text(self) -> str:
        return self.text_label.text()

    def setText(self, text: str) -> None:
        self.text_label.setText(text)

    def setLevel(self, level: InfoLevel) -> None:
        self.icon_badge.setLevel(level)

    def level(self) -> InfoLevel:
        return self.icon_badge.level

    def setIcon(self, icon: FluentIconBase) -> None:
        self.icon_badge.setIcon(icon)
