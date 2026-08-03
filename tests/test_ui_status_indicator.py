from PyQt6.QtWidgets import QApplication
from qfluentwidgets import BodyLabel, FluentIcon, IconInfoBadge, InfoLevel

from ui.widgets.status_indicator import StatusIndicator


def test_status_indicator_uses_fluent_icon_and_text_components(
    qt_application: QApplication,
) -> None:
    indicator = StatusIndicator(
        "等待",
        FluentIcon.SYNC,
        level=InfoLevel.INFOAMTION,
    )
    try:
        assert indicator.findChild(IconInfoBadge, "statusIndicatorIcon") is (
            indicator.icon_badge
        )
        assert indicator.findChild(BodyLabel, "statusIndicatorText") is (
            indicator.text_label
        )
        assert indicator.text() == "等待"
        assert indicator.level() is InfoLevel.INFOAMTION

        indicator.setText("已同步")
        indicator.setLevel(InfoLevel.SUCCESS)
        indicator.setIcon(FluentIcon.COMPLETED)
        qt_application.processEvents()

        assert indicator.text() == "已同步"
        assert indicator.level() is InfoLevel.SUCCESS
        assert indicator.icon_badge.size().width() == 18
        assert indicator.icon_badge.iconSize().width() == 10
        assert not indicator.grab().isNull()
    finally:
        indicator.close()
