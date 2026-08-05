from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QApplication, QWidget

from ui.application.runtime_state import RuntimeSnapshot
from ui.configuration import ConfigurationService
from ui.views.processing_settings import ProcessingSettingsView


class _ReadOnlyRuntime:
    def __init__(self, service: ConfigurationService) -> None:
        self.service = service
        self.config = SimpleNamespace(port=8770)
        self.pairing_token = "eval-token"

    def configuration_snapshot(self):  # type: ignore[no-untyped-def]
        return self.service.snapshot()

    def stage_configuration(self, values):  # type: ignore[no-untyped-def]
        return self.service.stage(values)

    def validate_configuration(self, values=None):  # type: ignore[no-untyped-def]
        return self.service.validate(values)

    def discard_configuration(self):  # type: ignore[no-untyped-def]
        self.service.discard()
        return self.service.snapshot()

    def restore_configuration_defaults(self, module_id):  # type: ignore[no-untyped-def]
        self.service.restore_defaults(module_id)
        return self.service.snapshot()

    def snapshot(self) -> RuntimeSnapshot:
        return RuntimeSnapshot(server_ready=True)


@pytest.mark.parametrize("size", ((1280, 800), (1440, 900), (1920, 1080)))
def test_parameter_workspace_keeps_navigation_and_actions_visible(
    qt_application: QApplication,
    tmp_path: Path,
    size: tuple[int, int],
) -> None:
    config = tmp_path / "config"
    shutil.copytree(Path(__file__).parents[1] / "config", config)
    service = ConfigurationService(
        config,
        recordings_root=tmp_path / "recordings",
        jobs_database_path=tmp_path / "recordings" / ".processing" / "jobs.sqlite3",
    )
    view = ProcessingSettingsView(_ReadOnlyRuntime(service))  # type: ignore[arg-type]
    try:
        view.resize(*size)
        view.show()
        qt_application.processEvents()
        assert view.module_navigation.width() == 214
        assert view.stack.width() > 700
        assert view.stack.height() > 600
        assert view.apply_button.isVisible()
        assert "background-color: #f7f9fc" in view.pages["sensor_preprocessing"].styleSheet()
        body = view.findChild(QWidget, "settingsBody")
        assert body is not None
        assert "background-color: #f7f9fc" in body.styleSheet()
        for module_id in view.pages:
            view._select_module(module_id)
            qt_application.processEvents()
            assert view.stack.currentWidget() is view.pages[module_id]
            assert view.module_title.text()
    finally:
        view.close_resources()
        view.close()


def test_parameter_workspace_uses_fluent_labels_and_typed_runtime_boundary() -> None:
    source = (Path(__file__).parents[1] / "ui" / "views" / "processing_settings.py").read_text(
        encoding="utf-8"
    )
    assert "NavigationInterface" in source
    assert "SettingCardGroup" in source
    assert "SmoothScrollArea" in source
    assert "QLabel" not in source
    assert "yaml.safe_load" not in source
    assert "configuration_snapshot" in source
    assert "stage_configuration" in source
    assert "request_configuration_save" in source
