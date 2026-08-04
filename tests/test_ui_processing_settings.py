from __future__ import annotations

import shutil
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

from PyQt6.QtWidgets import QWidget

from perception.configuration import ConfigApplyResult, ConfigurationService
from ui.state import RuntimeSnapshot
from ui.views.processing_settings import ProcessingSettingsView


class _SettingsRuntimeStub:
    def __init__(self, service: ConfigurationService) -> None:
        self.configuration_service = service
        self.pairing_token = "test-token"
        self.config = SimpleNamespace(port=8770)

    def configuration_snapshot(self):  # type: ignore[no-untyped-def]
        return self.configuration_service.snapshot()

    def stage_configuration(self, values):  # type: ignore[no-untyped-def]
        return self.configuration_service.stage(values)

    def validate_configuration(self, values=None):  # type: ignore[no-untyped-def]
        return self.configuration_service.validate(values)

    def discard_configuration(self):  # type: ignore[no-untyped-def]
        self.configuration_service.discard()
        return self.configuration_service.snapshot()

    def restore_configuration_defaults(self, module_id):  # type: ignore[no-untyped-def]
        self.configuration_service.restore_defaults(module_id)
        return self.configuration_service.snapshot()

    def request_configuration_save(  # type: ignore[no-untyped-def]
        self,
        values=None,
        *,
        apply: bool,
    ) -> Future[ConfigApplyResult]:
        future: Future[ConfigApplyResult] = Future()
        try:
            result = self.configuration_service.save(values)
        except Exception as error:
            future.set_exception(error)
        else:
            future.set_result(result)
        return future

    def snapshot(self) -> RuntimeSnapshot:
        return RuntimeSnapshot(server_ready=True)


def _runtime(tmp_path: Path) -> _SettingsRuntimeStub:
    source = Path(__file__).parents[1] / "config"
    destination = tmp_path / "config"
    shutil.copytree(source, destination)
    service = ConfigurationService(
        destination,
        recordings_root=tmp_path / "recordings",
        jobs_database_path=tmp_path / "recordings" / ".processing" / "jobs.sqlite3",
    )
    return _SettingsRuntimeStub(service)


def test_parameter_page_registers_five_module_routes(
    qt_application,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    view = ProcessingSettingsView(_runtime(tmp_path))  # type: ignore[arg-type]
    try:
        assert tuple(view.pages) == (
            "client_runtime",
            "sensor_preprocessing",
            "live_hand_tracking",
            "offline_hand_tracking",
            "video_processing",
        )
        assert view.stack.count() == 5
        live_cards = view.pages["live_hand_tracking"].cards
        offline_cards = view.pages["offline_hand_tracking"].cards
        assert "runtime.max_live_inference_fps" in live_cards
        assert "algorithm.detector" in live_cards
        assert "runtime.max_live_inference_fps" not in offline_cards
        assert "detector" in offline_cards
        assert "algorithm.detector" not in offline_cards
        assert not offline_cards["detector"].editor.isEnabled()
        assert not offline_cards["vitpose_variant"].editor.isEnabled()
        assert not offline_cards["enable_cuda_amp"].editor.isEnabled()
        assert offline_cards["minimum_hand_confidence"].editor.isEnabled()
        assert (
            "default_inference_stride_frames"
            not in view.pages["video_processing"].cards
        )
        source = Path(__file__).parents[1] / "ui" / "views" / "processing_settings.py"
        assert "QLabel" not in source.read_text(encoding="utf-8")
    finally:
        view.close_resources()
        view.close()


def test_parameter_page_tracks_and_discards_dirty_values(
    qt_application,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    view = ProcessingSettingsView(runtime)  # type: ignore[arg-type]
    try:
        view._select_module("live_hand_tracking")
        card = view.pages["live_hand_tracking"].cards[
            "runtime.max_live_inference_fps"
        ]
        card.editor.setValue(7.5)  # type: ignore[attr-defined]

        assert runtime.configuration_snapshot().dirty
        assert view.dirty_badge.text() == "有未保存更改"
        assert view.discard_button.isEnabled()

        view._discard()
        assert not runtime.configuration_snapshot().dirty
        assert card.value() == 6.0
        assert not view.discard_button.isEnabled()
    finally:
        view.close_resources()
        view.close()


def test_parameter_page_does_not_render_an_opaque_outer_frame(
    qt_application,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    view = ProcessingSettingsView(_runtime(tmp_path))  # type: ignore[arg-type]
    try:
        page = view.pages["sensor_preprocessing"]
        assert "background-color: #f7f9fc" in page.styleSheet()
        content = page.findChild(QWidget, "sensor_preprocessingSettingsContent")
        assert content is not None
        assert "background-color: #f7f9fc" in content.styleSheet()
        body = view.findChild(QWidget, "settingsBody")
        assert body is not None
        assert "background-color: #f7f9fc" in body.styleSheet()
    finally:
        view.close_resources()
        view.close()


def test_parameter_page_validates_and_saves_through_runtime_boundary(
    qt_application,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    view = ProcessingSettingsView(runtime)  # type: ignore[arg-type]
    try:
        view.show()
        view._select_module("video_processing")
        auto_enqueue = view.pages["video_processing"].cards[
            "auto_enqueue_on_session_complete"
        ]
        auto_enqueue.editor.setChecked(True)  # type: ignore[attr-defined]
        revision = runtime.configuration_snapshot().revision

        view._save_and_apply()
        view._update_status()

        snapshot = runtime.configuration_snapshot()
        assert snapshot.revision == revision + 1
        assert not snapshot.dirty
        assert snapshot.require_module("video_processing").values[
            "auto_enqueue_on_session_complete"
        ] is True
        assert view.apply_button.text() == "保存并应用"
    finally:
        view.close_resources()
        view.close()
