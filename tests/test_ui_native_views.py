from pathlib import Path
from types import SimpleNamespace

import dearpygui.dearpygui as dpg

from annotation.store import AnnotationStore
from ui.app import NativeApplication


def test_native_views_cover_library_annotation_and_diagnostics() -> None:
    repository = Path(__file__).parents[1]
    app = (repository / "ui" / "app.py").read_text(encoding="utf-8")
    library = (repository / "ui" / "views" / "library.py").read_text(encoding="utf-8")
    annotation = (repository / "ui" / "views" / "annotation.py").read_text(
        encoding="utf-8"
    )
    diagnostics = (repository / "ui" / "views" / "diagnostics.py").read_text(
        encoding="utf-8"
    )

    assert "LibraryView" in app
    assert "AnnotationView" in app
    assert "DiagnosticsView" in app
    assert "self.live_view.open_clip" in library
    assert "AnnotationController" in annotation
    assert "self.live_view.open_clip" in annotation
    assert "add_phase" in annotation
    assert "controller.publish" in annotation
    assert "snapshot.recent_events" in diagnostics


def test_native_ui_has_one_video_surface_instance() -> None:
    repository = Path(__file__).parents[1]
    ui_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (repository / "ui").rglob("*.py")
        if path.name != "video_surface.py"
    )

    assert ui_sources.count("VideoSurface(") == 1


def test_every_native_view_builds_in_real_dearpygui_context(tmp_path: Path) -> None:
    runtime = SimpleNamespace(annotation=AnnotationStore(tmp_path))
    application = NativeApplication(runtime)  # type: ignore[arg-type]
    dpg.create_context()
    try:
        application._build()
        assert dpg.does_item_exist("live-tab")
        assert dpg.does_item_exist("library-tab")
        assert dpg.does_item_exist("annotation-tab")
        assert dpg.does_item_exist("diagnostics-tab")
    finally:
        if application.live_view is not None:
            application.live_view.close()
        dpg.destroy_context()
