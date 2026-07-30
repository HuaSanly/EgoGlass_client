from pathlib import Path
from types import SimpleNamespace

import dearpygui.dearpygui as dpg
import numpy as np

from annotation.store import AnnotationStore
from ingest_gateway.live_frames import LiveFrame
from ui.app import NativeApplication
from ui.widgets.video_surface import VideoSurface


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


def test_video_surface_replaces_dynamic_resolution_texture_without_reusing_alias() -> None:
    def frame(index: int, width: int, height: int) -> LiveFrame:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        return LiveFrame("session", "connection", index, index, index, image)

    dpg.create_context()
    try:
        with dpg.window() as parent:
            surface = VideoSurface(
                parent=parent,
                width=8,
                height=6,
                source_width=4,
                source_height=3,
            )
        original_tag = surface._texture_tag
        assert surface.update_frame(frame(0, 2, 2))
        second_tag = surface._texture_tag
        assert second_tag != original_tag
        assert dpg.does_item_exist(second_tag)
        assert surface.update_frame(frame(1, 4, 3))
        assert surface._texture_tag not in {original_tag, second_tag}
        assert dpg.does_item_exist(surface._texture_tag)
    finally:
        dpg.destroy_context()
