from pathlib import Path
from types import SimpleNamespace

import dearpygui.dearpygui as dpg
import numpy as np

from annotation.store import AnnotationStore
from ingest_gateway.live_frames import LiveFrame
from ui.app import NativeApplication
from ui.views.library import LibraryView
from ui.views.live import _perception_result_key
from ui.widgets.video_surface import VideoSurface, fit_image_geometry


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


def test_library_refresh_button_requests_a_real_background_scan() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.refresh_count = 0

        def request_library_refresh(self) -> None:
            self.refresh_count += 1

    runtime = Runtime()
    view = LibraryView.__new__(LibraryView)
    view.runtime = runtime  # type: ignore[assignment]

    view._force_refresh()

    assert runtime.refresh_count == 1


def test_native_app_updates_only_the_active_tab(monkeypatch) -> None:
    class View:
        def __init__(self) -> None:
            self.snapshots: list[object] = []

        def update(self, snapshot: object) -> None:
            self.snapshots.append(snapshot)

    application = NativeApplication(SimpleNamespace())  # type: ignore[arg-type]
    application.live_view = View()  # type: ignore[assignment]
    application.library_view = View()  # type: ignore[assignment]
    application.annotation_view = View()  # type: ignore[assignment]
    application.diagnostics_view = View()  # type: ignore[assignment]
    snapshot = object()
    monkeypatch.setattr(dpg, "get_value", lambda _tag: "live-tab")

    application._update_active_view(snapshot)  # type: ignore[arg-type]

    assert application.live_view.snapshots == [snapshot]  # type: ignore[union-attr]
    assert application.library_view.snapshots == []  # type: ignore[union-attr]
    assert application.annotation_view.snapshots == []  # type: ignore[union-attr]
    assert application.diagnostics_view.snapshots == []  # type: ignore[union-attr]


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
        assert surface.status().recent_upload_fps > 0
    finally:
        dpg.destroy_context()


def test_video_surface_letterboxes_four_by_three_without_cropping_or_stretching() -> None:
    geometry = fit_image_geometry(960, 540, 640, 480)

    assert geometry.minimum == (120.0, 0.0)
    assert geometry.maximum == (840.0, 540.0)
    assert geometry.scale == 1.125


def test_video_surface_swaps_raw_texture_buffers_instead_of_mutating_visible_frame() -> None:
    def frame(index: int, value: int) -> LiveFrame:
        image = np.full((3, 4, 3), value, dtype=np.uint8)
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
        initial_tag = surface._texture_tag
        assert surface.update_frame(frame(0, 51))
        first_frame_tag = surface._texture_tag
        first_frame_buffer = surface._texture_buffers[surface._front_texture_index]
        first_frame_snapshot = first_frame_buffer.copy()

        assert first_frame_tag != initial_tag
        assert surface.update_frame(frame(1, 204))
        assert surface._texture_tag == initial_tag
        np.testing.assert_array_equal(first_frame_buffer, first_frame_snapshot)
    finally:
        dpg.destroy_context()


def test_perception_result_identity_changes_for_each_inference_frame() -> None:
    first = {
        "session_id": "session",
        "sequence_id": "connection",
        "frame_index": 10,
    }
    second = {**first, "frame_index": 11}

    assert _perception_result_key(first) == ("session", "connection", 10)
    assert _perception_result_key(second) == ("session", "connection", 11)
