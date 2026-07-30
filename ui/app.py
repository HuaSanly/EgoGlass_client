from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

import dearpygui.dearpygui as dpg

from .logging_config import configure_logging
from .runtime import RuntimeConfig, UnifiedRuntimeHost
from .theme import bind_application_font, bind_application_theme
from .views.annotation import AnnotationView
from .views.diagnostics import DiagnosticsView
from .views.library import LibraryView
from .views.live import LiveView

LOGGER = logging.getLogger(__name__)


class NativeApplication:
    """Own the Dear PyGui main thread and the unified in-process runtime."""

    def __init__(self, runtime: UnifiedRuntimeHost) -> None:
        self.runtime = runtime
        self.live_view: LiveView | None = None
        self.library_view: LibraryView | None = None
        self.annotation_view: AnnotationView | None = None
        self.diagnostics_view: DiagnosticsView | None = None

    def run(self) -> int:
        self.runtime.start()
        dpg.create_context()
        dpg.configure_app(manual_callback_management=True)
        try:
            bind_application_font()
            bind_application_theme()
            self._build()
            dpg.create_viewport(
                title="EgoGlass",
                width=1440,
                height=900,
                min_width=1180,
                min_height=760,
                vsync=True,
            )
            dpg.setup_dearpygui()
            dpg.show_viewport()
            dpg.set_primary_window("egoglass-primary-window", True)
            while dpg.is_dearpygui_running():
                dpg.run_callbacks(dpg.get_callback_queue())
                snapshot = self.runtime.snapshot()
                self._update_active_view(snapshot)
                for result in self.runtime.command_results():
                    dpg.set_value("global-command-status", result.detail)
                    dpg.configure_item(
                        "global-command-status",
                        color=(93, 199, 164) if result.succeeded else (235, 104, 92),
                    )
                dpg.render_dearpygui_frame()
            return 0
        finally:
            if self.live_view is not None:
                self.live_view.close()
            self.runtime.stop()
            dpg.destroy_context()

    def _update_active_view(self, snapshot: object) -> None:
        active_tab = dpg.get_value("main-navigation")
        views = {
            "live-tab": self.live_view,
            "library-tab": self.library_view,
            "annotation-tab": self.annotation_view,
            "diagnostics-tab": self.diagnostics_view,
        }
        view = views.get(active_tab)
        if view is not None:
            view.update(snapshot)

    def _build(self) -> None:
        with dpg.window(tag="egoglass-primary-window", no_title_bar=True, no_move=True):
            with dpg.group(horizontal=True):
                dpg.add_text("EGOGLASS", color=(232, 236, 235))
                dpg.add_text("CLIENT RUNTIME", color=(112, 124, 124))
                dpg.add_spacer(width=24)
                dpg.add_text("就绪", tag="global-command-status", color=(93, 199, 164))
            dpg.add_separator()
            with dpg.tab_bar(tag="main-navigation"):
                with dpg.tab(label="实时", tag="live-tab"):
                    self.live_view = LiveView(self.runtime, dpg.last_item())
                with dpg.tab(label="资料库", tag="library-tab"):
                    assert self.live_view is not None
                    self.library_view = LibraryView(
                        self.runtime,
                        self.live_view,
                        dpg.last_item(),
                    )
                with dpg.tab(label="标注", tag="annotation-tab"):
                    assert self.live_view is not None
                    self.annotation_view = AnnotationView(
                        self.runtime,
                        self.live_view,
                        dpg.last_item(),
                    )
                with dpg.tab(label="诊断", tag="diagnostics-tab"):
                    self.diagnostics_view = DiagnosticsView(dpg.last_item())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the native EgoGlass client")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--discovery-port", type=int, default=8771)
    parser.add_argument("--disable-discovery", action="store_true")
    parser.add_argument("--pairing-token", default=os.environ.get("EGOGLASS_PAIRING_TOKEN"))
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--recordings-root",
        type=Path,
        default=Path(os.environ.get("EGOGLASS_RECORDINGS_ROOT", "local-data/recordings")),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke_test:
        return 0
    configure_logging()
    runtime = UnifiedRuntimeHost(
        RuntimeConfig(
            host=args.host,
            port=args.port,
            discovery_port=args.discovery_port,
            recordings_root=args.recordings_root,
            pairing_token=args.pairing_token,
            enable_discovery=not args.disable_discovery,
        )
    )
    try:
        return NativeApplication(runtime).run()
    except KeyboardInterrupt:
        LOGGER.info("shutdown requested")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
