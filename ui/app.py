from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication
from qfluentwidgets import FluentIcon, FluentWindow, Theme, setTheme, setThemeColor

from perception.configuration import ClientRuntimeConfig

from .logging_config import configure_logging
from .runtime import RuntimeConfig, UnifiedRuntimeHost
from .views.home import HomeView
from .views.processing_pipeline import ProcessingPipelineView
from .views.processing_settings import ProcessingSettingsView
from .views.video_processing import VideoProcessingView

LOGGER = logging.getLogger(__name__)


class MainWindow(FluentWindow):
    """Own the processing-first Fluent workspace and orderly runtime shutdown."""

    def __init__(self, runtime: UnifiedRuntimeHost) -> None:
        super().__init__()
        application = QApplication.instance()
        if application is not None:
            application.setFont(QFont("Microsoft YaHei UI", 10))
        self.runtime = runtime
        self.processing_view = VideoProcessingView(runtime, self)
        self.pipeline_view = ProcessingPipelineView(runtime, self)
        self.home_view = HomeView(runtime, self)
        self.settings_view = ProcessingSettingsView(runtime, self)
        self._shutdown_complete = False

        self.setWindowTitle("EgoGlass")
        self.resize(1440, 900)
        self.setMinimumSize(1180, 780)
        self.setMicaEffectEnabled(False)
        self.navigationInterface.setAcrylicEnabled(True)
        self.navigationInterface.setExpandWidth(220)
        QTimer.singleShot(0, self._set_navigation_labels)
        self.addSubInterface(self.processing_view, FluentIcon.MOVIE, "视频处理")
        self.addSubInterface(self.pipeline_view, FluentIcon.HISTORY, "流水线")
        self.addSubInterface(self.home_view, FluentIcon.CAMERA, "实时采集")
        self.addSubInterface(self.settings_view, FluentIcon.SETTING, "系统设置")

    def _set_navigation_labels(self) -> None:
        labels = {
            "videoProcessingView": "\u89c6\u9891\u5904\u7406",
            "processingPipelineView": "\u6d41\u6c34\u7ebf",
            "homeView": "\u5b9e\u65f6\u91c7\u96c6",
            "processingSettingsView": "\u53c2\u6570\u7ba1\u7406",
        }
        for route_key, label in labels.items():
            item = self.navigationInterface.widget(route_key)
            if item is not None:
                item.setText(label)
                item.setToolTip(label)

    def shutdown(self) -> None:
        if self._shutdown_complete:
            return
        self._shutdown_complete = True
        errors: list[Exception] = []
        for operation in (
            self.processing_view.close_resources,
            self.pipeline_view.close_resources,
            self.home_view.close_resources,
            self.settings_view.close_resources,
            self.runtime.stop,
        ):
            try:
                operation()
            except Exception as error:
                errors.append(error)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("native client shutdown failed", errors)

    def closeEvent(self, event: object) -> None:
        try:
            self.shutdown()
        except Exception:
            LOGGER.exception("native client shutdown failed")
        super().closeEvent(event)


class NativeApplication:
    """Own QApplication while gateway, recording, and perception stay in-process."""

    def __init__(self, runtime: UnifiedRuntimeHost) -> None:
        self.runtime = runtime
        self.window: MainWindow | None = None

    def run(self) -> int:
        application = QApplication.instance() or QApplication(sys.argv[:1])
        application.setApplicationName("EgoGlass")
        application.setOrganizationName("EgoGlass")
        application.setFont(QFont("Microsoft YaHei UI", 10))
        setTheme(Theme.LIGHT)
        setThemeColor("#2f6fed")

        self.runtime.start()
        self.window = MainWindow(self.runtime)
        self.window.show()
        try:
            return application.exec()
        finally:
            if self.window is not None:
                self.window.shutdown()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the native EgoGlass client")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--discovery-port", type=int)
    parser.add_argument("--disable-discovery", action="store_true")
    parser.add_argument("--pairing-token", default=os.environ.get("EGOGLASS_PAIRING_TOKEN"))
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--recordings-root",
        type=Path,
        default=(
            Path(os.environ["EGOGLASS_RECORDINGS_ROOT"])
            if "EGOGLASS_RECORDINGS_ROOT" in os.environ
            else None
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke_test:
        return 0
    configure_logging()
    configured = ClientRuntimeConfig.load("config/client-runtime.yaml")
    recordings_root = (
        Path(args.recordings_root)
        if args.recordings_root is not None
        else configured.recordings_root
    )
    runtime = UnifiedRuntimeHost(
        RuntimeConfig(
            host=configured.host if args.host is None else args.host,
            port=configured.port if args.port is None else args.port,
            discovery_port=(
                configured.discovery_port
                if args.discovery_port is None
                else args.discovery_port
            ),
            recordings_root=recordings_root,
            pairing_token=args.pairing_token,
            enable_discovery=configured.enable_discovery and not args.disable_discovery,
        )
    )
    try:
        return NativeApplication(runtime).run()
    except KeyboardInterrupt:
        LOGGER.info("shutdown requested")
        runtime.stop()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
