from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication
from qfluentwidgets import FluentIcon, FluentWindow, Theme, setTheme, setThemeColor

from .logging_config import configure_logging
from .runtime import RuntimeConfig, UnifiedRuntimeHost
from .views.home import HomeView

LOGGER = logging.getLogger(__name__)


class MainWindow(FluentWindow):
    """Own the single Fluent home interface and orderly runtime shutdown."""

    def __init__(self, runtime: UnifiedRuntimeHost) -> None:
        super().__init__()
        application = QApplication.instance()
        if application is not None:
            application.setFont(QFont("Microsoft YaHei UI", 10))
        self.runtime = runtime
        self.home_view = HomeView(runtime, self)
        self._shutdown_complete = False

        self.setWindowTitle("EgoGlass")
        self.resize(1440, 900)
        self.setMinimumSize(1180, 780)
        self.setMicaEffectEnabled(False)
        self.navigationInterface.setAcrylicEnabled(True)
        self.navigationInterface.setExpandWidth(220)
        self.addSubInterface(self.home_view, FluentIcon.HOME, "主页")

    def shutdown(self) -> None:
        if self._shutdown_complete:
            return
        self._shutdown_complete = True
        self.home_view.close_resources()
        self.runtime.stop()

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
        runtime.stop()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
