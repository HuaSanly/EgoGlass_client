from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication
from qfluentwidgets import FluentIcon, FluentWindow, Theme, setTheme, setThemeColor

from .application.runtime_host import RuntimeConfig, UnifiedRuntimeHost
from .logging_config import configure_logging
from .views.home import HomeView
from .views.recording_library import RecordingLibraryView

LOGGER = logging.getLogger(__name__)
_CONFIG_KEYS = {
    "schema_version",
    "host",
    "port",
    "discovery_port",
    "enable_discovery",
    "recordings_root",
}


class MainWindow(FluentWindow):
    """Own the two-page recording client and orderly runtime shutdown."""

    def __init__(self, runtime: UnifiedRuntimeHost) -> None:
        super().__init__()
        application = QApplication.instance()
        if application is not None:
            application.setFont(QFont("Microsoft YaHei UI", 10))
        self.runtime = runtime
        self.home_view = HomeView(runtime, self)
        self.library_view = RecordingLibraryView(runtime, self)
        self._shutdown_complete = False

        self.setWindowTitle("EgoGlass Recording Client")
        self.resize(1440, 900)
        self.setMinimumSize(1180, 760)
        self.setMicaEffectEnabled(False)
        self.navigationInterface.setAcrylicEnabled(True)
        self.navigationInterface.setExpandWidth(200)
        self.addSubInterface(self.home_view, FluentIcon.CAMERA, "\u5b9e\u65f6\u5f55\u5236")
        self.addSubInterface(self.library_view, FluentIcon.LIBRARY, "\u5f55\u5236\u5e93")

    def shutdown(self) -> None:
        if self._shutdown_complete:
            return
        self._shutdown_complete = True
        errors: list[Exception] = []
        for operation in (
            self.home_view.close_resources,
            self.library_view.close_resources,
            self.runtime.stop,
        ):
            try:
                operation()
            except Exception as error:
                errors.append(error)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("recording client shutdown failed", errors)

    def closeEvent(self, event: object) -> None:
        try:
            self.shutdown()
        except Exception:
            LOGGER.exception("recording client shutdown failed")
        super().closeEvent(event)


class NativeApplication:
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
    parser = argparse.ArgumentParser(description="Run the EgoGlass recording client")
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


def load_runtime_config(path: Path = Path("config/client-runtime.yaml")) -> RuntimeConfig:
    resolved = path.expanduser().resolve(strict=True)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("client runtime config must be a YAML mapping")
    unknown = set(payload) - _CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown client runtime config keys: {sorted(unknown)}")
    if payload.get("schema_version") != "1.0":
        raise ValueError("unsupported client runtime config schema")
    recordings_root = Path(str(payload.get("recordings_root", "../local-data/recordings")))
    if not recordings_root.is_absolute():
        recordings_root = (resolved.parent / recordings_root).resolve()
    return RuntimeConfig(
        host=str(payload.get("host", "0.0.0.0")),
        port=int(payload.get("port", 8770)),
        discovery_port=int(payload.get("discovery_port", 8771)),
        recordings_root=recordings_root,
        enable_discovery=bool(payload.get("enable_discovery", True)),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke_test:
        return 0
    configure_logging()
    configured = load_runtime_config()
    runtime = UnifiedRuntimeHost(
        RuntimeConfig(
            host=configured.host if args.host is None else args.host,
            port=configured.port if args.port is None else args.port,
            discovery_port=(
                configured.discovery_port
                if args.discovery_port is None
                else args.discovery_port
            ),
            recordings_root=(
                configured.recordings_root
                if args.recordings_root is None
                else args.recordings_root
            ),
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
