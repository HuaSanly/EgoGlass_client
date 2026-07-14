from __future__ import annotations

import argparse
import ctypes
import http.cookiejar
import importlib
import logging
import os
import secrets
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from ctypes import wintypes
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType
from typing import Any

import uvicorn

from .app import create_app
from .runtime import ConsoleRuntime

APP_NAME = "EgoGlass"
MUTEX_NAME = "Local\\EgoGlassOperatorConsole"
ERROR_ALREADY_EXISTS = 183
DEFAULT_WIDTH = 1440
DEFAULT_HEIGHT = 900
MIN_WIDTH = 1100
MIN_HEIGHT = 700
STARTUP_TIMEOUT_SECONDS = 10.0

logger = logging.getLogger(__name__)


class AlreadyRunningError(RuntimeError):
    """Raised when another EgoGlass desktop process owns the Windows mutex."""


class SingleInstance:
    def __init__(self, name: str = MUTEX_NAME) -> None:
        self._name = name
        self._handle: int | None = None

    def __enter__(self) -> SingleInstance:
        if os.name != "nt":
            return self

        kernel32 = ctypes.windll.kernel32
        kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.CreateMutexW(None, False, self._name)
        if not handle:
            raise OSError("failed to create Windows single-instance mutex")
        if kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            raise AlreadyRunningError("EgoGlass is already running")
        self._handle = handle
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._handle is not None:
            ctypes.windll.kernel32.CloseHandle(self._handle)
            self._handle = None


class LocalConsoleServer:
    """Runs Uvicorn on a reserved loopback socket in a background thread."""

    def __init__(self, desktop_token: str) -> None:
        app = create_app(ConsoleRuntime(), desktop_token=desktop_token)
        self._config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=0,
            log_level="warning",
            log_config=None,
            access_log=False,
        )
        self._socket = self._config.bind_socket()
        self._server = uvicorn.Server(self._config)
        self._thread = threading.Thread(
            target=self._run,
            name="egoglass-local-server",
            daemon=True,
        )
        self.port = int(self._socket.getsockname()[1])

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout_seconds: float = STARTUP_TIMEOUT_SECONDS) -> None:
        try:
            self._thread.start()
            deadline = time.monotonic() + timeout_seconds
            while not self._server.started:
                if not self._thread.is_alive():
                    raise RuntimeError("local operator server stopped during startup")
                if time.monotonic() >= deadline:
                    raise TimeoutError("local operator server did not start in time")
                time.sleep(0.01)
        except BaseException:
            self.stop()
            raise

    def stop(self, timeout_seconds: float = 5.0) -> None:
        if not self._thread.is_alive():
            self._socket.close()
            return
        self._server.should_exit = True
        self._thread.join(timeout_seconds)
        if self._thread.is_alive():
            self._server.force_exit = True
            self._thread.join(1.0)
        self._socket.close()

    def _run(self) -> None:
        self._server.run(sockets=[self._socket])


def default_data_root(environment: dict[str, str] | None = None) -> Path:
    env = environment if environment is not None else os.environ
    local_app_data = env.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / APP_NAME
    return Path.home() / ".egoglass"


def configure_logging(data_root: Path) -> Path:
    log_dir = data_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "desktop.log"
    handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    resolved_log_path = log_path.resolve()
    has_handler = any(
        isinstance(existing, RotatingFileHandler)
        and Path(existing.baseFilename).resolve() == resolved_log_path
        for existing in root_logger.handlers
    )
    if not has_handler:
        root_logger.addHandler(handler)
    else:
        handler.close()
    return log_path


def build_desktop_url(origin: str, desktop_token: str) -> str:
    encoded_token = urllib.parse.quote(desktop_token, safe="")
    return f"{origin}/?desktop_token={encoded_token}"


def require_webview2_runtime(
    import_module: Callable[[str], Any] = importlib.import_module,
    *,
    platform_name: str = os.name,
) -> str:
    if platform_name != "nt":
        raise RuntimeError("the EgoGlass desktop application requires Windows")

    try:
        import_module("webview.platforms.edgechromium")
        webview2_core = import_module("Microsoft.Web.WebView2.Core")
        version = str(
            webview2_core.CoreWebView2Environment.GetAvailableBrowserVersionString()
        )
    except Exception as error:
        raise RuntimeError(
            "Microsoft Edge WebView2 Runtime is required. "
            "Install the Evergreen Runtime and start EgoGlass again."
        ) from error

    if not version:
        raise RuntimeError("Microsoft Edge WebView2 Runtime did not report a version")
    return version


def run_smoke_test(
    *,
    webview2_check: Callable[[], str] = require_webview2_runtime,
) -> int:
    webview2_version = webview2_check()
    logger.info("desktop smoke test found WebView2 %s", webview2_version)
    token = "egoglass-packaged-smoke-test"
    server = LocalConsoleServer(token)
    try:
        server.start()
        cookie_jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
        with opener.open(build_desktop_url(server.origin, token), timeout=5) as response:
            page = response.read().decode("utf-8")
        with opener.open(f"{server.origin}/api/v1/state", timeout=5) as response:
            state_payload = response.read().decode("utf-8")
        if "EgoGlass Operator Console" not in page or '"mode":"simulation"' not in state_payload:
            return 1
        return 0
    finally:
        server.stop()


def launch_desktop(
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    debug: bool = False,
    data_root: Path | None = None,
    webview_module: Any | None = None,
    desktop_token: str | None = None,
    mutex_name: str = MUTEX_NAME,
) -> int:
    resolved_data_root = data_root or default_data_root()
    resolved_data_root.mkdir(parents=True, exist_ok=True)
    configure_logging(resolved_data_root)
    token = desktop_token or secrets.token_urlsafe(32)

    with SingleInstance(mutex_name):
        server = LocalConsoleServer(token)
        try:
            server.start()
            if webview_module is None:
                webview2_version = require_webview2_runtime()
                logger.info("using WebView2 %s", webview2_version)
                import webview as webview_module

            webview_module.create_window(
                APP_NAME,
                build_desktop_url(server.origin, token),
                width=width,
                height=height,
                min_size=(MIN_WIDTH, MIN_HEIGHT),
                resizable=True,
                frameless=False,
                background_color="#101211",
                text_select=False,
            )
            webview_module.start(
                gui="edgechromium",
                debug=debug,
                private_mode=True,
            )
            return 0
        finally:
            server.stop()


def show_windows_message(title: str, message: str, *, error: bool = False) -> None:
    if os.name != "nt":
        return
    icon_flag = 0x10 if error else 0x40
    ctypes.windll.user32.MessageBoxW(None, message, title, icon_flag)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EgoGlass as a Windows desktop app")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke_test:
        return run_smoke_test()

    try:
        return launch_desktop(width=args.width, height=args.height, debug=args.debug)
    except AlreadyRunningError:
        show_windows_message(APP_NAME, "EgoGlass 已经在运行。")
        return 2
    except Exception:
        data_root = default_data_root()
        log_path = data_root / "logs" / "desktop.log"
        try:
            log_path = configure_logging(data_root)
            logger.exception("desktop startup failed")
        except Exception:
            pass
        show_windows_message(
            APP_NAME,
            f"EgoGlass 启动失败。\n\n日志：{log_path}",
            error=True,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
