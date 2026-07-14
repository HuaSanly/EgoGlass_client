from __future__ import annotations

import os
import sys
import urllib.request
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import egoglass_operator_console.desktop as desktop_module
from egoglass_operator_console.desktop import (
    AlreadyRunningError,
    LocalConsoleServer,
    SingleInstance,
    build_desktop_url,
    default_data_root,
    launch_desktop,
    require_webview2_runtime,
    run_smoke_test,
)


class FakeWebview:
    def __init__(self) -> None:
        self.window_args: tuple[Any, ...] | None = None
        self.window_kwargs: dict[str, Any] | None = None
        self.start_kwargs: dict[str, Any] | None = None

    def create_window(self, *args: Any, **kwargs: Any) -> object:
        self.window_args = args
        self.window_kwargs = kwargs
        return object()

    def start(self, **kwargs: Any) -> None:
        self.start_kwargs = kwargs
        assert self.window_args is not None
        with urllib.request.urlopen(self.window_args[1], timeout=5) as response:
            assert b"EgoGlass Operator Console" in response.read()


def test_default_data_root_uses_local_app_data() -> None:
    root = default_data_root({"LOCALAPPDATA": r"C:\Users\test\AppData\Local"})

    assert root == Path(r"C:\Users\test\AppData\Local") / "EgoGlass"


def test_desktop_url_escapes_session_token() -> None:
    url = build_desktop_url("http://127.0.0.1:12345", "token with spaces/+?")

    assert url == "http://127.0.0.1:12345/?desktop_token=token%20with%20spaces%2F%2B%3F"


def test_desktop_launcher_uses_native_window_and_webview2(tmp_path: Path) -> None:
    fake_webview = FakeWebview()
    exit_code = launch_desktop(
        data_root=tmp_path,
        webview_module=fake_webview,
        desktop_token="desktop-test-token",
        mutex_name=f"Local\\EgoGlassTest-{uuid.uuid4()}",
    )

    assert exit_code == 0
    assert fake_webview.window_args is not None
    assert fake_webview.window_args[0] == "EgoGlass"
    assert fake_webview.window_args[1].startswith("http://127.0.0.1:")
    assert fake_webview.window_kwargs is not None
    assert fake_webview.window_kwargs["min_size"] == (1100, 700)
    assert fake_webview.window_kwargs["frameless"] is False
    assert fake_webview.start_kwargs == {
        "gui": "edgechromium",
        "debug": False,
        "private_mode": True,
    }
    assert (tmp_path / "logs" / "desktop.log").is_file()


def test_desktop_server_smoke_test() -> None:
    assert run_smoke_test(webview2_check=lambda: "test-webview2") == 0


def test_webview2_preflight_reports_runtime_version() -> None:
    core_module = SimpleNamespace(
        CoreWebView2Environment=SimpleNamespace(
            GetAvailableBrowserVersionString=lambda: "150.0.4078.65"
        )
    )

    def import_module(name: str) -> object:
        if name == "webview.platforms.edgechromium":
            return object()
        if name == "Microsoft.Web.WebView2.Core":
            return core_module
        raise ImportError(name)

    assert (
        require_webview2_runtime(import_module, platform_name="nt") == "150.0.4078.65"
    )


def test_webview2_preflight_has_actionable_failure() -> None:
    def fail_import(name: str) -> object:
        raise ImportError(name)

    with pytest.raises(RuntimeError, match="WebView2 Runtime is required"):
        require_webview2_runtime(fail_import, platform_name="nt")


def test_desktop_server_does_not_require_console_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as context:
        context.setattr(sys, "stdout", None)
        context.setattr(sys, "stderr", None)
        server = LocalConsoleServer("no-console-streams")

    try:
        server.start()
        with urllib.request.urlopen(f"{server.origin}/api/v1/health", timeout=5) as response:
            assert response.status == 200
    finally:
        server.stop()


def test_main_still_reports_failure_when_logging_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[tuple[str, str, bool]] = []

    def fail_launch(**kwargs: Any) -> int:
        raise RuntimeError("startup failed")

    def fail_logging(data_root: Path) -> Path:
        raise OSError("log directory is unavailable")

    def capture_message(title: str, message: str, *, error: bool = False) -> None:
        messages.append((title, message, error))

    monkeypatch.setattr(desktop_module, "launch_desktop", fail_launch)
    monkeypatch.setattr(desktop_module, "configure_logging", fail_logging)
    monkeypatch.setattr(desktop_module, "show_windows_message", capture_message)

    assert desktop_module.main([]) == 1
    assert messages == [
        (
            "EgoGlass",
            f"EgoGlass 启动失败。\n\n日志：{default_data_root() / 'logs' / 'desktop.log'}",
            True,
        )
    ]


@pytest.mark.skipif(os.name != "nt", reason="Windows named mutex")
def test_single_instance_rejects_second_process_slot() -> None:
    mutex_name = f"Local\\EgoGlassTest-{uuid.uuid4()}"

    with (
        SingleInstance(mutex_name),
        pytest.raises(AlreadyRunningError),
        SingleInstance(mutex_name),
    ):
        pass
