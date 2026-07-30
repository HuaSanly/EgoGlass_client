from __future__ import annotations

import argparse
import logging

from ui import app
from ui.logging_config import RepeatedMediaErrorFilter


def _record(name: str, message: str = "decode failed") -> logging.LogRecord:
    return logging.LogRecord(name, logging.WARNING, __file__, 1, message, (), None)


def test_repeated_media_errors_are_rate_limited_and_counted() -> None:
    timestamps = iter((0.0, 1.0, 6.0))
    error_filter = RepeatedMediaErrorFilter(clock=lambda: next(timestamps))

    assert error_filter.filter(_record("aiortc.codecs.h264"))
    assert not error_filter.filter(_record("aiortc.codecs.h264"))
    resumed = _record("aiortc.codecs.h264")
    assert error_filter.filter(resumed)
    assert resumed.getMessage().endswith("(suppressed 1 repeats)")


def test_non_media_logs_are_never_rate_limited() -> None:
    error_filter = RepeatedMediaErrorFilter(clock=lambda: 0.0)

    assert error_filter.filter(_record("ingest_gateway.webrtc_runtime"))
    assert error_filter.filter(_record("ingest_gateway.webrtc_runtime"))


def test_ctrl_c_exits_native_client_without_propagating_traceback(monkeypatch) -> None:
    args = argparse.Namespace(
        smoke_test=False,
        host="127.0.0.1",
        port=8770,
        discovery_port=8771,
        recordings_root="recordings",
        pairing_token="test-pairing-token-123456",
        disable_discovery=True,
    )

    class InterruptingApplication:
        def __init__(self, _runtime: object) -> None:
            pass

        def run(self) -> int:
            raise KeyboardInterrupt

    monkeypatch.setattr(app, "parse_args", lambda _argv: args)
    monkeypatch.setattr(app, "UnifiedRuntimeHost", lambda _config: object())
    monkeypatch.setattr(app, "NativeApplication", InterruptingApplication)
    monkeypatch.setattr(app, "configure_logging", lambda: None)

    assert app.main([]) == 0
