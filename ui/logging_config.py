from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Callable


class RepeatedMediaErrorFilter(logging.Filter):
    """Keep one representative decoder error per interval instead of flooding stdout."""

    _MEDIA_LOGGERS = frozenset({"aiortc.codecs.h264", "libav.h264"})

    def __init__(
        self,
        interval_seconds: float = 5.0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._interval_seconds = interval_seconds
        self._clock = clock
        self._last_emitted: dict[tuple[str, int, str], float] = {}
        self._suppressed: dict[tuple[str, int, str], int] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name not in self._MEDIA_LOGGERS:
            return True
        message = record.getMessage()
        key = (record.name, record.levelno, message)
        now = self._clock()
        last_emitted = self._last_emitted.get(key)
        if last_emitted is not None and now - last_emitted < self._interval_seconds:
            self._suppressed[key] = self._suppressed.get(key, 0) + 1
            return False
        suppressed = self._suppressed.pop(key, 0)
        if suppressed:
            record.msg = f"{message} (suppressed {suppressed} repeats)"
            record.args = ()
        self._last_emitted[key] = now
        return True


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    for handler in logging.getLogger().handlers:
        handler.addFilter(RepeatedMediaErrorFilter())
    logging.getLogger("aioice").setLevel(logging.WARNING)
    logging.getLogger("OpenGL.acceleratesupport").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message=r"Importing from timm\.models\.layers is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Using torch\.cross without specifying the dim arg is deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"You are using `torch\.load` with `weights_only=False`.*",
        category=FutureWarning,
    )
