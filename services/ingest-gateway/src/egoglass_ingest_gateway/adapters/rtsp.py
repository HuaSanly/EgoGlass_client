from __future__ import annotations

import time
from collections.abc import Callable
from importlib import import_module
from typing import Any, Protocol

from ..models import ProbeResult, RtspSourceConfig


class RtspProbeError(RuntimeError):
    """A safe, user-facing RTSP probe failure."""


class RtspDecoder(Protocol):
    def probe(self, config: RtspSourceConfig) -> ProbeResult: ...


class PyAvRtspDecoder:
    """Decode the first video frame from an RTSP source through PyAV."""

    def __init__(
        self,
        av_module: Any | None = None,
        *,
        unix_clock: Callable[[], int] = time.time_ns,
        perf_clock: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self._av = av_module
        self._unix_clock = unix_clock
        self._perf_clock = perf_clock

    def probe(self, config: RtspSourceConfig) -> ProbeResult:
        av_module = self._av or import_module("av")
        started_perf_ns = self._perf_clock()
        opened_at_unix_ns = self._unix_clock()
        options = {"rtsp_transport": config.transport.value}
        timeout = (
            config.open_timeout_ms / 1000,
            config.read_timeout_ms / 1000,
        )

        try:
            with av_module.open(
                config.rtsp_url,
                mode="r",
                options=options,
                timeout=timeout,
            ) as container:
                stream = next(
                    (candidate for candidate in container.streams if candidate.type == "video"),
                    None,
                )
                if stream is None:
                    raise RtspProbeError("RTSP source has no video stream")

                frame = next(container.decode(video=getattr(stream, "index", 0)), None)
                if frame is None:
                    raise RtspProbeError("RTSP source produced no video frame")

                time_base = getattr(frame, "time_base", None)
                return ProbeResult(
                    redacted_url=config.redacted_url,
                    transport=config.transport,
                    codec=_codec_name(stream),
                    width=int(frame.width),
                    height=int(frame.height),
                    pixel_format=str(frame.format.name),
                    average_fps=_average_fps(stream),
                    first_frame_pts=getattr(frame, "pts", None),
                    first_frame_time_base_num=getattr(time_base, "numerator", None),
                    first_frame_time_base_den=getattr(time_base, "denominator", None),
                    opened_at_unix_ns=opened_at_unix_ns,
                    first_frame_received_at_perf_counter_ns=self._perf_clock(),
                    probe_latency_ms=round((self._perf_clock() - started_perf_ns) / 1_000_000, 3),
                )
        except RtspProbeError:
            raise
        except Exception as error:
            raise RtspProbeError(
                f"RTSP probe failed for {config.redacted_url}: {type(error).__name__}"
            ) from error


def _codec_name(stream: Any) -> str:
    codec_context = getattr(stream, "codec_context", None)
    return str(
        getattr(codec_context, "name", None)
        or getattr(getattr(codec_context, "codec", None), "name", None)
        or "unknown"
    )


def _average_fps(stream: Any) -> float | None:
    value = getattr(stream, "average_rate", None)
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None
