from fractions import Fraction
from types import SimpleNamespace

import pytest

from egoglass_ingest_gateway.adapters.rtsp import PyAvRtspDecoder, RtspProbeError
from egoglass_ingest_gateway.models import RtspSourceConfig, RtspTransport


class FakeContainer:
    def __init__(self) -> None:
        self.stream = SimpleNamespace(
            type="video",
            index=0,
            codec_context=SimpleNamespace(name="h264"),
            average_rate=Fraction(20, 1),
        )
        self.streams = [self.stream]

    def __enter__(self) -> "FakeContainer":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def decode(self, *, video: int):
        assert video == 0
        yield SimpleNamespace(
            width=1280,
            height=720,
            format=SimpleNamespace(name="yuv420p"),
            pts=1800,
            time_base=Fraction(1, 90_000),
        )


class FakeAv:
    def __init__(self) -> None:
        self.open_args: tuple[object, ...] | None = None
        self.open_kwargs: dict[str, object] | None = None

    def open(self, *args: object, **kwargs: object) -> FakeContainer:
        self.open_args = args
        self.open_kwargs = kwargs
        return FakeContainer()


def test_probe_decodes_first_frame_and_preserves_stream_timestamp() -> None:
    fake_av = FakeAv()
    perf_values = iter([1_000_000_000, 1_123_000_000, 1_123_000_000])
    decoder = PyAvRtspDecoder(
        fake_av,
        unix_clock=lambda: 1_750_000_000_000_000_000,
        perf_clock=lambda: next(perf_values),
    )
    config = RtspSourceConfig(
        host="media.example.test",
        port=5540,
        device_id="34020000001550000668",
        transport=RtspTransport.TCP,
        open_timeout_ms=2500,
        read_timeout_ms=4000,
    )

    result = decoder.probe(config)

    assert fake_av.open_args == (config.rtsp_url,)
    assert fake_av.open_kwargs == {
        "mode": "r",
        "options": {"rtsp_transport": "tcp"},
        "timeout": (2.5, 4.0),
    }
    assert result.codec == "h264"
    assert (result.width, result.height) == (1280, 720)
    assert result.average_fps == 20.0
    assert result.first_frame_pts == 1800
    assert result.first_frame_time_base_num == 1
    assert result.first_frame_time_base_den == 90_000
    assert result.probe_latency_ms == 123.0


def test_probe_failure_never_exposes_credentials() -> None:
    class FailingAv:
        def open(self, url: str, **kwargs: object) -> None:
            raise OSError(f"failed to open {url}")

    config = RtspSourceConfig(
        host="media.example.test",
        device_id="34020000001550000668",
        username="operator",
        password="top-secret",
    )
    decoder = PyAvRtspDecoder(FailingAv())

    with pytest.raises(RtspProbeError) as failure:
        decoder.probe(config)

    assert "top-secret" not in str(failure.value)
    assert "operator@" not in str(failure.value)
    assert config.redacted_url in str(failure.value)
