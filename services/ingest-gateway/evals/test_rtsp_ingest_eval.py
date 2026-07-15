from __future__ import annotations

import asyncio

import pytest

from egoglass_ingest_gateway.adapters.rtsp import RtspProbeError
from egoglass_ingest_gateway.models import (
    IngestPhase,
    ProbeResult,
    RtspSourceConfig,
    RtspTransport,
)
from egoglass_ingest_gateway.runtime import IngestRuntime


def test_probe_recovers_after_transient_disconnect_without_leaking_credentials() -> None:
    expected = ProbeResult(
        redacted_url=(
            "rtsp://media.example.test:5540/rtp/"
            "34020000001550000668_34020000001550000668"
        ),
        transport=RtspTransport.TCP,
        codec="h264",
        width=1920,
        height=1080,
        pixel_format="yuv420p",
        average_fps=18,
        first_frame_pts=90_000,
        first_frame_time_base_num=1,
        first_frame_time_base_den=90_000,
        opened_at_unix_ns=1,
        first_frame_received_at_perf_counter_ns=1,
        probe_latency_ms=180,
    )

    class RecoveringDecoder:
        def __init__(self) -> None:
            self.attempt = 0

        def probe(self, config: RtspSourceConfig) -> ProbeResult:
            self.attempt += 1
            if self.attempt == 1:
                raise RtspProbeError(
                    f"RTSP probe failed for {config.redacted_url}: ConnectionResetError"
                )
            return expected

    async def scenario() -> None:
        runtime = IngestRuntime(RecoveringDecoder())
        config = RtspSourceConfig(
            host="media.example.test",
            port=5540,
            device_id="34020000001550000668",
            username="operator",
            password="top-secret",
        )

        with pytest.raises(RtspProbeError):
            await runtime.probe(config)
        failed = await runtime.status()
        recovered = await runtime.probe(config)
        ready = await runtime.status()

        assert failed.phase is IngestPhase.FAILED
        assert failed.last_error is not None
        assert "top-secret" not in failed.last_error
        assert "operator@" not in failed.last_error
        assert recovered == expected
        assert ready.phase is IngestPhase.READY
        assert ready.last_error is None

    asyncio.run(scenario())
