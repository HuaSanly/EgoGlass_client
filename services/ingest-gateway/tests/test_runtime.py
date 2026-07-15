from __future__ import annotations

import asyncio
import threading

import pytest

from egoglass_ingest_gateway.adapters.rtsp import RtspProbeError
from egoglass_ingest_gateway.models import (
    IngestPhase,
    ProbeResult,
    RtspSourceConfig,
    RtspTransport,
)
from egoglass_ingest_gateway.runtime import IngestRuntime, ProbeBusyError


def probe_result() -> ProbeResult:
    return ProbeResult(
        redacted_url=(
            "rtsp://media.example.test:554/rtp/"
            "34020000001550000668_34020000001550000668"
        ),
        transport=RtspTransport.TCP,
        codec="h264",
        width=1280,
        height=720,
        pixel_format="yuv420p",
        average_fps=20.0,
        first_frame_pts=0,
        first_frame_time_base_num=1,
        first_frame_time_base_den=90_000,
        opened_at_unix_ns=1,
        first_frame_received_at_perf_counter_ns=1,
        probe_latency_ms=45.0,
    )


def source_config() -> RtspSourceConfig:
    return RtspSourceConfig(
        host="media.example.test",
        device_id="34020000001550000668",
    )


def test_successful_probe_updates_ready_status() -> None:
    class Decoder:
        def probe(self, config: RtspSourceConfig) -> ProbeResult:
            return probe_result()

    async def scenario() -> None:
        runtime = IngestRuntime(Decoder())
        result = await runtime.probe(source_config())
        status = await runtime.status()

        assert result.codec == "h264"
        assert status.phase is IngestPhase.READY
        assert status.last_error is None
        assert status.last_probe == result

    asyncio.run(scenario())


def test_failed_probe_updates_failure_status() -> None:
    class Decoder:
        def probe(self, config: RtspSourceConfig) -> ProbeResult:
            raise RtspProbeError(f"RTSP probe failed for {config.redacted_url}: TimeoutError")

    async def scenario() -> None:
        runtime = IngestRuntime(Decoder())
        with pytest.raises(RtspProbeError):
            await runtime.probe(source_config())
        status = await runtime.status()

        assert status.phase is IngestPhase.FAILED
        assert status.last_probe is None
        assert status.last_error is not None
        assert "TimeoutError" in status.last_error

    asyncio.run(scenario())


def test_concurrent_probe_is_rejected_instead_of_queued() -> None:
    started = threading.Event()
    release = threading.Event()

    class Decoder:
        def probe(self, config: RtspSourceConfig) -> ProbeResult:
            started.set()
            assert release.wait(timeout=2)
            return probe_result()

    async def scenario() -> None:
        runtime = IngestRuntime(Decoder())
        first_probe = asyncio.create_task(runtime.probe(source_config()))
        assert await asyncio.to_thread(started.wait, 1)

        with pytest.raises(ProbeBusyError, match="already running"):
            await runtime.probe(source_config())

        release.set()
        assert await first_probe == probe_result()

    asyncio.run(scenario())
