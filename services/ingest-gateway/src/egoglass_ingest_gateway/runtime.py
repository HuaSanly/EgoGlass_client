from __future__ import annotations

import asyncio

from .adapters.rtsp import PyAvRtspDecoder, RtspDecoder, RtspProbeError
from .models import IngestPhase, IngestStatus, ProbeResult, RtspSourceConfig


class ProbeBusyError(RuntimeError):
    """Raised instead of queueing another blocking external probe."""


class IngestRuntime:
    """Owns probe state while keeping blocking decoder work off the event loop."""

    def __init__(self, decoder: RtspDecoder | None = None) -> None:
        self._decoder = decoder or PyAvRtspDecoder()
        self._lock = asyncio.Lock()
        self._probe_lock = asyncio.Lock()
        self._status = IngestStatus(phase=IngestPhase.IDLE)

    async def status(self) -> IngestStatus:
        async with self._lock:
            return self._status.model_copy(deep=True)

    async def probe(self, config: RtspSourceConfig) -> ProbeResult:
        if self._probe_lock.locked():
            raise ProbeBusyError("an RTSP probe is already running")

        async with self._probe_lock:
            async with self._lock:
                self._status = IngestStatus(phase=IngestPhase.PROBING)

            try:
                result = await asyncio.to_thread(self._decoder.probe, config)
            except RtspProbeError as error:
                async with self._lock:
                    self._status = IngestStatus(
                        phase=IngestPhase.FAILED,
                        last_error=str(error),
                    )
                raise

            async with self._lock:
                self._status = IngestStatus(
                    phase=IngestPhase.READY,
                    last_probe=result,
                )
            return result
