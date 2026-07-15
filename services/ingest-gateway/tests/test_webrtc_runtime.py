from __future__ import annotations

import asyncio
import json
from fractions import Fraction

import pytest

from egoglass_ingest_gateway.adapters.webrtc import DecodedVideoFrame, WebRtcPeerCallbacks
from egoglass_ingest_gateway.webrtc_models import WebRtcOffer, WebRtcPhase
from egoglass_ingest_gateway.webrtc_runtime import (
    PairingTokenError,
    WebRtcSessionBusyError,
    WebRtcSessionRuntime,
)

TOKEN = "test-pairing-token-123456"


class FakePeer:
    def __init__(self, callbacks: WebRtcPeerCallbacks) -> None:
        self.callbacks = callbacks
        self.closed = False

    @property
    def negotiated_video_codec(self) -> str:
        return "H264"

    async def accept_offer(self, offer: WebRtcOffer) -> str:
        assert offer.type == "offer"
        return "v=0\r\nanswer-session-description"

    async def close(self) -> None:
        self.closed = True


def offer(device_session_id: str = "device-session-0001") -> WebRtcOffer:
    return WebRtcOffer(
        device_session_id=device_session_id,
        sdp="v=0\r\noffer-session-description",
    )


def metadata_json(frame_id: int = 1, rtp_timestamp: int = 90_000) -> str:
    return json.dumps(
        {
            "schema_version": "1.0",
            "message_type": "video_frame",
            "stream_id": "camera",
            "frame_id": frame_id,
            "captured_at_rokid_sdk_ms": 1000,
            "received_at_elapsed_realtime_ns": 2_000_000_000,
            "video_at_monotonic_ns": 2_000_000_000,
            "rtp_timestamp_90khz": rtp_timestamp,
            "width": 1280,
            "height": 720,
            "rotation_degrees": 0,
            "capture_config_id": "720p20",
        }
    )


def test_authenticated_session_receives_and_matches_video_metadata() -> None:
    peers: list[FakePeer] = []
    perf_values = iter([1_000_000_000, 1_120_000_000])

    def factory(callbacks: WebRtcPeerCallbacks) -> FakePeer:
        peer = FakePeer(callbacks)
        peers.append(peer)
        return peer

    async def scenario() -> None:
        runtime = WebRtcSessionRuntime(
            TOKEN,
            factory,
            perf_clock=lambda: next(perf_values),
        )
        answer = await runtime.accept_offer(offer(), TOKEN)
        assert answer.type == "answer"
        await peers[0].callbacks.on_connection_state("connected")
        await peers[0].callbacks.on_metadata(metadata_json())
        await peers[0].callbacks.on_video_frame(
            DecodedVideoFrame(
                1280,
                720,
                0,
                Fraction(1, 90_000),
                preview_jpeg=b"\xff\xd8preview\xff\xd9",
            )
        )
        status = await runtime.status()
        preview = await runtime.latest_preview_jpeg()

        assert status.phase is WebRtcPhase.STREAMING
        assert status.frames_received == 1
        assert status.metadata_received == 1
        assert status.metadata_matched == 1
        assert status.metadata_rtp_origin_90khz == 90_000
        assert status.video_codec == "H264"
        assert preview == b"\xff\xd8preview\xff\xd9"
        assert status.first_frame_latency_ms == 120.0
        assert (status.width, status.height) == (1280, 720)

    asyncio.run(scenario())


def test_auth_busy_malformed_and_reconnect_paths_are_explicit() -> None:
    peers: list[FakePeer] = []

    def factory(callbacks: WebRtcPeerCallbacks) -> FakePeer:
        peer = FakePeer(callbacks)
        peers.append(peer)
        return peer

    async def scenario() -> None:
        runtime = WebRtcSessionRuntime(TOKEN, factory)
        with pytest.raises(PairingTokenError):
            await runtime.accept_offer(offer(), "wrong-pairing-token")

        await runtime.accept_offer(offer(), TOKEN)
        with pytest.raises(WebRtcSessionBusyError):
            await runtime.accept_offer(offer("device-session-0002"), TOKEN)

        await peers[0].callbacks.on_metadata("not-json")
        await peers[0].callbacks.on_connection_state("disconnected")
        recovered = await runtime.accept_offer(offer(), TOKEN)
        status = await runtime.status()

        assert peers[0].closed
        assert recovered.type == "answer"
        assert status.phase is WebRtcPhase.NEGOTIATING
        assert status.malformed_metadata == 0

    asyncio.run(scenario())
