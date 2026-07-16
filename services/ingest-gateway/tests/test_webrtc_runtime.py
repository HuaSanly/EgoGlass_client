from __future__ import annotations

import asyncio
import json
from fractions import Fraction

import pytest

from egoglass_ingest_gateway.adapters.webrtc import (
    DecodedVideoFrame,
    WebRtcPeerCallbacks,
)
from egoglass_ingest_gateway.webrtc_models import (
    WebRtcOffer,
    WebRtcPhase,
    WebRtcViewerOffer,
)
from egoglass_ingest_gateway.webrtc_runtime import (
    PairingTokenError,
    WebRtcSessionRuntime,
    WebRtcViewerUnavailableError,
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


class FakeVideoSource:
    def __init__(self) -> None:
        self.subscriptions: list[tuple[object, bool]] = []

    def subscribe(self, *, buffered: bool) -> object:
        track = object()
        self.subscriptions.append((track, buffered))
        return track


class FakeViewerPeer:
    def __init__(self, track: object) -> None:
        self.track = track
        self.closed = False

    async def accept_offer(self, offer: WebRtcViewerOffer) -> str:
        assert offer.type == "offer"
        return "v=0\r\nviewer-answer-description"

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
            "capture_config_id": "720p30",
        }
    )


def test_authenticated_session_receives_and_matches_video_metadata() -> None:
    peers: list[FakePeer] = []
    viewers: list[FakeViewerPeer] = []
    perf_values = iter([1_000_000_000, 1_120_000_000])

    def factory(callbacks: WebRtcPeerCallbacks) -> FakePeer:
        peer = FakePeer(callbacks)
        peers.append(peer)
        return peer

    def viewer_factory(track: object) -> FakeViewerPeer:
        viewer = FakeViewerPeer(track)
        viewers.append(viewer)
        return viewer

    async def scenario() -> None:
        runtime = WebRtcSessionRuntime(
            TOKEN,
            factory,
            viewer_factory,
            perf_clock=lambda: next(perf_values),
        )
        answer = await runtime.accept_offer(offer(), TOKEN)
        assert answer.type == "answer"
        await peers[0].callbacks.on_connection_state("connected")
        source = FakeVideoSource()
        await peers[0].callbacks.on_video_source(source)
        await peers[0].callbacks.on_metadata(metadata_json())
        await peers[0].callbacks.on_video_frame(
            DecodedVideoFrame(1280, 720, 0, Fraction(1, 90_000))
        )
        viewer_answer = await runtime.accept_viewer_offer(
            WebRtcViewerOffer(sdp="v=0\r\nviewer-offer-description")
        )
        status = await runtime.status()

        assert status.phase is WebRtcPhase.STREAMING
        assert status.frames_received == 1
        assert status.metadata_received == 1
        assert status.metadata_matched == 1
        assert status.metadata_rtp_origin_90khz == 90_000
        assert status.video_codec == "H264"
        assert viewer_answer.type == "answer"
        assert source.subscriptions == [(viewers[0].track, False)]
        assert status.first_frame_latency_ms == 120.0
        assert (status.width, status.height) == (1280, 720)

    asyncio.run(scenario())


def test_authenticated_offer_replaces_active_peer_and_ignores_stale_callbacks() -> None:
    peers: list[FakePeer] = []
    viewers: list[FakeViewerPeer] = []

    def factory(callbacks: WebRtcPeerCallbacks) -> FakePeer:
        peer = FakePeer(callbacks)
        peers.append(peer)
        return peer

    def viewer_factory(track: object) -> FakeViewerPeer:
        viewer = FakeViewerPeer(track)
        viewers.append(viewer)
        return viewer

    async def scenario() -> None:
        runtime = WebRtcSessionRuntime(TOKEN, factory, viewer_factory)
        with pytest.raises(WebRtcViewerUnavailableError):
            await runtime.accept_viewer_offer(
                WebRtcViewerOffer(sdp="v=0\r\nviewer-offer-description")
            )
        with pytest.raises(PairingTokenError):
            await runtime.accept_offer(offer(), "wrong-pairing-token")

        first_answer = await runtime.accept_offer(offer(), TOKEN)
        source = FakeVideoSource()
        await peers[0].callbacks.on_connection_state("connected")
        await peers[0].callbacks.on_video_source(source)
        await runtime.accept_viewer_offer(
            WebRtcViewerOffer(sdp="v=0\r\nviewer-offer-description")
        )
        await peers[0].callbacks.on_metadata("not-json")
        replacement = await runtime.accept_offer(offer("device-session-0002"), TOKEN)
        await peers[0].callbacks.on_connection_state("failed")
        await peers[0].callbacks.on_metadata("not-json")
        status = await runtime.status()

        assert peers[0].closed
        assert viewers[0].closed
        assert replacement.type == "answer"
        assert replacement.session_id != first_answer.session_id
        assert status.phase is WebRtcPhase.NEGOTIATING
        assert status.device_session_id == "device-session-0002"
        assert status.connection_state is None
        assert status.malformed_metadata == 0

    asyncio.run(scenario())
