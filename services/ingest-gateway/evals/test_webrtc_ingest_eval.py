from __future__ import annotations

import asyncio
import inspect
import json
from fractions import Fraction

from egoglass_ingest_gateway.adapters.aiortc_peer import lan_rtc_configuration
from egoglass_ingest_gateway.adapters.webrtc import DecodedVideoFrame, WebRtcPeerCallbacks
from egoglass_ingest_gateway.webrtc_models import WebRtcOffer, WebRtcPhase
from egoglass_ingest_gateway.webrtc_runtime import WebRtcSessionRuntime


def test_preview_polling_does_not_generate_per_frame_access_logs() -> None:
    from egoglass_ingest_gateway.app import main

    assert "access_log=False" in inspect.getsource(main)


def test_reordered_metadata_disconnect_and_resume_stay_bounded() -> None:
    assert lan_rtc_configuration().iceServers == []
    peers: list[Peer] = []

    class Peer:
        def __init__(self, callbacks: WebRtcPeerCallbacks) -> None:
            self.callbacks = callbacks
            self.closed = False
            peers.append(self)

        @property
        def negotiated_video_codec(self) -> str:
            return "H264"

        async def accept_offer(self, _offer: WebRtcOffer) -> str:
            return "v=0\r\nanswer-session-description"

        async def close(self) -> None:
            self.closed = True

    def payload(frame_id: int, timestamp: int, sdk_timestamp: int) -> str:
        return json.dumps(
            {
                "schema_version": "1.0",
                "message_type": "video_frame",
                "stream_id": "camera",
                "frame_id": frame_id,
                "captured_at_rokid_sdk_ms": sdk_timestamp,
                "received_at_elapsed_realtime_ns": 2_000_000_000 + frame_id,
                "video_at_monotonic_ns": 2_000_000_000 + frame_id,
                "rtp_timestamp_90khz": timestamp,
                "width": 1280,
                "height": 720,
                "rotation_degrees": 0,
                "capture_config_id": "720p20",
            }
        )

    async def scenario() -> None:
        token = "eval-pairing-token-123456"
        runtime = WebRtcSessionRuntime(token, Peer, max_pending_metadata=2)
        offer = WebRtcOffer(
            device_session_id="device-session-eval01",
            sdp="v=0\r\noffer-session-description",
        )
        await runtime.accept_offer(offer, token)
        await peers[0].callbacks.on_connection_state("connected")

        await peers[0].callbacks.on_video_frame(
            DecodedVideoFrame(1280, 720, 0, Fraction(1, 90_000))
        )
        await peers[0].callbacks.on_metadata(payload(1, 90_000, 1000))
        await peers[0].callbacks.on_metadata(payload(2, 180_000, 1050))
        await peers[0].callbacks.on_metadata(payload(3, 270_000, 900))
        await peers[0].callbacks.on_metadata(payload(4, 360_000, 950))
        await peers[0].callbacks.on_metadata("malformed")
        await peers[0].callbacks.on_connection_state("disconnected")

        disconnected = await runtime.status()
        assert disconnected.phase is WebRtcPhase.DISCONNECTED
        assert disconnected.metadata_matched == 1
        assert disconnected.video_codec == "H264"
        assert disconnected.unmatched_entries_dropped == 1
        assert disconnected.sdk_clock_discontinuities == 1
        assert disconnected.malformed_metadata == 1

        await runtime.accept_offer(offer, token)
        assert peers[0].closed
        assert (await runtime.status()).phase is WebRtcPhase.NEGOTIATING

    asyncio.run(scenario())
