from __future__ import annotations

import asyncio
import inspect
import json
from fractions import Fraction

from aiortc import RTCPeerConnection, VideoStreamTrack

from egoglass_ingest_gateway.adapters.aiortc_peer import (
    AiortcViewerPeer,
    lan_rtc_configuration,
    negotiated_video_codec_from_sdp,
)
from egoglass_ingest_gateway.adapters.webrtc import (
    DecodedVideoFrame,
    WebRtcControlChannel,
    WebRtcPeerCallbacks,
)
from egoglass_ingest_gateway.webrtc_models import (
    StreamControlAction,
    StreamControlCommand,
    StreamControlState,
    WebRtcOffer,
    WebRtcPhase,
    WebRtcViewerOffer,
)
from egoglass_ingest_gateway.webrtc_runtime import WebRtcSessionRuntime


def test_gateway_disables_per_frame_access_logs() -> None:
    from egoglass_ingest_gateway.app import main

    assert "access_log=False" in inspect.getsource(main)


def test_viewer_peer_negotiates_h264_with_a_browser_offer() -> None:
    async def scenario() -> None:
        browser = RTCPeerConnection(configuration=lan_rtc_configuration())
        viewer = AiortcViewerPeer(VideoStreamTrack())
        browser.addTransceiver("video", direction="recvonly")
        try:
            offer = await browser.createOffer()
            await browser.setLocalDescription(offer)
            local_offer = browser.localDescription
            assert local_offer is not None
            answer_sdp = await viewer.accept_offer(WebRtcViewerOffer(sdp=local_offer.sdp))
            assert negotiated_video_codec_from_sdp(answer_sdp) == "H264"
            await browser.setRemoteDescription(
                type(local_offer)(sdp=answer_sdp, type="answer")
            )
        finally:
            await viewer.close()
            await browser.close()

    asyncio.run(scenario())


def test_reordered_metadata_and_authenticated_replacement_stay_bounded() -> None:
    assert lan_rtc_configuration().iceServers == []
    peers: list[Peer] = []
    viewer_tracks: list[object] = []

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

    class VideoSource:
        def subscribe(self, *, buffered: bool) -> object:
            assert not buffered
            track = object()
            viewer_tracks.append(track)
            return track

    class ViewerPeer:
        def __init__(self, track: object) -> None:
            assert track is viewer_tracks[-1]

        async def accept_offer(self, _offer: WebRtcViewerOffer) -> str:
            return "v=0\r\nviewer-answer-description"

        async def close(self) -> None:
            return None

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
                "capture_config_id": "720p30",
            }
        )

    async def scenario() -> None:
        token = "eval-pairing-token-123456"
        runtime = WebRtcSessionRuntime(token, Peer, ViewerPeer, max_pending_metadata=2)
        offer = WebRtcOffer(
            device_session_id="device-session-eval01",
            sdp="v=0\r\noffer-session-description",
        )
        await runtime.accept_offer(offer, token)
        await peers[0].callbacks.on_connection_state("connected")
        await peers[0].callbacks.on_video_source(VideoSource())
        viewer_answer = await runtime.accept_viewer_offer(
            WebRtcViewerOffer(sdp="v=0\r\nviewer-offer-description")
        )
        assert viewer_answer.type == "answer"
        assert len(viewer_tracks) == 1

        await peers[0].callbacks.on_video_frame(
            DecodedVideoFrame(1280, 720, 0, Fraction(1, 90_000))
        )
        await peers[0].callbacks.on_metadata(payload(1, 90_000, 1000))
        await peers[0].callbacks.on_metadata(payload(2, 180_000, 1050))
        await peers[0].callbacks.on_metadata(payload(3, 270_000, 900))
        await peers[0].callbacks.on_metadata(payload(4, 360_000, 950))
        await peers[0].callbacks.on_metadata("malformed")
        streaming = await runtime.status()
        assert streaming.phase is WebRtcPhase.STREAMING
        assert streaming.metadata_matched == 1
        assert streaming.video_codec == "H264"
        assert streaming.unmatched_entries_dropped == 1
        assert streaming.sdk_clock_discontinuities == 1
        assert streaming.malformed_metadata == 1

        replacement_offer = WebRtcOffer(
            device_session_id="device-session-eval02",
            sdp="v=0\r\noffer-session-description",
        )
        replacement = await runtime.accept_offer(replacement_offer, token)
        await peers[0].callbacks.on_connection_state("failed")
        assert peers[0].closed
        replacement_status = await runtime.status()
        assert replacement.type == "answer"
        assert replacement_status.phase is WebRtcPhase.NEGOTIATING
        assert replacement_status.device_session_id == "device-session-eval02"
        assert replacement_status.connection_state is None

    asyncio.run(scenario())


def test_stream_control_round_trip_reuses_the_connected_peer() -> None:
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

    class ControlChannel(WebRtcControlChannel):
        def __init__(self) -> None:
            self.sent: list[str] = []

        @property
        def is_open(self) -> bool:
            return True

        def send(self, message: str) -> None:
            self.sent.append(message)

    async def acknowledge(
        channel: ControlChannel,
        command: StreamControlCommand,
        state: StreamControlState,
    ) -> None:
        pending = asyncio.create_task(runtime.send_control_command(command))
        await asyncio.sleep(0)
        assert json.loads(channel.sent[-1]) == command.model_dump(mode="json")
        await peers[0].callbacks.on_control_status(
            channel,
            json.dumps(
                {
                    "schema_version": "1.0",
                    "message_type": "stream_control_status",
                    "command_id": command.command_id,
                    "state": state,
                    "detail": None,
                }
            ),
        )
        assert (await pending).state is state

    async def scenario() -> None:
        token = "eval-pairing-token-123456"
        nonlocal runtime
        runtime = WebRtcSessionRuntime(token, Peer)
        await runtime.accept_offer(
            WebRtcOffer(
                device_session_id="device-session-control-eval",
                sdp="v=0\r\noffer-session-description",
            ),
            token,
        )
        channel = ControlChannel()
        await peers[0].callbacks.on_control_channel_ready(channel)

        await acknowledge(
            channel,
            StreamControlCommand(command_id="1" * 32, action=StreamControlAction.STOP),
            StreamControlState.STOPPED,
        )
        await acknowledge(
            channel,
            StreamControlCommand(command_id="2" * 32, action=StreamControlAction.START),
            StreamControlState.STREAMING,
        )

        assert not peers[0].closed
        assert channel.is_open
        assert (await runtime.control_status()).state is StreamControlState.STREAMING

    runtime: WebRtcSessionRuntime
    asyncio.run(scenario())
