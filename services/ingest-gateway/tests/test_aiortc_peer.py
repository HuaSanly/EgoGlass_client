from __future__ import annotations

import asyncio
from collections.abc import Callable

from aiortc import RTCBundlePolicy

from egoglass_ingest_gateway.adapters.aiortc_peer import (
    STREAM_CONTROL_CHANNEL_LABEL,
    AiortcPeer,
    h264_video_codecs,
    lan_rtc_configuration,
    negotiated_video_codec_from_sdp,
)
from egoglass_ingest_gateway.adapters.webrtc import WebRtcPeerCallbacks


class FakeDataChannel:
    def __init__(
        self,
        *,
        label: str = STREAM_CONTROL_CHANNEL_LABEL,
        ordered: bool = True,
        max_retransmits: int | None = None,
    ) -> None:
        self.label = label
        self.ordered = ordered
        self.maxRetransmits = max_retransmits
        self.maxPacketLifeTime = None
        self.readyState = "open"
        self.sent: list[str] = []
        self._handlers: dict[str, Callable[..., None]] = {}

    def on(self, event: str) -> Callable[[Callable[..., None]], Callable[..., None]]:
        def register(handler: Callable[..., None]) -> Callable[..., None]:
            self._handlers[event] = handler
            return handler

        return register

    def emit(self, event: str, *args: object) -> None:
        self._handlers[event](*args)

    def send(self, message: str) -> None:
        self.sent.append(message)


def test_lan_configuration_disables_default_public_stun_and_maximizes_bundling() -> None:
    configuration = lan_rtc_configuration()

    assert configuration.iceServers == []
    assert configuration.bundlePolicy is RTCBundlePolicy.MAX_BUNDLE


def test_negotiated_video_codec_is_parsed_from_structured_sdp() -> None:
    sdp = "\r\n".join(
        (
            "v=0",
            "o=- 1 1 IN IP4 127.0.0.1",
            "s=-",
            "t=0 0",
            "m=video 9 UDP/TLS/RTP/SAVPF 102",
            "c=IN IP4 0.0.0.0",
            "a=rtpmap:102 H264/90000",
            "",
        )
    )

    assert negotiated_video_codec_from_sdp(sdp) == "H264"


def test_viewer_forwarding_uses_only_h264_video_codecs() -> None:
    codecs = h264_video_codecs()

    assert codecs
    assert {codec.mimeType.casefold() for codec in codecs} == {"video/h264"}


def test_remote_reliable_stream_control_channel_is_bidirectional() -> None:
    async def scenario() -> None:
        ready_channels: list[object] = []
        closed_channels: list[object] = []
        statuses: list[str | bytes] = []

        async def ignore(*_args: object) -> None:
            return None

        callbacks = WebRtcPeerCallbacks(
            on_connection_state=ignore,
            on_video_source=ignore,
            on_video_frame=ignore,
            on_metadata=ignore,
            on_control_channel_ready=lambda channel: append_async(ready_channels, channel),
            on_control_channel_closed=lambda channel: append_async(closed_channels, channel),
            on_control_status=lambda _channel, payload: append_async(statuses, payload),
        )
        peer = AiortcPeer(callbacks)
        channel = FakeDataChannel()
        try:
            peer._peer.emit("datachannel", channel)
            await asyncio.sleep(0)
            assert len(ready_channels) == 1
            ready_channels[0].send("command")  # type: ignore[attr-defined]
            assert channel.sent == ["command"]

            channel.emit("message", '{"state":"streaming"}')
            await asyncio.sleep(0)
            assert statuses == ['{"state":"streaming"}']

            channel.readyState = "closed"
            channel.emit("close")
            await asyncio.sleep(0)
            assert closed_channels == ready_channels
        finally:
            await peer.close()

    asyncio.run(scenario())


def test_unreliable_stream_control_channel_is_rejected() -> None:
    async def scenario() -> None:
        ready_channels: list[object] = []

        async def ignore(*_args: object) -> None:
            return None

        callbacks = WebRtcPeerCallbacks(
            on_connection_state=ignore,
            on_video_source=ignore,
            on_video_frame=ignore,
            on_metadata=ignore,
            on_control_channel_ready=lambda channel: append_async(ready_channels, channel),
            on_control_channel_closed=ignore,
            on_control_status=ignore,
        )
        peer = AiortcPeer(callbacks)
        try:
            peer._peer.emit(
                "datachannel",
                FakeDataChannel(ordered=False, max_retransmits=0),
            )
            await asyncio.sleep(0)
            assert ready_channels == []
        finally:
            await peer.close()

    asyncio.run(scenario())


async def append_async(items: list[object], item: object) -> None:
    items.append(item)
