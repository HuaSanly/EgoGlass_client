from __future__ import annotations

import asyncio
from collections.abc import Callable
from types import SimpleNamespace

from aiortc import RTCBundlePolicy
from aiortc.mediastreams import MediaStreamError

from ingest_gateway.adapters.aiortc_peer import (
    FRAME_METADATA_CHANNEL_LABEL,
    IMU_TELEMETRY_CHANNEL_LABEL,
    STREAM_CONTROL_CHANNEL_LABEL,
    AiortcPeer,
    AiortcVideoSource,
    lan_rtc_configuration,
    negotiated_video_codec_from_sdp,
)
from ingest_gateway.adapters.webrtc import WebRtcPeerCallbacks


class FakeDataChannel:
    def __init__(
        self,
        *,
        label: str = STREAM_CONTROL_CHANNEL_LABEL,
        ordered: bool = True,
        max_retransmits: int | None = None,
        max_packet_lifetime: int | None = None,
    ) -> None:
        self.label = label
        self.ordered = ordered
        self.maxRetransmits = max_retransmits
        self.maxPacketLifeTime = max_packet_lifetime
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


def test_live_subscription_is_unbuffered_to_prevent_latency_backlog() -> None:
    source = object.__new__(AiortcVideoSource)

    class Relay:
        def __init__(self) -> None:
            self.buffered: bool | None = None

        def subscribe(self, _track: object, *, buffered: bool) -> object:
            self.buffered = buffered
            return object()

    relay = Relay()
    source._track = object()
    source._relay = relay

    source.subscribe(buffered=False)

    assert relay.buffered is False


def test_receiver_stats_report_packet_loss_jitter_and_corrupt_frame_drops() -> None:
    async def scenario() -> None:
        delivered: list[object] = []

        async def ignore(*_args: object) -> None:
            return None

        callbacks = WebRtcPeerCallbacks(
            on_connection_state=ignore,
            on_video_source=ignore,
            on_video_frame=lambda frame: append_async(delivered, frame),
            on_metadata=ignore,
            on_control_channel_ready=ignore,
            on_control_channel_closed=ignore,
            on_control_status=ignore,
            on_imu_channel_ready=ignore,
            on_imu_channel_closed=ignore,
            on_imu_telemetry=ignore,
        )
        peer = AiortcPeer(callbacks)
        original_peer = peer._peer

        class StatsPeer:
            async def getStats(self) -> dict[str, object]:
                return {
                    "video": SimpleNamespace(
                        type="inbound-rtp",
                        kind="video",
                        packetsReceived=980,
                        packetsLost=20,
                        jitter=405,
                    )
                }

            async def close(self) -> None:
                return None

        class Track:
            def __init__(self) -> None:
                self.frames = [
                    SimpleNamespace(is_corrupt=True),
                    SimpleNamespace(
                        is_corrupt=False,
                        width=8,
                        height=6,
                        pts=90_000,
                        time_base=None,
                    ),
                ]

            async def recv(self) -> object:
                if self.frames:
                    return self.frames.pop(0)
                raise MediaStreamError

        await original_peer.close()
        peer._peer = StatsPeer()  # type: ignore[assignment]
        await peer._consume_video(Track())
        stats = await peer.receiver_stats()
        await peer.close()

        assert len(delivered) == 1
        assert stats.packets_received == 980
        assert stats.packets_lost == 20
        assert stats.jitter_ms == 4.5
        assert stats.corrupt_frames_dropped == 1

    asyncio.run(scenario())


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
            on_imu_channel_ready=ignore,
            on_imu_channel_closed=ignore,
            on_imu_telemetry=ignore,
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
            on_imu_channel_ready=ignore,
            on_imu_channel_closed=ignore,
            on_imu_telemetry=ignore,
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


def test_unordered_reliable_metadata_channel_forwards_messages() -> None:
    async def scenario() -> None:
        payloads: list[str | bytes] = []

        async def ignore(*_args: object) -> None:
            return None

        callbacks = WebRtcPeerCallbacks(
            on_connection_state=ignore,
            on_video_source=ignore,
            on_video_frame=ignore,
            on_metadata=lambda payload: append_async(payloads, payload),
            on_control_channel_ready=ignore,
            on_control_channel_closed=ignore,
            on_control_status=ignore,
            on_imu_channel_ready=ignore,
            on_imu_channel_closed=ignore,
            on_imu_telemetry=ignore,
        )
        peer = AiortcPeer(callbacks)
        channel = FakeDataChannel(
            label=FRAME_METADATA_CHANNEL_LABEL,
            ordered=False,
        )
        try:
            peer._peer.emit("datachannel", channel)
            channel.emit("message", b'{"message_type":"video_frame"}')
            await asyncio.sleep(0)
            assert payloads == [b'{"message_type":"video_frame"}']
        finally:
            await peer.close()

    asyncio.run(scenario())


def test_metadata_channel_rejects_ordered_or_partially_reliable_policy() -> None:
    async def scenario() -> None:
        payloads: list[str | bytes] = []

        async def ignore(*_args: object) -> None:
            return None

        callbacks = WebRtcPeerCallbacks(
            on_connection_state=ignore,
            on_video_source=ignore,
            on_video_frame=ignore,
            on_metadata=lambda payload: append_async(payloads, payload),
            on_control_channel_ready=ignore,
            on_control_channel_closed=ignore,
            on_control_status=ignore,
            on_imu_channel_ready=ignore,
            on_imu_channel_closed=ignore,
            on_imu_telemetry=ignore,
        )
        peer = AiortcPeer(callbacks)
        invalid_channels = (
            FakeDataChannel(label=FRAME_METADATA_CHANNEL_LABEL, ordered=True),
            FakeDataChannel(
                label=FRAME_METADATA_CHANNEL_LABEL,
                ordered=False,
                max_retransmits=0,
            ),
            FakeDataChannel(
                label=FRAME_METADATA_CHANNEL_LABEL,
                ordered=False,
                max_packet_lifetime=10,
            ),
        )
        try:
            for channel in invalid_channels:
                peer._peer.emit("datachannel", channel)
                if "message" in channel._handlers:
                    channel.emit("message", "invalid")
            await asyncio.sleep(0)
            assert payloads == []
        finally:
            await peer.close()

    asyncio.run(scenario())


def test_unordered_zero_retransmit_imu_channel_forwards_messages() -> None:
    async def scenario() -> None:
        ready_channels: list[object] = []
        closed_channels: list[object] = []
        payloads: list[str | bytes] = []

        async def ignore(*_args: object) -> None:
            return None

        callbacks = WebRtcPeerCallbacks(
            on_connection_state=ignore,
            on_video_source=ignore,
            on_video_frame=ignore,
            on_metadata=ignore,
            on_control_channel_ready=ignore,
            on_control_channel_closed=ignore,
            on_control_status=ignore,
            on_imu_channel_ready=lambda channel: append_async(ready_channels, channel),
            on_imu_channel_closed=lambda channel: append_async(closed_channels, channel),
            on_imu_telemetry=lambda _channel, payload: append_async(payloads, payload),
        )
        peer = AiortcPeer(callbacks)
        channel = FakeDataChannel(
            label=IMU_TELEMETRY_CHANNEL_LABEL,
            ordered=False,
            max_retransmits=0,
        )
        try:
            peer._peer.emit("datachannel", channel)
            await asyncio.sleep(0)
            assert len(ready_channels) == 1

            channel.emit("message", b'{"message_type":"imu_sample"}')
            await asyncio.sleep(0)
            assert payloads == [b'{"message_type":"imu_sample"}']

            channel.readyState = "closed"
            channel.emit("close")
            await asyncio.sleep(0)
            assert closed_channels == ready_channels
        finally:
            await peer.close()

    asyncio.run(scenario())


def test_imu_channel_rejects_wrong_reliability_policy() -> None:
    async def scenario() -> None:
        ready_channels: list[object] = []

        async def ignore(*_args: object) -> None:
            return None

        callbacks = WebRtcPeerCallbacks(
            on_connection_state=ignore,
            on_video_source=ignore,
            on_video_frame=ignore,
            on_metadata=ignore,
            on_control_channel_ready=ignore,
            on_control_channel_closed=ignore,
            on_control_status=ignore,
            on_imu_channel_ready=lambda channel: append_async(ready_channels, channel),
            on_imu_channel_closed=ignore,
            on_imu_telemetry=ignore,
        )
        peer = AiortcPeer(callbacks)
        invalid_channels = (
            FakeDataChannel(
                label=IMU_TELEMETRY_CHANNEL_LABEL,
                ordered=True,
                max_retransmits=0,
            ),
            FakeDataChannel(
                label=IMU_TELEMETRY_CHANNEL_LABEL,
                ordered=False,
                max_retransmits=None,
            ),
            FakeDataChannel(
                label=IMU_TELEMETRY_CHANNEL_LABEL,
                ordered=False,
                max_retransmits=0,
                max_packet_lifetime=10,
            ),
        )
        try:
            for channel in invalid_channels:
                peer._peer.emit("datachannel", channel)
            await asyncio.sleep(0)
            assert ready_channels == []
        finally:
            await peer.close()

    asyncio.run(scenario())


async def append_async(items: list[object], item: object) -> None:
    items.append(item)
