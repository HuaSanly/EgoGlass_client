from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from fractions import Fraction

from aiortc import (
    RTCBundlePolicy,
    RTCConfiguration,
    RTCPeerConnection,
    RTCRtpCodecCapability,
    RTCRtpSender,
    RTCSessionDescription,
)
from aiortc.contrib.media import MediaRelay
from aiortc.mediastreams import MediaStreamError
from aiortc.sdp import SessionDescription

from ..webrtc_models import WebRtcOffer, WebRtcViewerOffer
from .webrtc import (
    DecodedVideoFrame,
    WebRtcControlChannel,
    WebRtcPeerCallbacks,
    WebRtcVideoSource,
)

FRAME_METADATA_CHANNEL_LABEL = "frame-metadata-v1"
STREAM_CONTROL_CHANNEL_LABEL = "stream-control-v1"


def lan_rtc_configuration() -> RTCConfiguration:
    return RTCConfiguration(
        iceServers=[],
        bundlePolicy=RTCBundlePolicy.MAX_BUNDLE,
    )


def negotiated_video_codec_from_sdp(sdp: str) -> str | None:
    session = SessionDescription.parse(sdp)
    for media in session.media:
        if media.kind != "video":
            continue
        for codec in media.rtp.codecs:
            if codec.mimeType.startswith("video/"):
                return codec.mimeType.removeprefix("video/").upper()
    return None


def h264_video_codecs() -> list[RTCRtpCodecCapability]:
    capabilities = RTCRtpSender.getCapabilities("video")
    return [
        codec
        for codec in capabilities.codecs
        if codec.mimeType.casefold() == "video/h264"
    ]


class AiortcVideoSource(WebRtcVideoSource):
    def __init__(self, track: object) -> None:
        self._track = track
        self._relay = MediaRelay()

    def subscribe(self, *, buffered: bool) -> object:
        return self._relay.subscribe(self._track, buffered=buffered)


class AiortcControlChannel(WebRtcControlChannel):
    def __init__(self, channel: object) -> None:
        self._channel = channel

    @property
    def is_open(self) -> bool:
        return getattr(self._channel, "readyState", None) == "open"

    def send(self, message: str) -> None:
        if not self.is_open:
            raise ConnectionError("stream control channel is not open")
        self._channel.send(message)  # type: ignore[attr-defined]


class AiortcPeer:
    """One aiortc peer that receives a video track and metadata DataChannel."""

    def __init__(self, callbacks: WebRtcPeerCallbacks) -> None:
        self._callbacks = callbacks
        self._peer = RTCPeerConnection(configuration=lan_rtc_configuration())
        self._tasks: set[asyncio.Task[None]] = set()
        self._negotiated_video_codec: str | None = None
        self._video_source: AiortcVideoSource | None = None

        @self._peer.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            await self._callbacks.on_connection_state(self._peer.connectionState)

        @self._peer.on("datachannel")
        def on_datachannel(channel: object) -> None:
            label = getattr(channel, "label", None)
            if label == FRAME_METADATA_CHANNEL_LABEL:
                self._bind_metadata_channel(channel)
                return
            if label == STREAM_CONTROL_CHANNEL_LABEL:
                self._bind_control_channel(channel)

        @self._peer.on("track")
        def on_track(track: object) -> None:
            if getattr(track, "kind", None) == "video":
                source = AiortcVideoSource(track)
                self._video_source = source
                self._schedule(self._callbacks.on_video_source(source))
                self._schedule(self._consume_video(source.subscribe(buffered=True)))

    @property
    def negotiated_video_codec(self) -> str | None:
        return self._negotiated_video_codec

    async def accept_offer(self, offer: WebRtcOffer) -> str:
        await self._peer.setRemoteDescription(
            RTCSessionDescription(sdp=offer.sdp, type=offer.type)
        )
        answer = await self._peer.createAnswer()
        await self._peer.setLocalDescription(answer)
        if self._peer.localDescription is None:
            raise RuntimeError("aiortc produced no local description")
        local_sdp = self._peer.localDescription.sdp
        self._negotiated_video_codec = negotiated_video_codec_from_sdp(local_sdp)
        return local_sdp

    async def close(self) -> None:
        for task in tuple(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._peer.close()
        self._video_source = None

    async def _consume_video(self, track: object) -> None:
        try:
            while True:
                frame = await track.recv()
                time_base = getattr(frame, "time_base", None)
                await self._callbacks.on_video_frame(
                    DecodedVideoFrame(
                        width=int(frame.width),
                        height=int(frame.height),
                        pts=getattr(frame, "pts", None),
                        time_base=Fraction(time_base) if time_base is not None else None,
                    )
                )
        except (MediaStreamError, asyncio.CancelledError):
            return

    def _bind_metadata_channel(self, channel: object) -> None:
        @channel.on("message")  # type: ignore[attr-defined]
        def on_message(message: str | bytes) -> None:
            self._schedule(self._callbacks.on_metadata(message))

    def _bind_control_channel(self, channel: object) -> None:
        if (
            getattr(channel, "ordered", None) is not True
            or getattr(channel, "maxRetransmits", None) is not None
            or getattr(channel, "maxPacketLifeTime", None) is not None
        ):
            return

        control_channel = AiortcControlChannel(channel)

        @channel.on("open")  # type: ignore[attr-defined]
        def on_open() -> None:
            self._schedule(self._callbacks.on_control_channel_ready(control_channel))

        @channel.on("close")  # type: ignore[attr-defined]
        def on_close() -> None:
            self._schedule(self._callbacks.on_control_channel_closed(control_channel))

        @channel.on("message")  # type: ignore[attr-defined]
        def on_message(message: str | bytes) -> None:
            self._schedule(self._callbacks.on_control_status(control_channel, message))

        if control_channel.is_open:
            self._schedule(self._callbacks.on_control_channel_ready(control_channel))

    def _schedule(self, awaitable: Awaitable[None]) -> None:
        task = asyncio.create_task(awaitable)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


class AiortcViewerPeer:
    """One loopback peer that forwards the live Glass3 track to WebView2."""

    def __init__(self, video_track: object) -> None:
        self._peer = RTCPeerConnection(configuration=lan_rtc_configuration())
        transceiver = self._peer.addTransceiver(video_track, direction="sendonly")
        codecs = h264_video_codecs()
        if not codecs:
            raise RuntimeError("aiortc has no H.264 video encoder")
        transceiver.setCodecPreferences(codecs)

    async def accept_offer(self, offer: WebRtcViewerOffer) -> str:
        await self._peer.setRemoteDescription(
            RTCSessionDescription(sdp=offer.sdp, type=offer.type)
        )
        answer = await self._peer.createAnswer()
        await self._peer.setLocalDescription(answer)
        if self._peer.localDescription is None:
            raise RuntimeError("aiortc produced no local viewer description")
        return self._peer.localDescription.sdp

    async def close(self) -> None:
        await self._peer.close()
