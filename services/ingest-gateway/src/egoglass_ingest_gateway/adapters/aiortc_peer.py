from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from fractions import Fraction

from aiortc import (
    RTCBundlePolicy,
    RTCConfiguration,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import MediaStreamError
from aiortc.sdp import SessionDescription

from ..webrtc_models import WebRtcOffer
from .preview_encoder import MjpegPreviewEncoder
from .webrtc import DecodedVideoFrame, WebRtcPeerCallbacks


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


class AiortcPeer:
    """One aiortc peer that receives a video track and metadata DataChannel."""

    def __init__(self, callbacks: WebRtcPeerCallbacks) -> None:
        self._callbacks = callbacks
        self._peer = RTCPeerConnection(configuration=lan_rtc_configuration())
        self._tasks: set[asyncio.Task[None]] = set()
        self._negotiated_video_codec: str | None = None

        @self._peer.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            await self._callbacks.on_connection_state(self._peer.connectionState)

        @self._peer.on("datachannel")
        def on_datachannel(channel: object) -> None:
            if getattr(channel, "label", None) != "frame-metadata-v1":
                return

            @channel.on("message")
            def on_message(message: str | bytes) -> None:
                self._schedule(self._callbacks.on_metadata(message))

        @self._peer.on("track")
        def on_track(track: object) -> None:
            if getattr(track, "kind", None) == "video":
                self._schedule(self._consume_video(track))

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

    async def _consume_video(self, track: object) -> None:
        preview_encoder = MjpegPreviewEncoder()
        frame_index = 0
        try:
            while True:
                frame = await track.recv()
                time_base = getattr(frame, "time_base", None)
                preview_jpeg = preview_encoder.encode(frame) if frame_index % 2 == 0 else None
                frame_index += 1
                await self._callbacks.on_video_frame(
                    DecodedVideoFrame(
                        width=int(frame.width),
                        height=int(frame.height),
                        pts=getattr(frame, "pts", None),
                        time_base=Fraction(time_base) if time_base is not None else None,
                        preview_jpeg=preview_jpeg,
                    )
                )
        except (MediaStreamError, asyncio.CancelledError):
            return

    def _schedule(self, awaitable: Awaitable[None]) -> None:
        task = asyncio.create_task(awaitable)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
