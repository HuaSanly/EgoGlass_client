from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

from ..webrtc_models import WebRtcOffer, WebRtcViewerOffer


@dataclass(frozen=True)
class DecodedVideoFrame:
    width: int
    height: int
    pts: int | None
    time_base: Fraction | None


@dataclass(frozen=True)
class WebRtcVideoRecordingSource:
    session_id: str
    source: WebRtcVideoSource
    width: int
    height: int


class WebRtcVideoSource(Protocol):
    def subscribe(self, *, buffered: bool) -> object: ...


class WebRtcControlChannel(Protocol):
    @property
    def is_open(self) -> bool: ...

    def send(self, message: str) -> None: ...


class WebRtcImuChannel(Protocol):
    @property
    def is_open(self) -> bool: ...


@dataclass(frozen=True)
class WebRtcPeerCallbacks:
    on_connection_state: Callable[[str], Awaitable[None]]
    on_video_source: Callable[[WebRtcVideoSource], Awaitable[None]]
    on_video_frame: Callable[[DecodedVideoFrame], Awaitable[None]]
    on_metadata: Callable[[str | bytes], Awaitable[None]]
    on_control_channel_ready: Callable[[WebRtcControlChannel], Awaitable[None]]
    on_control_channel_closed: Callable[[WebRtcControlChannel], Awaitable[None]]
    on_control_status: Callable[
        [WebRtcControlChannel, str | bytes], Awaitable[None]
    ]
    on_imu_channel_ready: Callable[[WebRtcImuChannel], Awaitable[None]]
    on_imu_channel_closed: Callable[[WebRtcImuChannel], Awaitable[None]]
    on_imu_telemetry: Callable[[WebRtcImuChannel, str | bytes], Awaitable[None]]


class WebRtcPeer(Protocol):
    @property
    def negotiated_video_codec(self) -> str | None: ...

    async def accept_offer(self, offer: WebRtcOffer) -> str: ...

    async def close(self) -> None: ...


class WebRtcViewerPeer(Protocol):
    async def accept_offer(self, offer: WebRtcViewerOffer) -> str: ...

    async def close(self) -> None: ...


def create_aiortc_peer(callbacks: WebRtcPeerCallbacks) -> WebRtcPeer:
    from .aiortc_peer import AiortcPeer

    return AiortcPeer(callbacks)


def create_aiortc_viewer_peer(video_track: object) -> WebRtcViewerPeer:
    from .aiortc_peer import AiortcViewerPeer

    return AiortcViewerPeer(video_track)
