from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

from ..webrtc_models import WebRtcOffer


@dataclass(frozen=True)
class DecodedVideoFrame:
    width: int
    height: int
    pts: int | None
    time_base: Fraction | None
    preview_jpeg: bytes | None = None


@dataclass(frozen=True)
class WebRtcPeerCallbacks:
    on_connection_state: Callable[[str], Awaitable[None]]
    on_video_frame: Callable[[DecodedVideoFrame], Awaitable[None]]
    on_metadata: Callable[[str | bytes], Awaitable[None]]


class WebRtcPeer(Protocol):
    @property
    def negotiated_video_codec(self) -> str | None: ...

    async def accept_offer(self, offer: WebRtcOffer) -> str: ...

    async def close(self) -> None: ...


def create_aiortc_peer(callbacks: WebRtcPeerCallbacks) -> WebRtcPeer:
    from .aiortc_peer import AiortcPeer

    return AiortcPeer(callbacks)
