from __future__ import annotations

import asyncio
import hmac
import json
import time
import uuid
from collections.abc import Callable

from pydantic import ValidationError

from .adapters.webrtc import (
    DecodedVideoFrame,
    WebRtcPeer,
    WebRtcPeerCallbacks,
    WebRtcVideoSource,
    WebRtcViewerPeer,
    create_aiortc_peer,
    create_aiortc_viewer_peer,
)
from .webrtc_matcher import DuplicateMetadataError, FrameMetadataMatcher
from .webrtc_models import (
    VideoFrameMetadata,
    WebRtcAnswer,
    WebRtcOffer,
    WebRtcPhase,
    WebRtcStatus,
    WebRtcViewerAnswer,
    WebRtcViewerOffer,
)


class PairingTokenError(RuntimeError):
    """Raised when signaling authentication fails."""


class WebRtcSessionBusyError(RuntimeError):
    """Raised when a second peer attempts to replace a live stream."""


class WebRtcSessionError(RuntimeError):
    """A safe signaling failure that never contains SDP or credentials."""


class WebRtcViewerUnavailableError(RuntimeError):
    """Raised when no Glass3 video track is available for local preview."""


class WebRtcViewerSessionError(RuntimeError):
    """A safe local viewer signaling failure."""


WebRtcPeerFactory = Callable[[WebRtcPeerCallbacks], WebRtcPeer]
WebRtcViewerPeerFactory = Callable[[object], WebRtcViewerPeer]


class WebRtcSessionRuntime:
    """Own one authenticated WebRTC receive session and bounded stream metrics."""

    def __init__(
        self,
        pairing_token: str,
        peer_factory: WebRtcPeerFactory = create_aiortc_peer,
        viewer_peer_factory: WebRtcViewerPeerFactory = create_aiortc_viewer_peer,
        *,
        perf_clock: Callable[[], int] = time.perf_counter_ns,
        max_pending_metadata: int = 256,
    ) -> None:
        if len(pairing_token) < 16:
            raise ValueError("pairing_token must contain at least 16 characters")
        self._pairing_token = pairing_token
        self._peer_factory = peer_factory
        self._viewer_peer_factory = viewer_peer_factory
        self._perf_clock = perf_clock
        self._max_pending_metadata = max_pending_metadata
        self._session_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._peer: WebRtcPeer | None = None
        self._viewer_peer: WebRtcViewerPeer | None = None
        self._video_source: WebRtcVideoSource | None = None
        self._generation = 0
        self._reset_state()

    async def status(self) -> WebRtcStatus:
        async with self._state_lock:
            duration_ns = None
            if self._first_frame_at_ns is not None and self._last_frame_at_ns is not None:
                duration_ns = self._last_frame_at_ns - self._first_frame_at_ns
            average_fps = None
            if duration_ns and duration_ns > 0 and self._frames_received > 1:
                average_fps = round(
                    (self._frames_received - 1) * 1_000_000_000 / duration_ns,
                    3,
                )
            matcher = self._matcher
            return WebRtcStatus(
                phase=self._phase,
                session_id=self._session_id,
                device_session_id=self._device_session_id,
                connection_state=self._connection_state,
                frames_received=self._frames_received,
                metadata_received=self._metadata_received,
                metadata_matched=matcher.matched,
                malformed_metadata=self._malformed_metadata,
                duplicate_metadata=matcher.duplicates,
                unmatched_entries_dropped=matcher.dropped,
                sdk_clock_discontinuities=matcher.sdk_clock_discontinuities,
                pending_frames=matcher.pending_frames,
                pending_metadata=matcher.pending_metadata,
                max_timestamp_match_error_90khz=matcher.max_timestamp_match_error_90khz,
                width=self._width,
                height=self._height,
                video_codec=self._video_codec,
                average_fps=average_fps,
                first_frame_latency_ms=self._first_frame_latency_ms,
                last_frame_pts=self._last_frame_pts,
                last_frame_time_base_num=self._last_frame_time_base_num,
                last_frame_time_base_den=self._last_frame_time_base_den,
                metadata_rtp_origin_90khz=matcher.metadata_origin_90khz,
                last_frame_received_at_perf_counter_ns=self._last_frame_at_ns,
                last_error=self._last_error,
            )

    async def accept_offer(self, offer: WebRtcOffer, pairing_token: str) -> WebRtcAnswer:
        if not hmac.compare_digest(pairing_token, self._pairing_token):
            raise PairingTokenError("invalid pairing token")

        async with self._session_lock:
            current = await self.status()
            if self._peer is not None and current.phase in {
                WebRtcPhase.NEGOTIATING,
                WebRtcPhase.CONNECTED,
                WebRtcPhase.STREAMING,
            }:
                raise WebRtcSessionBusyError("a WebRTC session is already active")

            if self._peer is not None:
                await self._peer.close()
            if self._viewer_peer is not None:
                await self._viewer_peer.close()
                self._viewer_peer = None
            self._video_source = None

            self._generation += 1
            generation = self._generation
            callbacks = WebRtcPeerCallbacks(
                on_connection_state=lambda state: self._on_connection_state(generation, state),
                on_video_source=lambda source: self._on_video_source(generation, source),
                on_video_frame=lambda frame: self._on_video_frame(generation, frame),
                on_metadata=lambda payload: self._on_metadata(generation, payload),
            )
            peer = self._peer_factory(callbacks)
            self._peer = peer
            session_id = uuid.uuid4().hex
            async with self._state_lock:
                self._reset_state()
                self._phase = WebRtcPhase.NEGOTIATING
                self._session_id = session_id
                self._device_session_id = offer.device_session_id
                self._negotiation_started_at_ns = self._perf_clock()

            try:
                answer_sdp = await peer.accept_offer(offer)
                video_codec = peer.negotiated_video_codec
            except Exception as error:
                async with self._state_lock:
                    self._phase = WebRtcPhase.FAILED
                    self._last_error = f"WebRTC negotiation failed: {type(error).__name__}"
                raise WebRtcSessionError(self._last_error) from error

            async with self._state_lock:
                self._video_codec = video_codec

            return WebRtcAnswer(session_id=session_id, sdp=answer_sdp)

    async def accept_viewer_offer(self, offer: WebRtcViewerOffer) -> WebRtcViewerAnswer:
        async with self._session_lock:
            source = self._video_source
            if source is None:
                raise WebRtcViewerUnavailableError("Glass3 video is not ready")

            if self._viewer_peer is not None:
                await self._viewer_peer.close()
            peer = self._viewer_peer_factory(source.subscribe(buffered=False))
            self._viewer_peer = peer
            try:
                answer_sdp = await peer.accept_offer(offer)
            except Exception as error:
                await peer.close()
                if self._viewer_peer is peer:
                    self._viewer_peer = None
                message = f"viewer negotiation failed: {type(error).__name__}"
                raise WebRtcViewerSessionError(message) from error
            return WebRtcViewerAnswer(sdp=answer_sdp)

    async def close(self) -> None:
        async with self._session_lock:
            self._generation += 1
            peer, self._peer = self._peer, None
            viewer_peer, self._viewer_peer = self._viewer_peer, None
            self._video_source = None
            if viewer_peer is not None:
                await viewer_peer.close()
            if peer is not None:
                await peer.close()
            async with self._state_lock:
                self._phase = WebRtcPhase.IDLE
                self._connection_state = "closed"

    async def _on_connection_state(self, generation: int, state: str) -> None:
        if generation != self._generation:
            return
        async with self._state_lock:
            self._connection_state = state
            if state == "connected" and self._phase is WebRtcPhase.NEGOTIATING:
                self._phase = WebRtcPhase.CONNECTED
            elif state in {"disconnected", "closed"}:
                self._phase = WebRtcPhase.DISCONNECTED
            elif state == "failed":
                self._phase = WebRtcPhase.FAILED
                self._last_error = "WebRTC peer connection failed"

    async def _on_video_source(
        self,
        generation: int,
        source: WebRtcVideoSource,
    ) -> None:
        if generation == self._generation:
            self._video_source = source

    async def _on_video_frame(self, generation: int, frame: DecodedVideoFrame) -> None:
        if generation != self._generation:
            return
        now_ns = self._perf_clock()
        async with self._state_lock:
            self._phase = WebRtcPhase.STREAMING
            self._frames_received += 1
            self._width = frame.width
            self._height = frame.height
            self._last_frame_pts = frame.pts
            self._last_frame_at_ns = now_ns
            if frame.time_base is not None:
                self._last_frame_time_base_num = frame.time_base.numerator
                self._last_frame_time_base_den = frame.time_base.denominator
            if self._first_frame_at_ns is None:
                self._first_frame_at_ns = now_ns
                if self._negotiation_started_at_ns is not None:
                    self._first_frame_latency_ms = round(
                        (now_ns - self._negotiation_started_at_ns) / 1_000_000,
                        3,
                    )
            self._matcher.add_frame(frame.pts)

    async def _on_metadata(self, generation: int, payload: str | bytes) -> None:
        if generation != self._generation:
            return
        try:
            if isinstance(payload, bytes):
                if len(payload) > 16_384:
                    raise ValueError("metadata payload too large")
                raw = payload.decode("utf-8")
            else:
                raw = payload
            if len(raw) > 16_384:
                raise ValueError("metadata payload too large")
            metadata = VideoFrameMetadata.model_validate(json.loads(raw))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError, ValidationError):
            async with self._state_lock:
                self._malformed_metadata += 1
            return

        async with self._state_lock:
            self._metadata_received += 1
            try:
                self._matcher.add_metadata(metadata)
            except DuplicateMetadataError:
                return

    def _reset_state(self) -> None:
        self._phase = WebRtcPhase.IDLE
        self._session_id: str | None = None
        self._device_session_id: str | None = None
        self._connection_state: str | None = None
        self._frames_received = 0
        self._metadata_received = 0
        self._malformed_metadata = 0
        self._width: int | None = None
        self._height: int | None = None
        self._video_codec: str | None = None
        self._negotiation_started_at_ns: int | None = None
        self._first_frame_at_ns: int | None = None
        self._last_frame_at_ns: int | None = None
        self._first_frame_latency_ms: float | None = None
        self._last_frame_pts: int | None = None
        self._last_frame_time_base_num: int | None = None
        self._last_frame_time_base_den: int | None = None
        self._last_error: str | None = None
        self._matcher = FrameMetadataMatcher(self._max_pending_metadata)
