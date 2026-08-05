from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from av import VideoFrame
from pydantic import ValidationError

from .adapters.webrtc import (
    DecodedVideoFrame,
    WebRtcControlChannel,
    WebRtcImuChannel,
    WebRtcPeer,
    WebRtcPeerCallbacks,
    WebRtcReceiverStats,
    WebRtcVideoRecordingSource,
    WebRtcVideoSource,
    create_aiortc_peer,
)
from .webrtc_matcher import (
    DuplicateMetadataError,
    FrameMetadataMatch,
    FrameMetadataMatcher,
)
from .webrtc_models import (
    IMU_TELEMETRY_ADAPTER,
    ImuCapabilities,
    ImuChannelState,
    ImuSample,
    ImuSensorStatus,
    ImuSensorType,
    ImuTelemetryStatus,
    StreamControlCommand,
    StreamControlState,
    StreamControlStatus,
    VideoFrameMetadata,
    WebRtcAnswer,
    WebRtcOffer,
    WebRtcPhase,
    WebRtcStatus,
)

IMU_MAX_PAYLOAD_BYTES = 16_384
RECORDING_FRAME_MAX_AGE_NS = 2_000_000_000
LOGGER = logging.getLogger(__name__)


class PairingTokenError(RuntimeError):
    """Raised when signaling authentication fails."""


class WebRtcSessionError(RuntimeError):
    """A safe signaling failure that never contains SDP or credentials."""


class StreamControlUnavailableError(RuntimeError):
    """Raised when no current Glass3 control channel can accept commands."""


class StreamControlCommandTimeoutError(RuntimeError):
    """Raised when Glass3 does not acknowledge a control command in time."""


class StreamControlCommandError(RuntimeError):
    """Raised when a control command cannot be sent safely."""


class CaptureTelemetrySink(Protocol):
    async def on_connection_started(
        self,
        connection_session_id: str,
        device_session_id: str,
        observed_at_client_monotonic_ns: int,
    ) -> None: ...

    async def on_connection_state(
        self,
        connection_session_id: str,
        state: str,
        observed_at_client_monotonic_ns: int,
    ) -> None: ...

    async def on_imu_capabilities(
        self,
        connection_session_id: str,
        capabilities: ImuCapabilities,
        received_at_client_monotonic_ns: int,
    ) -> None: ...

    async def on_imu_sample(
        self,
        connection_session_id: str,
        sample: ImuSample,
        received_at_client_monotonic_ns: int,
    ) -> None: ...

    async def on_frame_metadata_match(
        self,
        connection_session_id: str,
        match: FrameMetadataMatch,
    ) -> None: ...

    async def on_video_frame_metadata(
        self,
        connection_session_id: str,
        metadata: VideoFrameMetadata,
        received_at_client_monotonic_ns: int,
        camera_start_generation: int,
        ingest_status: str,
    ) -> None: ...


class DecodedFrameSink(Protocol):
    """Receives decoded frames after ingest state has been updated."""

    async def submit_gateway_frame(
        self,
        *,
        session_id: str,
        connection_session_id: str,
        frame_index: int,
        received_at_client_monotonic_ns: int,
        decoded_frame: VideoFrame,
    ) -> None: ...


class ImuSampleSink(Protocol):
    """Receives every validated IMU sample without performing UI work."""

    async def submit_imu_sample(
        self,
        *,
        session_id: str,
        sample: ImuSample,
        received_at_client_monotonic_ns: int,
    ) -> None: ...


WebRtcPeerFactory = Callable[[WebRtcPeerCallbacks], WebRtcPeer]


@dataclass
class _ImuSensorAccumulator:
    sample_count: int = 0
    first_received_at_ns: int | None = None
    last_received_at_ns: int | None = None
    latest_sequence_number: int | None = None
    sequence_gaps: int = 0
    out_of_order_samples: int = 0
    last_event_to_callback_delta_ns: int | None = None
    min_event_to_callback_delta_ns: int | None = None
    max_event_to_callback_delta_ns: int | None = None
    last_sample: ImuSample | None = None


class WebRtcSessionRuntime:
    """Own one authenticated WebRTC receive session and bounded stream metrics."""

    def __init__(
        self,
        pairing_token: str,
        peer_factory: WebRtcPeerFactory = create_aiortc_peer,
        *,
        perf_clock: Callable[[], int] = time.perf_counter_ns,
        max_pending_metadata: int = 256,
        control_command_timeout_seconds: float = 3.0,
        capture_telemetry_sink: CaptureTelemetrySink | None = None,
        perception_live_frame_sink: DecodedFrameSink | None = None,
        display_frame_sink: DecodedFrameSink | None = None,
        display_imu_sink: ImuSampleSink | None = None,
    ) -> None:
        if len(pairing_token) < 16:
            raise ValueError("pairing_token must contain at least 16 characters")
        self._pairing_token = pairing_token
        self._peer_factory = peer_factory
        self._perf_clock = perf_clock
        self._max_pending_metadata = max_pending_metadata
        if control_command_timeout_seconds <= 0:
            raise ValueError("control_command_timeout_seconds must be positive")
        self._control_command_timeout_seconds = control_command_timeout_seconds
        self._capture_telemetry_sink = capture_telemetry_sink
        self._perception_live_frame_sink = perception_live_frame_sink
        self._display_frame_sink = display_frame_sink
        self._display_imu_sink = display_imu_sink
        self._session_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._control_command_lock = asyncio.Lock()
        self._peer: WebRtcPeer | None = None
        self._video_source: WebRtcVideoSource | None = None
        self._pending_control_commands: dict[
            str, asyncio.Future[StreamControlStatus]
        ] = {}
        self._generation = 0
        self._reset_state()

    def set_capture_telemetry_sink(self, sink: CaptureTelemetrySink) -> None:
        self._capture_telemetry_sink = sink

    def set_perception_live_frame_sink(self, sink: DecodedFrameSink | None) -> None:
        """Attach an enqueue-only perception consumer for decoded source frames."""

        self._perception_live_frame_sink = sink

    def set_display_frame_sink(self, sink: DecodedFrameSink) -> None:
        """Attach the enqueue-only native UI consumer to the decoded frame fan-out."""

        self._display_frame_sink = sink

    def set_display_imu_sink(self, sink: ImuSampleSink) -> None:
        """Attach the bounded native pose-preview consumer."""

        self._display_imu_sink = sink

    async def status(self) -> WebRtcStatus:
        receiver_stats = WebRtcReceiverStats()
        peer = self._peer
        if peer is not None:
            try:
                receiver_stats = await peer.receiver_stats()
            except Exception:
                LOGGER.debug("failed to sample WebRTC receiver stats", exc_info=True)
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
            packet_total = receiver_stats.packets_received + receiver_stats.packets_lost
            packet_loss_percent = (
                round(receiver_stats.packets_lost * 100 / packet_total, 3)
                if packet_total > 0
                else 0.0
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
                metadata_anchor_matches=matcher.anchor_matches,
                metadata_ordered_gap_matches=matcher.ordered_gap_matches,
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
                rtp_packets_received=receiver_stats.packets_received,
                rtp_packets_lost=receiver_stats.packets_lost,
                rtp_packet_loss_percent=packet_loss_percent,
                rtp_jitter_ms=receiver_stats.jitter_ms,
                corrupt_frames_dropped=receiver_stats.corrupt_frames_dropped,
                first_frame_latency_ms=self._first_frame_latency_ms,
                last_frame_pts=self._last_frame_pts,
                last_frame_time_base_num=self._last_frame_time_base_num,
                last_frame_time_base_den=self._last_frame_time_base_den,
                metadata_rtp_origin_90khz=matcher.metadata_origin_90khz,
                metadata_calibrated=matcher.calibrated,
                metadata_calibration_support=matcher.calibration_support,
                last_frame_received_at_perf_counter_ns=self._last_frame_at_ns,
                last_error=self._last_error,
            )

    async def control_status(self) -> StreamControlStatus:
        async with self._state_lock:
            return self._control_status.model_copy(deep=True)

    async def imu_status(self) -> ImuTelemetryStatus:
        async with self._state_lock:
            sensors: dict[ImuSensorType, ImuSensorStatus] = {}
            for sensor_type, accumulator in self._imu_sensors.items():
                observed_rate_hz = None
                if (
                    accumulator.sample_count > 1
                    and accumulator.first_received_at_ns is not None
                    and accumulator.last_received_at_ns is not None
                ):
                    duration_ns = (
                        accumulator.last_received_at_ns
                        - accumulator.first_received_at_ns
                    )
                    if duration_ns > 0:
                        observed_rate_hz = round(
                            (accumulator.sample_count - 1) * 1_000_000_000 / duration_ns,
                            3,
                        )
                sensors[sensor_type] = ImuSensorStatus(
                    sample_count=accumulator.sample_count,
                    observed_rate_hz=observed_rate_hz,
                    first_received_at_perf_counter_ns=accumulator.first_received_at_ns,
                    last_received_at_perf_counter_ns=accumulator.last_received_at_ns,
                    latest_sequence_number=accumulator.latest_sequence_number,
                    sequence_gaps=accumulator.sequence_gaps,
                    out_of_order_samples=accumulator.out_of_order_samples,
                    last_event_to_callback_delta_ns=(
                        accumulator.last_event_to_callback_delta_ns
                    ),
                    min_event_to_callback_delta_ns=(
                        accumulator.min_event_to_callback_delta_ns
                    ),
                    max_event_to_callback_delta_ns=(
                        accumulator.max_event_to_callback_delta_ns
                    ),
                    last_sample=(
                        accumulator.last_sample.model_copy(deep=True)
                        if accumulator.last_sample is not None
                        else None
                    ),
                )
            return ImuTelemetryStatus(
                session_id=self._session_id,
                device_session_id=self._device_session_id,
                channel_state=self._imu_channel_state,
                messages_received=self._imu_messages_received,
                capabilities_received=self._imu_capabilities_received,
                samples_received=self._imu_samples_received,
                malformed_messages=self._imu_malformed_messages,
                capabilities=(
                    self._imu_capabilities.model_copy(deep=True)
                    if self._imu_capabilities is not None
                    else None
                ),
                sensors=sensors,
            )

    async def send_control_command(
        self,
        command: StreamControlCommand,
    ) -> StreamControlStatus:
        async with self._control_command_lock:
            loop = asyncio.get_running_loop()
            acknowledgement: asyncio.Future[StreamControlStatus] = loop.create_future()
            async with self._state_lock:
                channel = self._control_channel
                generation = self._generation
                if channel is None or not channel.is_open:
                    raise StreamControlUnavailableError(
                        "Glass3 stream control channel is unavailable"
                    )
                self._pending_control_commands[command.command_id] = acknowledgement
                transition = (
                    StreamControlState.STARTING
                    if command.action == "start"
                    else StreamControlState.STOPPING
                )
                self._control_status = StreamControlStatus(
                    command_id=command.command_id,
                    state=transition,
                )
                try:
                    channel.send(command.model_dump_json())
                except Exception as error:
                    self._pending_control_commands.pop(command.command_id, None)
                    self._control_status = StreamControlStatus(
                        command_id=command.command_id,
                        state=StreamControlState.ERROR,
                        detail="control command send failed",
                    )
                    raise StreamControlCommandError(
                        "failed to send Glass3 stream control command"
                    ) from error

            try:
                return await asyncio.wait_for(
                    acknowledgement,
                    timeout=self._control_command_timeout_seconds,
                )
            except TimeoutError as error:
                async with self._state_lock:
                    if generation == self._generation and self._control_channel is channel:
                        self._control_status = StreamControlStatus(
                            command_id=command.command_id,
                            state=StreamControlState.ERROR,
                            detail="control command acknowledgement timed out",
                        )
                raise StreamControlCommandTimeoutError(
                    "Glass3 did not acknowledge the stream control command"
                ) from error
            finally:
                async with self._state_lock:
                    pending = self._pending_control_commands.get(command.command_id)
                    if pending is acknowledgement:
                        self._pending_control_commands.pop(command.command_id, None)

    async def accept_offer(self, offer: WebRtcOffer, pairing_token: str) -> WebRtcAnswer:
        if not hmac.compare_digest(pairing_token, self._pairing_token):
            raise PairingTokenError("invalid pairing token")

        async with self._session_lock:
            self._generation += 1
            generation = self._generation
            async with self._state_lock:
                previous_session_id = self._session_id
                self._fail_pending_control_locked(
                    StreamControlUnavailableError("WebRTC session was replaced")
                )
                self._reset_state()
            if previous_session_id is not None and self._capture_telemetry_sink is not None:
                await self._emit_capture_event(
                    "on_connection_state",
                    previous_session_id,
                    "replaced",
                    self._perf_clock(),
                )
            peer, self._peer = self._peer, None
            self._video_source = None
            if peer is not None:
                await peer.close()

            callbacks = WebRtcPeerCallbacks(
                on_connection_state=lambda state: self._on_connection_state(generation, state),
                on_video_source=lambda source: self._on_video_source(generation, source),
                on_video_frame=lambda frame: self._on_video_frame(generation, frame),
                on_metadata=lambda payload: self._on_metadata(generation, payload),
                on_control_channel_ready=lambda channel: self._on_control_channel_ready(
                    generation, channel
                ),
                on_control_channel_closed=lambda channel: self._on_control_channel_closed(
                    generation, channel
                ),
                on_control_status=lambda channel, payload: self._on_control_status(
                    generation, channel, payload
                ),
                on_imu_channel_ready=lambda channel: self._on_imu_channel_ready(
                    generation, channel
                ),
                on_imu_channel_closed=lambda channel: self._on_imu_channel_closed(
                    generation, channel
                ),
                on_imu_telemetry=lambda channel, payload: self._on_imu_telemetry(
                    generation, channel, payload
                ),
            )
            peer = self._peer_factory(callbacks)
            self._peer = peer
            session_id = uuid.uuid4().hex
            async with self._state_lock:
                self._phase = WebRtcPhase.NEGOTIATING
                self._session_id = session_id
                self._device_session_id = offer.device_session_id
                self._negotiation_started_at_ns = self._perf_clock()
                negotiation_started_at_ns = self._negotiation_started_at_ns
            await self._emit_capture_event(
                "on_connection_started",
                session_id,
                offer.device_session_id,
                negotiation_started_at_ns,
            )

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

    async def recording_source(self) -> WebRtcVideoRecordingSource | None:
        async with self._state_lock:
            if (
                self._video_source is None
                or self._session_id is None
                or self._width is None
                or self._height is None
                or self._phase is not WebRtcPhase.STREAMING
                or self._last_frame_at_ns is None
                or self._camera_start_generation is None
                or self._perf_clock() - self._last_frame_at_ns
                > RECORDING_FRAME_MAX_AGE_NS
            ):
                return None
            return WebRtcVideoRecordingSource(
                connection_session_id=self._session_id,
                source=self._video_source,
                width=self._width,
                height=self._height,
                camera_start_generation=self._camera_start_generation,
            )

    async def close(self) -> None:
        async with self._session_lock:
            self._generation += 1
            async with self._state_lock:
                session_id = self._session_id
                self._fail_pending_control_locked(
                    StreamControlUnavailableError("WebRTC session closed")
                )
                self._control_channel = None
                self._control_status = StreamControlStatus(
                    state=StreamControlState.UNAVAILABLE,
                    detail="WebRTC session is closed",
                )
                self._set_imu_unavailable_locked()
            if session_id is not None and self._capture_telemetry_sink is not None:
                await self._emit_capture_event(
                    "on_connection_state",
                    session_id,
                    "closed",
                    self._perf_clock(),
                )
            peer, self._peer = self._peer, None
            self._video_source = None
            if peer is not None:
                await peer.close()
            async with self._state_lock:
                self._phase = WebRtcPhase.IDLE
                self._connection_state = "closed"

    async def _on_connection_state(self, generation: int, state: str) -> None:
        if generation != self._generation:
            return
        async with self._state_lock:
            session_id = self._session_id
            self._connection_state = state
            if state == "connected" and self._phase is WebRtcPhase.NEGOTIATING:
                self._phase = WebRtcPhase.CONNECTED
            elif state in {"disconnected", "closed"}:
                self._phase = WebRtcPhase.DISCONNECTED
                if state == "closed":
                    self._set_control_unavailable_locked("control channel is closed")
                    self._set_imu_unavailable_locked()
            elif state == "failed":
                self._phase = WebRtcPhase.FAILED
                self._last_error = "WebRTC peer connection failed"
                self._set_control_unavailable_locked("WebRTC peer connection failed")
                self._set_imu_unavailable_locked()
        if session_id is not None and self._capture_telemetry_sink is not None:
            await self._emit_capture_event(
                "on_connection_state",
                session_id,
                state,
                self._perf_clock(),
            )

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
            frame_index = self._frames_received - 1
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
            first_match = self._matcher.add_frame(
                frame.pts,
                frame_index=frame_index,
                time_base_num=(
                    frame.time_base.numerator if frame.time_base is not None else None
                ),
                time_base_den=(
                    frame.time_base.denominator if frame.time_base is not None else None
                ),
                received_at_client_monotonic_ns=now_ns,
            )
            matches = (
                (() if first_match is None else (first_match,))
                + self._matcher.drain_matches()
            )
            session_id = self._session_id
        if session_id is not None:
            for match in matches:
                await self._emit_capture_event(
                    "on_frame_metadata_match",
                    session_id,
                    match,
                )
            if frame.video_frame is not None:
                for sink_name, sink in (
                    ("native display", self._display_frame_sink),
                    ("perception", self._perception_live_frame_sink),
                ):
                    if sink is None:
                        continue
                    try:
                        await sink.submit_gateway_frame(
                            session_id=session_id,
                            connection_session_id=session_id,
                            frame_index=frame_index,
                            received_at_client_monotonic_ns=now_ns,
                            decoded_frame=frame.video_frame,
                        )
                    except Exception:
                        LOGGER.exception("%s frame sink rejected decoded frame", sink_name)

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

        received_at_ns = self._perf_clock()
        async with self._state_lock:
            self._metadata_received += 1
            self._camera_start_generation = metadata.camera_start_generation
            first_match = None
            ingest_status = "accepted"
            try:
                first_match = self._matcher.add_metadata(
                    metadata,
                    received_at_client_monotonic_ns=received_at_ns,
                )
            except DuplicateMetadataError:
                ingest_status = "duplicate"
            matches = (
                (() if first_match is None else (first_match,))
                + self._matcher.drain_matches()
            )
            session_id = self._session_id
        if session_id is not None and self._capture_telemetry_sink is not None:
            await self._emit_capture_event(
                "on_video_frame_metadata",
                session_id,
                metadata,
                received_at_ns,
                metadata.camera_start_generation,
                ingest_status,
            )
        if session_id is not None:
            for match in matches:
                await self._emit_capture_event(
                    "on_frame_metadata_match",
                    session_id,
                    match,
                )

    async def _on_control_channel_ready(
        self,
        generation: int,
        channel: WebRtcControlChannel,
    ) -> None:
        if generation != self._generation or not channel.is_open:
            return
        async with self._state_lock:
            if generation != self._generation:
                return
            if self._control_channel is not None and self._control_channel is not channel:
                self._fail_pending_control_locked(
                    StreamControlUnavailableError("stream control channel was replaced")
                )
            if self._control_channel is channel:
                return
            self._control_channel = channel
            self._control_status = StreamControlStatus(state=StreamControlState.READY)

    async def _on_control_channel_closed(
        self,
        generation: int,
        channel: WebRtcControlChannel,
    ) -> None:
        if generation != self._generation:
            return
        async with self._state_lock:
            if generation == self._generation and self._control_channel is channel:
                self._set_control_unavailable_locked("control channel is closed")

    async def _on_control_status(
        self,
        generation: int,
        channel: WebRtcControlChannel,
        payload: str | bytes,
    ) -> None:
        try:
            if isinstance(payload, bytes):
                if len(payload) > 4096:
                    raise ValueError("control status payload too large")
                raw = payload.decode("utf-8")
            else:
                raw = payload
            if len(raw) > 4096:
                raise ValueError("control status payload too large")
            status = StreamControlStatus.model_validate_json(raw)
        except (UnicodeDecodeError, ValueError, ValidationError):
            return

        if generation != self._generation:
            return
        async with self._state_lock:
            if generation != self._generation or self._control_channel is not channel:
                return
            self._control_status = status
            if status.command_id is not None:
                pending = self._pending_control_commands.get(status.command_id)
                if pending is not None and not pending.done():
                    pending.set_result(status)

    async def _on_imu_channel_ready(
        self,
        generation: int,
        channel: WebRtcImuChannel,
    ) -> None:
        if generation != self._generation or not channel.is_open:
            return
        async with self._state_lock:
            if generation != self._generation or not channel.is_open:
                return
            if self._imu_channel is channel:
                return
            self._imu_channel = channel
            self._imu_channel_state = (
                ImuChannelState.RECEIVING
                if self._imu_samples_received > 0
                else ImuChannelState.READY
            )

    async def _on_imu_channel_closed(
        self,
        generation: int,
        channel: WebRtcImuChannel,
    ) -> None:
        if generation != self._generation:
            return
        async with self._state_lock:
            if generation == self._generation and self._imu_channel is channel:
                self._set_imu_unavailable_locked()

    async def _on_imu_telemetry(
        self,
        generation: int,
        channel: WebRtcImuChannel,
        payload: str | bytes,
    ) -> None:
        if generation != self._generation:
            return
        message: ImuCapabilities | ImuSample | None = None
        malformed = False
        try:
            if isinstance(payload, bytes):
                if len(payload) > IMU_MAX_PAYLOAD_BYTES:
                    raise ValueError("IMU telemetry payload too large")
                raw: str | bytes = payload
            else:
                if len(payload.encode("utf-8")) > IMU_MAX_PAYLOAD_BYTES:
                    raise ValueError("IMU telemetry payload too large")
                raw = payload
            message = IMU_TELEMETRY_ADAPTER.validate_json(raw)
        except (UnicodeError, ValueError, ValidationError):
            malformed = True

        received_at_ns = (
            self._perf_clock()
            if isinstance(message, ImuSample)
            or (message is not None and self._capture_telemetry_sink is not None)
            else None
        )
        async with self._state_lock:
            if generation != self._generation or self._imu_channel is not channel:
                return
            self._imu_messages_received += 1
            if malformed or message is None:
                self._imu_malformed_messages += 1
                return
            if isinstance(message, ImuCapabilities):
                self._imu_capabilities_received += 1
                self._imu_capabilities = message
            else:
                self._imu_samples_received += 1
                self._imu_channel_state = ImuChannelState.RECEIVING
                accumulator = self._imu_sensors[message.sensor_type]
                accumulator.sample_count += 1
                if accumulator.first_received_at_ns is None:
                    accumulator.first_received_at_ns = received_at_ns
                accumulator.last_received_at_ns = received_at_ns
                previous_sequence = accumulator.latest_sequence_number
                if previous_sequence is None:
                    accumulator.latest_sequence_number = message.sequence_number
                elif message.sequence_number <= previous_sequence:
                    accumulator.out_of_order_samples += 1
                else:
                    accumulator.sequence_gaps += max(
                        0,
                        message.sequence_number - previous_sequence - 1,
                    )
                    accumulator.latest_sequence_number = message.sequence_number
                delta_ns = (
                    message.received_at_elapsed_realtime_ns
                    - message.sensor_event_monotonic_ns
                )
                accumulator.last_event_to_callback_delta_ns = delta_ns
                accumulator.min_event_to_callback_delta_ns = (
                    delta_ns
                    if accumulator.min_event_to_callback_delta_ns is None
                    else min(accumulator.min_event_to_callback_delta_ns, delta_ns)
                )
                accumulator.max_event_to_callback_delta_ns = (
                    delta_ns
                    if accumulator.max_event_to_callback_delta_ns is None
                    else max(accumulator.max_event_to_callback_delta_ns, delta_ns)
                )
                accumulator.last_sample = message
            session_id = self._session_id
        if session_id is None or received_at_ns is None:
            return
        if isinstance(message, ImuCapabilities):
            await self._emit_capture_event(
                "on_imu_capabilities",
                session_id,
                message,
                received_at_ns,
            )
        else:
            await self._emit_capture_event(
                "on_imu_sample",
                session_id,
                message,
                received_at_ns,
            )
            if self._display_imu_sink is not None:
                try:
                    await self._display_imu_sink.submit_imu_sample(
                        session_id=session_id,
                        sample=message,
                        received_at_client_monotonic_ns=received_at_ns,
                    )
                except Exception:
                    LOGGER.exception("native IMU preview rejected a sample")

    async def _emit_capture_event(self, method_name: str, *args: object) -> None:
        sink = self._capture_telemetry_sink
        if sink is None:
            return
        try:
            method = getattr(sink, method_name)
            await method(*args)
        except Exception:
            LOGGER.exception("capture telemetry sink failed during %s", method_name)

    def _set_control_unavailable_locked(self, detail: str) -> None:
        self._control_channel = None
        self._control_status = StreamControlStatus(
            state=StreamControlState.UNAVAILABLE,
            detail=detail,
        )
        self._fail_pending_control_locked(StreamControlUnavailableError(detail))

    def _set_imu_unavailable_locked(self) -> None:
        self._imu_channel = None
        self._imu_channel_state = ImuChannelState.UNAVAILABLE

    def _fail_pending_control_locked(self, error: Exception) -> None:
        for pending in self._pending_control_commands.values():
            if not pending.done():
                pending.set_exception(error)
        self._pending_control_commands.clear()

    def _reset_state(self) -> None:
        self._phase = WebRtcPhase.IDLE
        self._session_id: str | None = None
        self._device_session_id: str | None = None
        self._connection_state: str | None = None
        self._frames_received = 0
        self._metadata_received = 0
        self._camera_start_generation: int | None = None
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
        self._control_channel: WebRtcControlChannel | None = None
        self._control_status = StreamControlStatus(
            state=StreamControlState.UNAVAILABLE,
            detail="control channel is not ready",
        )
        self._imu_channel: WebRtcImuChannel | None = None
        self._imu_channel_state = ImuChannelState.UNAVAILABLE
        self._imu_messages_received = 0
        self._imu_capabilities_received = 0
        self._imu_samples_received = 0
        self._imu_malformed_messages = 0
        self._imu_capabilities: ImuCapabilities | None = None
        self._imu_sensors = {
            sensor_type: _ImuSensorAccumulator() for sensor_type in ImuSensorType
        }
        self._matcher = FrameMetadataMatcher(self._max_pending_metadata)
