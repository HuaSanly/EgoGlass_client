from __future__ import annotations

import asyncio
import json
from fractions import Fraction

import pytest

from egoglass_ingest_gateway.adapters.webrtc import (
    DecodedVideoFrame,
    WebRtcControlChannel,
    WebRtcImuChannel,
    WebRtcPeerCallbacks,
)
from egoglass_ingest_gateway.webrtc_models import (
    ImuChannelState,
    StreamControlAction,
    StreamControlCommand,
    StreamControlState,
    WebRtcOffer,
    WebRtcPhase,
    WebRtcViewerOffer,
)
from egoglass_ingest_gateway.webrtc_runtime import (
    IMU_MAX_PAYLOAD_BYTES,
    RECORDING_FRAME_MAX_AGE_NS,
    PairingTokenError,
    StreamControlCommandError,
    StreamControlCommandTimeoutError,
    StreamControlUnavailableError,
    WebRtcSessionRuntime,
    WebRtcViewerUnavailableError,
)

TOKEN = "test-pairing-token-123456"


class FakePeer:
    def __init__(self, callbacks: WebRtcPeerCallbacks) -> None:
        self.callbacks = callbacks
        self.closed = False

    @property
    def negotiated_video_codec(self) -> str:
        return "H264"

    async def accept_offer(self, offer: WebRtcOffer) -> str:
        assert offer.type == "offer"
        return "v=0\r\nanswer-session-description"

    async def close(self) -> None:
        self.closed = True


class FakeVideoSource:
    def __init__(self) -> None:
        self.subscriptions: list[tuple[object, bool]] = []

    def subscribe(self, *, buffered: bool) -> object:
        track = object()
        self.subscriptions.append((track, buffered))
        return track


class FakeViewerPeer:
    def __init__(self, track: object) -> None:
        self.track = track
        self.closed = False

    async def accept_offer(self, offer: WebRtcViewerOffer) -> str:
        assert offer.type == "offer"
        return "v=0\r\nviewer-answer-description"

    async def close(self) -> None:
        self.closed = True


class FakeControlChannel(WebRtcControlChannel):
    def __init__(self, *, fail_send: bool = False) -> None:
        self.open = True
        self.fail_send = fail_send
        self.sent: list[str] = []

    @property
    def is_open(self) -> bool:
        return self.open

    def send(self, message: str) -> None:
        if self.fail_send:
            raise ConnectionError("test send failure")
        self.sent.append(message)


class FakeImuChannel(WebRtcImuChannel):
    def __init__(self) -> None:
        self.open = True

    @property
    def is_open(self) -> bool:
        return self.open


def offer(device_session_id: str = "device-session-0001") -> WebRtcOffer:
    return WebRtcOffer(
        device_session_id=device_session_id,
        sdp="v=0\r\noffer-session-description",
    )


def metadata_json(frame_id: int = 1, rtp_timestamp: int = 90_000) -> str:
    return json.dumps(
        {
            "schema_version": "1.0",
            "message_type": "video_frame",
            "stream_id": "camera",
            "frame_id": frame_id,
            "captured_at_rokid_sdk_ms": 1000,
            "received_at_elapsed_realtime_ns": 2_000_000_000,
            "video_at_monotonic_ns": 2_000_000_000,
            "rtp_timestamp_90khz": rtp_timestamp,
            "width": 1280,
            "height": 720,
            "rotation_degrees": 0,
            "capture_config_id": "720p30",
        }
    )


def imu_capabilities_json() -> str:
    return json.dumps(
        {
            "schema_version": "0.1",
            "message_type": "imu_capabilities",
            "source": "android_sensor_manager",
            "requested_sampling_period_us": 10_000,
            "sensors": [
                {
                    "sensor_type": "accelerometer",
                    "android_sensor_type": 1,
                    "name": "BMI acceleration",
                    "vendor": "Bosch",
                    "version": 1,
                    "unit": "m_s2",
                    "resolution": 0.001,
                    "max_range": 78.4,
                    "min_delay_us": 2500,
                    "max_delay_us": 100000,
                    "is_wake_up": False,
                },
                {
                    "sensor_type": "gyroscope",
                    "android_sensor_type": 4,
                    "name": "BMI gyroscope",
                    "vendor": "Bosch",
                    "version": 1,
                    "unit": "rad_s",
                    "resolution": 0.001,
                    "max_range": 34.9,
                    "min_delay_us": 2500,
                    "max_delay_us": 100000,
                    "is_wake_up": False,
                },
            ],
            "missing_sensor_types": [],
        }
    )


def imu_sample_json(
    sensor_type: str,
    sequence_number: int,
    event_ns: int,
    callback_ns: int,
) -> str:
    return json.dumps(
        {
            "schema_version": "0.1",
            "message_type": "imu_sample",
            "sensor_type": sensor_type,
            "android_sensor_type": 1 if sensor_type == "accelerometer" else 4,
            "sequence_number": sequence_number,
            "sensor_event_monotonic_ns": event_ns,
            "received_at_elapsed_realtime_ns": callback_ns,
            "accuracy": 3,
            "values": [0.1, -0.2, 0.3],
        }
    )


def test_authenticated_session_receives_and_matches_video_metadata() -> None:
    peers: list[FakePeer] = []
    viewers: list[FakeViewerPeer] = []
    perf_values = iter([1_000_000_000, 1_120_000_000])

    def factory(callbacks: WebRtcPeerCallbacks) -> FakePeer:
        peer = FakePeer(callbacks)
        peers.append(peer)
        return peer

    def viewer_factory(track: object) -> FakeViewerPeer:
        viewer = FakeViewerPeer(track)
        viewers.append(viewer)
        return viewer

    async def scenario() -> None:
        runtime = WebRtcSessionRuntime(
            TOKEN,
            factory,
            viewer_factory,
            perf_clock=lambda: next(perf_values),
        )
        answer = await runtime.accept_offer(offer(), TOKEN)
        assert answer.type == "answer"
        await peers[0].callbacks.on_connection_state("connected")
        source = FakeVideoSource()
        await peers[0].callbacks.on_video_source(source)
        await peers[0].callbacks.on_metadata(metadata_json())
        await peers[0].callbacks.on_video_frame(
            DecodedVideoFrame(1280, 720, 0, Fraction(1, 90_000))
        )
        viewer_answer = await runtime.accept_viewer_offer(
            WebRtcViewerOffer(sdp="v=0\r\nviewer-offer-description")
        )
        status = await runtime.status()

        assert status.phase is WebRtcPhase.STREAMING
        assert status.frames_received == 1
        assert status.metadata_received == 1
        assert status.metadata_matched == 1
        assert status.metadata_rtp_origin_90khz == 90_000
        assert status.video_codec == "H264"
        assert viewer_answer.type == "answer"
        assert source.subscriptions == [(viewers[0].track, False)]
        assert status.first_frame_latency_ms == 120.0
        assert (status.width, status.height) == (1280, 720)

    asyncio.run(scenario())


def test_recording_source_requires_recent_streaming_frames_not_control_state() -> None:
    peers: list[FakePeer] = []
    now_ns = 1_000_000_000

    def factory(callbacks: WebRtcPeerCallbacks) -> FakePeer:
        peer = FakePeer(callbacks)
        peers.append(peer)
        return peer

    async def scenario() -> None:
        nonlocal now_ns
        runtime = WebRtcSessionRuntime(TOKEN, factory, perf_clock=lambda: now_ns)
        answer = await runtime.accept_offer(offer(), TOKEN)
        source = FakeVideoSource()
        channel = FakeControlChannel()
        await peers[0].callbacks.on_connection_state("connected")
        await peers[0].callbacks.on_video_source(source)
        assert await runtime.recording_source() is None

        await peers[0].callbacks.on_video_frame(
            DecodedVideoFrame(1280, 720, 0, Fraction(1, 90_000))
        )
        await peers[0].callbacks.on_control_channel_ready(channel)
        await peers[0].callbacks.on_control_status(
            channel,
            json.dumps(
                {
                    "schema_version": "1.0",
                    "message_type": "stream_control_status",
                    "command_id": None,
                    "state": "error",
                    "detail": "device control acknowledgement timed out",
                }
            ),
        )

        recording_source = await runtime.recording_source()
        assert recording_source is not None
        assert recording_source.session_id == answer.session_id
        assert (recording_source.width, recording_source.height) == (1280, 720)
        assert recording_source.source is source
        assert (await runtime.control_status()).state is StreamControlState.ERROR

        now_ns += RECORDING_FRAME_MAX_AGE_NS + 1
        assert await runtime.recording_source() is None

    asyncio.run(scenario())


def test_imu_telemetry_tracks_rates_gaps_clocks_and_malformed_messages() -> None:
    peers: list[FakePeer] = []
    perf_values = iter(
        [
            10,
            1_000_000_000,
            1_010_000_000,
            1_020_000_000,
            2_000_000_000,
            2_020_000_000,
        ]
    )

    def factory(callbacks: WebRtcPeerCallbacks) -> FakePeer:
        peer = FakePeer(callbacks)
        peers.append(peer)
        return peer

    async def scenario() -> None:
        runtime = WebRtcSessionRuntime(TOKEN, factory, perf_clock=lambda: next(perf_values))
        await runtime.accept_offer(offer(), TOKEN)
        channel = FakeImuChannel()
        await peers[0].callbacks.on_imu_channel_ready(channel)
        assert (await runtime.imu_status()).channel_state is ImuChannelState.READY

        await peers[0].callbacks.on_imu_telemetry(channel, imu_capabilities_json())
        await peers[0].callbacks.on_imu_telemetry(
            channel, imu_sample_json("accelerometer", 0, 1_000, 1_100)
        )
        await peers[0].callbacks.on_imu_telemetry(
            channel, imu_sample_json("accelerometer", 2, 2_000, 2_300)
        )
        await peers[0].callbacks.on_imu_telemetry(
            channel, imu_sample_json("accelerometer", 1, 1_500, 1_700)
        )
        await peers[0].callbacks.on_imu_telemetry(
            channel, imu_sample_json("gyroscope", 0, 3_000, 3_050)
        )
        await peers[0].callbacks.on_imu_telemetry(
            channel, imu_sample_json("gyroscope", 1, 4_000, 4_070)
        )
        await peers[0].callbacks.on_imu_telemetry(channel, "not-json")
        await peers[0].callbacks.on_imu_telemetry(
            channel,
            "x" * (IMU_MAX_PAYLOAD_BYTES + 1),
        )

        status = await runtime.imu_status()
        assert status.channel_state is ImuChannelState.RECEIVING
        assert status.messages_received == 8
        assert status.capabilities_received == 1
        assert status.samples_received == 5
        assert status.malformed_messages == 2
        assert status.capabilities is not None
        accelerometer = status.sensors["accelerometer"]
        gyroscope = status.sensors["gyroscope"]
        assert accelerometer.sample_count == 3
        assert accelerometer.observed_rate_hz == 100.0
        assert accelerometer.latest_sequence_number == 2
        assert accelerometer.sequence_gaps == 1
        assert accelerometer.out_of_order_samples == 1
        assert accelerometer.last_event_to_callback_delta_ns == 200
        assert accelerometer.min_event_to_callback_delta_ns == 100
        assert accelerometer.max_event_to_callback_delta_ns == 300
        assert accelerometer.last_sample is not None
        assert accelerometer.last_sample.sequence_number == 1
        assert gyroscope.sample_count == 2
        assert gyroscope.observed_rate_hz == 50.0

        await peers[0].callbacks.on_connection_state("disconnected")
        assert (await runtime.imu_status()).channel_state is ImuChannelState.RECEIVING

        channel.open = False
        await peers[0].callbacks.on_imu_channel_closed(channel)
        assert (await runtime.imu_status()).channel_state is ImuChannelState.UNAVAILABLE

    asyncio.run(scenario())


def test_authenticated_offer_replaces_active_peer_and_ignores_stale_callbacks() -> None:
    peers: list[FakePeer] = []
    viewers: list[FakeViewerPeer] = []

    def factory(callbacks: WebRtcPeerCallbacks) -> FakePeer:
        peer = FakePeer(callbacks)
        peers.append(peer)
        return peer

    def viewer_factory(track: object) -> FakeViewerPeer:
        viewer = FakeViewerPeer(track)
        viewers.append(viewer)
        return viewer

    async def scenario() -> None:
        runtime = WebRtcSessionRuntime(TOKEN, factory, viewer_factory)
        with pytest.raises(WebRtcViewerUnavailableError):
            await runtime.accept_viewer_offer(
                WebRtcViewerOffer(sdp="v=0\r\nviewer-offer-description")
            )
        with pytest.raises(PairingTokenError):
            await runtime.accept_offer(offer(), "wrong-pairing-token")

        first_answer = await runtime.accept_offer(offer(), TOKEN)
        source = FakeVideoSource()
        await peers[0].callbacks.on_connection_state("connected")
        await peers[0].callbacks.on_video_source(source)
        await runtime.accept_viewer_offer(
            WebRtcViewerOffer(sdp="v=0\r\nviewer-offer-description")
        )
        await peers[0].callbacks.on_metadata("not-json")
        old_imu_channel = FakeImuChannel()
        await peers[0].callbacks.on_imu_channel_ready(old_imu_channel)
        await peers[0].callbacks.on_imu_telemetry(
            old_imu_channel,
            imu_sample_json("accelerometer", 0, 1_000, 1_100),
        )
        replacement = await runtime.accept_offer(offer("device-session-0002"), TOKEN)
        await peers[0].callbacks.on_connection_state("failed")
        await peers[0].callbacks.on_metadata("not-json")
        await peers[0].callbacks.on_imu_telemetry(
            old_imu_channel,
            imu_sample_json("accelerometer", 1, 2_000, 2_100),
        )
        status = await runtime.status()
        imu_status = await runtime.imu_status()

        assert peers[0].closed
        assert viewers[0].closed
        assert replacement.type == "answer"
        assert replacement.session_id != first_answer.session_id
        assert status.phase is WebRtcPhase.NEGOTIATING
        assert status.device_session_id == "device-session-0002"
        assert status.connection_state is None
        assert status.malformed_metadata == 0
        assert imu_status.channel_state is ImuChannelState.UNAVAILABLE
        assert imu_status.messages_received == 0
        assert imu_status.samples_received == 0

    asyncio.run(scenario())


def test_control_command_is_strictly_serialized_and_correlated_with_status() -> None:
    peers: list[FakePeer] = []

    def factory(callbacks: WebRtcPeerCallbacks) -> FakePeer:
        peer = FakePeer(callbacks)
        peers.append(peer)
        return peer

    async def scenario() -> None:
        runtime = WebRtcSessionRuntime(TOKEN, factory)
        command = StreamControlCommand(
            command_id="0123456789abcdef0123456789abcdef",
            action=StreamControlAction.START,
        )
        assert (await runtime.control_status()).state is StreamControlState.UNAVAILABLE
        with pytest.raises(StreamControlUnavailableError):
            await runtime.send_control_command(command)

        await runtime.accept_offer(offer(), TOKEN)
        channel = FakeControlChannel()
        await peers[0].callbacks.on_control_channel_ready(channel)
        assert (await runtime.control_status()).state is StreamControlState.READY

        pending = asyncio.create_task(runtime.send_control_command(command))
        await asyncio.sleep(0)
        assert json.loads(channel.sent[0]) == {
            "schema_version": "1.0",
            "message_type": "stream_control_command",
            "command_id": command.command_id,
            "action": "start",
        }
        assert (await runtime.control_status()).state is StreamControlState.STARTING

        await peers[0].callbacks.on_control_status(channel, "not-json")
        await peers[0].callbacks.on_control_status(
            channel,
            json.dumps(
                {
                    "schema_version": "1.0",
                    "message_type": "stream_control_status",
                    "command_id": command.command_id,
                    "state": "streaming",
                    "detail": None,
                    "unexpected": True,
                }
            ),
        )
        assert not pending.done()

        acknowledgement_json = json.dumps(
            {
                "schema_version": "1.0",
                "message_type": "stream_control_status",
                "command_id": command.command_id,
                "state": "streaming",
                "detail": None,
            }
        )
        await peers[0].callbacks.on_control_status(channel, acknowledgement_json)
        acknowledgement = await pending
        assert acknowledgement.command_id == command.command_id
        assert acknowledgement.state is StreamControlState.STREAMING
        assert (await runtime.control_status()) == acknowledgement

    asyncio.run(scenario())


def test_replacement_session_rejects_stale_control_channel_callbacks() -> None:
    peers: list[FakePeer] = []

    def factory(callbacks: WebRtcPeerCallbacks) -> FakePeer:
        peer = FakePeer(callbacks)
        peers.append(peer)
        return peer

    async def scenario() -> None:
        runtime = WebRtcSessionRuntime(TOKEN, factory)
        await runtime.accept_offer(offer(), TOKEN)
        old_channel = FakeControlChannel()
        await peers[0].callbacks.on_control_channel_ready(old_channel)
        command = StreamControlCommand(
            command_id="11111111111111111111111111111111",
            action=StreamControlAction.STOP,
        )
        pending = asyncio.create_task(runtime.send_control_command(command))
        await asyncio.sleep(0)

        await runtime.accept_offer(offer("device-session-0002"), TOKEN)
        with pytest.raises(StreamControlUnavailableError, match="replaced"):
            await pending
        new_channel = FakeControlChannel()
        await peers[1].callbacks.on_control_channel_ready(new_channel)
        await peers[0].callbacks.on_control_status(
            old_channel,
            json.dumps(
                {
                    "schema_version": "1.0",
                    "message_type": "stream_control_status",
                    "command_id": command.command_id,
                    "state": "stopped",
                    "detail": None,
                }
            ),
        )
        await peers[0].callbacks.on_control_channel_closed(old_channel)

        current = await runtime.control_status()
        assert current.state is StreamControlState.READY
        assert current.command_id is None

    asyncio.run(scenario())


def test_control_timeout_and_send_failure_leave_safe_error_status() -> None:
    peers: list[FakePeer] = []

    def factory(callbacks: WebRtcPeerCallbacks) -> FakePeer:
        peer = FakePeer(callbacks)
        peers.append(peer)
        return peer

    async def scenario() -> None:
        runtime = WebRtcSessionRuntime(
            TOKEN,
            factory,
            control_command_timeout_seconds=0.01,
        )
        await runtime.accept_offer(offer(), TOKEN)
        channel = FakeControlChannel()
        await peers[0].callbacks.on_control_channel_ready(channel)
        timeout_command = StreamControlCommand(
            command_id="22222222222222222222222222222222",
            action=StreamControlAction.START,
        )
        with pytest.raises(StreamControlCommandTimeoutError):
            await runtime.send_control_command(timeout_command)
        timeout_status = await runtime.control_status()
        assert timeout_status.state is StreamControlState.ERROR
        assert timeout_status.command_id == timeout_command.command_id

        failed_channel = FakeControlChannel(fail_send=True)
        await peers[0].callbacks.on_control_channel_ready(failed_channel)
        failed_command = StreamControlCommand(
            command_id="33333333333333333333333333333333",
            action=StreamControlAction.STOP,
        )
        with pytest.raises(StreamControlCommandError):
            await runtime.send_control_command(failed_command)
        failed_status = await runtime.control_status()
        assert failed_status.state is StreamControlState.ERROR
        assert failed_status.command_id == failed_command.command_id

    asyncio.run(scenario())
