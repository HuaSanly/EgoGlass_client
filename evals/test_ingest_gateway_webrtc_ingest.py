from __future__ import annotations

import asyncio
import inspect
import json
from fractions import Fraction

from ingest_gateway.adapters.aiortc_peer import (
    lan_rtc_configuration,
)
from ingest_gateway.adapters.webrtc import (
    DecodedVideoFrame,
    WebRtcControlChannel,
    WebRtcImuChannel,
    WebRtcPeerCallbacks,
)
from ingest_gateway.webrtc_models import (
    ImuChannelState,
    StreamControlAction,
    StreamControlCommand,
    StreamControlState,
    WebRtcOffer,
    WebRtcPhase,
)
from ingest_gateway.webrtc_runtime import (
    RECORDING_FRAME_MAX_AGE_NS,
    WebRtcSessionRuntime,
)


def test_gateway_disables_per_frame_access_logs() -> None:
    from ingest_gateway.app import main

    assert "access_log=False" in inspect.getsource(main)


def test_reordered_metadata_and_authenticated_replacement_stay_bounded() -> None:
    assert lan_rtc_configuration().iceServers == []
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

    def payload(frame_id: int, timestamp: int, sdk_timestamp: int) -> str:
        return json.dumps(
            {
                "schema_version": "1.0",
                "message_type": "video_frame",
                "stream_id": "camera",
                "frame_id": frame_id,
                "camera_start_generation": 1,
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
        runtime = WebRtcSessionRuntime(token, Peer, max_pending_metadata=2)
        offer = WebRtcOffer(
            device_session_id="device-session-eval01",
            sdp="v=0\r\noffer-session-description",
        )
        await runtime.accept_offer(offer, token)
        await peers[0].callbacks.on_connection_state("connected")
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
        assert streaming.metadata_matched == 0
        assert not streaming.metadata_calibrated
        assert streaming.video_codec == "H264"
        assert streaming.unmatched_entries_dropped == 2
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
        assert not replacement_status.metadata_calibrated
        assert replacement_status.metadata_calibration_support == 0

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

    runtimes: list[WebRtcSessionRuntime] = []

    async def acknowledge(
        channel: ControlChannel,
        command: StreamControlCommand,
        state: StreamControlState,
    ) -> None:
        pending = asyncio.create_task(runtimes[0].send_control_command(command))
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
        runtime = WebRtcSessionRuntime(token, Peer)
        runtimes.append(runtime)
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

    asyncio.run(scenario())


def test_recording_availability_follows_video_flow_when_control_state_is_stale() -> None:
    peers: list[Peer] = []
    now_ns = 5_000_000_000

    class Peer:
        def __init__(self, callbacks: WebRtcPeerCallbacks) -> None:
            self.callbacks = callbacks
            peers.append(self)

        @property
        def negotiated_video_codec(self) -> str:
            return "H264"

        async def accept_offer(self, _offer: WebRtcOffer) -> str:
            return "v=0\r\nanswer-session-description"

        async def close(self) -> None:
            return None

    class VideoSource:
        def subscribe(self, *, buffered: bool) -> object:
            assert buffered
            return object()

    class ControlChannel(WebRtcControlChannel):
        @property
        def is_open(self) -> bool:
            return True

        def send(self, _message: str) -> None:
            return None

    async def scenario() -> None:
        nonlocal now_ns
        token = "eval-pairing-token-123456"
        runtime = WebRtcSessionRuntime(token, Peer, perf_clock=lambda: now_ns)
        await runtime.accept_offer(
            WebRtcOffer(
                device_session_id="device-session-recording-eval",
                sdp="v=0\r\noffer-session-description",
            ),
            token,
        )
        source = VideoSource()
        channel = ControlChannel()
        await peers[0].callbacks.on_connection_state("connected")
        await peers[0].callbacks.on_video_source(source)
        await peers[0].callbacks.on_metadata(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "message_type": "video_frame",
                    "stream_id": "camera",
                    "frame_id": 0,
                    "camera_start_generation": 1,
                    "captured_at_rokid_sdk_ms": 1000,
                    "received_at_elapsed_realtime_ns": 2_000_000_000,
                    "video_at_monotonic_ns": 2_000_000_000,
                    "rtp_timestamp_90khz": 90_000,
                    "width": 1280,
                    "height": 720,
                    "rotation_degrees": 0,
                    "capture_config_id": "720p30",
                }
            )
        )
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
                    "detail": "stale control status",
                }
            ),
        )

        assert await runtime.recording_source() is not None
        now_ns += RECORDING_FRAME_MAX_AGE_NS + 1
        assert await runtime.recording_source() is None

    asyncio.run(scenario())


def test_experimental_imu_stream_reports_both_sensors_and_resets_on_replacement() -> None:
    peers: list[Peer] = []
    clock_value = 0

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

    class ImuChannel(WebRtcImuChannel):
        @property
        def is_open(self) -> bool:
            return True

    def perf_clock() -> int:
        nonlocal clock_value
        current = clock_value
        clock_value += 5_000_000
        return current

    def capabilities() -> str:
        return json.dumps(
            {
                "schema_version": "0.1",
                "message_type": "imu_capabilities",
                "source": "android_sensor_manager",
                "requested_sampling_period_us": 10_000,
                "sensors": [
                    {
                        "sensor_type": sensor_type,
                        "android_sensor_type": android_type,
                        "name": f"eval {sensor_type}",
                        "vendor": "eval",
                        "version": 1,
                        "unit": unit,
                        "resolution": 0.001,
                        "max_range": 100.0,
                        "min_delay_us": 2500,
                        "max_delay_us": 100000,
                        "is_wake_up": False,
                    }
                    for sensor_type, android_type, unit in (
                        ("accelerometer", 1, "m_s2"),
                        ("gyroscope", 4, "rad_s"),
                    )
                ],
                "missing_sensor_types": [],
            }
        )

    def sample(sensor_type: str, sequence_number: int) -> str:
        event_ns = 1_000_000_000 + sequence_number * 10_000_000
        return json.dumps(
            {
                "schema_version": "0.1",
                "message_type": "imu_sample",
                "sensor_type": sensor_type,
                "android_sensor_type": 1 if sensor_type == "accelerometer" else 4,
                "sequence_number": sequence_number,
                "sensor_event_monotonic_ns": event_ns,
                "received_at_elapsed_realtime_ns": event_ns + 250_000,
                "accuracy": 3,
                "values": [0.1, 0.2, 0.3],
            }
        )

    async def scenario() -> None:
        token = "eval-pairing-token-123456"
        runtime = WebRtcSessionRuntime(token, Peer, perf_clock=perf_clock)
        await runtime.accept_offer(
            WebRtcOffer(
                device_session_id="device-session-imu-eval01",
                sdp="v=0\r\noffer-session-description",
            ),
            token,
        )
        channel = ImuChannel()
        await peers[0].callbacks.on_imu_channel_ready(channel)
        await peers[0].callbacks.on_imu_telemetry(channel, capabilities())
        for sequence_number in range(100):
            await peers[0].callbacks.on_imu_telemetry(
                channel,
                sample("accelerometer", sequence_number),
            )
            await peers[0].callbacks.on_imu_telemetry(
                channel,
                sample("gyroscope", sequence_number),
            )

        status = await runtime.imu_status()
        assert status.channel_state is ImuChannelState.RECEIVING
        assert status.samples_received == 200
        assert status.malformed_messages == 0
        assert status.sensors["accelerometer"].sample_count == 100
        assert status.sensors["gyroscope"].sample_count == 100
        assert status.sensors["accelerometer"].observed_rate_hz == 100.0
        assert status.sensors["gyroscope"].observed_rate_hz == 100.0
        assert status.sensors["accelerometer"].last_event_to_callback_delta_ns == 250_000

        await runtime.accept_offer(
            WebRtcOffer(
                device_session_id="device-session-imu-eval02",
                sdp="v=0\r\noffer-session-description",
            ),
            token,
        )
        await peers[0].callbacks.on_imu_telemetry(
            channel,
            sample("accelerometer", 100),
        )
        replacement = await runtime.imu_status()
        assert peers[0].closed
        assert replacement.channel_state is ImuChannelState.UNAVAILABLE
        assert replacement.samples_received == 0

    asyncio.run(scenario())
