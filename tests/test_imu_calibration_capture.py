from __future__ import annotations

import asyncio
import csv
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from ui.gateway.webrtc_models import (
    ImuCapabilities,
    ImuSample,
    ImuSensorDescriptor,
    ImuSensorType,
    StreamControlState,
    StreamControlStatus,
    WebRtcAnswer,
)
from ui.imu_calibration.app import create_app
from ui.imu_calibration.service import CapturePhase, ImuCalibrationService
from ui.imu_calibration.writer import IMU_HEADER, ImuCaptureWriter

CAPTURE_ID = "a" * 32
CONNECTION_ID = "connection-1"


class FakeRuntime:
    def __init__(self) -> None:
        self.capture_sink: object | None = None
        self.lifecycle_sink: object | None = None
        self.commands: list[object] = []
        self.closed = False
        self.control_state = StreamControlState.READY
        self.disconnect_on_stop = False

    def set_capture_telemetry_sink(self, sink: object) -> None:
        self.capture_sink = sink

    def set_imu_channel_lifecycle_sink(self, sink: object) -> None:
        self.lifecycle_sink = sink

    async def control_status(self) -> StreamControlStatus:
        return StreamControlStatus(state=self.control_state)

    async def send_control_command(self, command: object) -> StreamControlStatus:
        self.commands.append(command)
        if self.disconnect_on_stop and self.capture_sink is not None:
            await self.capture_sink.on_connection_state(  # type: ignore[attr-defined]
                CONNECTION_ID,
                "disconnected",
                10,
            )
        return StreamControlStatus(state=StreamControlState.STOPPED)

    async def accept_offer(self, _offer: object, token: str) -> WebRtcAnswer:
        if token != "test-token-123456":
            from ui.gateway.webrtc_runtime import PairingTokenError

            raise PairingTokenError("invalid pairing token")
        return WebRtcAnswer(session_id="b" * 32, sdp="v=0\r\n" + "x" * 16)

    async def close(self) -> None:
        self.closed = True


def imu_sample(
    sensor_type: ImuSensorType,
    sequence: int,
    timestamp_ns: int,
    values: tuple[float, float, float] = (0.1, -0.2, 0.3),
) -> ImuSample:
    return ImuSample(
        sensor_type=sensor_type,
        android_sensor_type=1 if sensor_type is ImuSensorType.ACCELEROMETER else 4,
        sequence_number=sequence,
        sensor_event_monotonic_ns=timestamp_ns,
        received_at_elapsed_realtime_ns=timestamp_ns + 1_000,
        accuracy=3,
        values=values,
    )


def complete_capabilities() -> ImuCapabilities:
    return ImuCapabilities(
        requested_sampling_period_us=10_000,
        sensors=[
            ImuSensorDescriptor(
                sensor_type=ImuSensorType.ACCELEROMETER,
                android_sensor_type=1,
                name="accelerometer",
                vendor="test",
                version=1,
                unit="m_s2",
                resolution=0.001,
                max_range=80.0,
                min_delay_us=2_500,
                max_delay_us=100_000,
                is_wake_up=False,
            ),
            ImuSensorDescriptor(
                sensor_type=ImuSensorType.GYROSCOPE,
                android_sensor_type=4,
                name="gyroscope",
                vendor="test",
                version=1,
                unit="rad_s",
                resolution=0.001,
                max_range=35.0,
                min_delay_us=2_500,
                max_delay_us=100_000,
                is_wake_up=False,
            ),
        ],
        missing_sensor_types=[],
    )


def test_writer_publishes_exact_raw_csv_and_keeps_sequence_gaps(tmp_path: Path) -> None:
    writer = ImuCaptureWriter(tmp_path, capture_id=CAPTURE_ID, batch_size=2)
    writer.start()
    writer.append(imu_sample(ImuSensorType.ACCELEROMETER, 10, 1_000, (1.25, -2.5, 3.75)))
    writer.append(imu_sample(ImuSensorType.GYROSCOPE, 20, 1_100, (-0.1, 0.2, -0.3)))
    writer.append(imu_sample(ImuSensorType.ACCELEROMETER, 12, 2_000))

    output = writer.finish(publish=True)

    assert output == tmp_path / CAPTURE_ID / "imu.csv"
    assert {path.name for path in output.parent.iterdir()} == {"imu.csv"}
    with output.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert tuple(rows[0]) == IMU_HEADER
    assert rows[1] == ["accelerometer", "10", "1000", "1.25", "-2.5", "3.75"]
    assert rows[2] == ["gyroscope", "20", "1100", "-0.1", "0.2", "-0.3"]
    assert writer.stats.sequence_gaps == 1
    assert not writer.partial_dir.exists()


@pytest.mark.parametrize(
    "samples",
    [
        (
            imu_sample(ImuSensorType.ACCELEROMETER, 1, 1_000),
            imu_sample(ImuSensorType.ACCELEROMETER, 1, 2_000),
        ),
        (
            imu_sample(ImuSensorType.ACCELEROMETER, 1, 2_000),
            imu_sample(ImuSensorType.ACCELEROMETER, 2, 1_000),
        ),
    ],
)
def test_writer_rejects_duplicate_or_non_monotonic_samples(
    tmp_path: Path,
    samples: tuple[ImuSample, ImuSample],
) -> None:
    writer = ImuCaptureWriter(tmp_path, capture_id=CAPTURE_ID)
    writer.start()
    writer.append(samples[0])
    with pytest.raises(ValueError):
        writer.append(samples[1])
    writer.finish(publish=False)
    assert not writer.partial_dir.exists()
    assert not writer.final_dir.exists()


def test_non_finite_raw_values_are_rejected_before_writing() -> None:
    with pytest.raises(ValidationError):
        imu_sample(ImuSensorType.GYROSCOPE, 1, 1_000, (float("nan"), 0.0, 0.0))


def test_queue_overflow_fails_and_removes_partial(tmp_path: Path) -> None:
    release = threading.Event()

    class BlockedWriter(ImuCaptureWriter):
        def _run(self) -> None:
            assert release.wait(timeout=5)
            super()._run()

    writer = BlockedWriter(tmp_path, capture_id=CAPTURE_ID, queue_size=1)
    writer.start()
    writer.append(imu_sample(ImuSensorType.ACCELEROMETER, 1, 1_000))
    with pytest.raises(RuntimeError, match="queue overflow"):
        writer.append(imu_sample(ImuSensorType.GYROSCOPE, 1, 1_100))
    release.set()
    with pytest.raises(RuntimeError, match="writer failed"):
        writer.finish(publish=False)
    assert not writer.partial_dir.exists()


def test_disk_write_failure_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = ImuCaptureWriter(tmp_path, capture_id=CAPTURE_ID, batch_size=1)

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(writer, "_write_batch", fail_write)
    writer.start()
    writer.append(imu_sample(ImuSensorType.ACCELEROMETER, 1, 1_000))
    with pytest.raises(RuntimeError, match="writer failed"):
        writer.finish(publish=True)
    assert not writer.partial_dir.exists()
    assert not writer.final_dir.exists()


def test_service_stops_video_once_and_publishes_both_sensors(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = FakeRuntime()
        service = ImuCalibrationService(  # type: ignore[arg-type]
            runtime,
            tmp_path,
            capture_id=CAPTURE_ID,
        )
        await service.on_connection_started(CONNECTION_ID, "device", 1)
        await service.on_imu_capabilities(CONNECTION_ID, complete_capabilities(), 2)
        await service.on_imu_capabilities(CONNECTION_ID, complete_capabilities(), 3)
        assert len(runtime.commands) == 1
        assert service.phase is CapturePhase.WAITING_SAMPLES

        await service.on_imu_sample(
            CONNECTION_ID,
            imu_sample(ImuSensorType.ACCELEROMETER, 1, 1_000),
            3,
        )
        await service.on_imu_sample(
            CONNECTION_ID,
            imu_sample(ImuSensorType.GYROSCOPE, 1, 1_100),
            4,
        )
        output = await service.finish()
        assert output == tmp_path / CAPTURE_ID / "imu.csv"
        assert service.phase is CapturePhase.COMPLETE

    asyncio.run(scenario())


def test_disconnect_during_stop_ack_cannot_restore_capture_state(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = FakeRuntime()
        runtime.disconnect_on_stop = True
        service = ImuCalibrationService(  # type: ignore[arg-type]
            runtime,
            tmp_path,
            capture_id=CAPTURE_ID,
        )
        await service.on_connection_started(CONNECTION_ID, "device", 1)
        await service.on_imu_capabilities(CONNECTION_ID, complete_capabilities(), 2)

        assert service.phase is CapturePhase.FAILED
        assert not service.writer.partial_dir.exists()

    asyncio.run(scenario())


def test_duration_and_manual_stop_publish_after_capture_started(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = FakeRuntime()
        service = ImuCalibrationService(  # type: ignore[arg-type]
            runtime,
            tmp_path,
            capture_id=CAPTURE_ID,
        )
        await service.on_connection_started(CONNECTION_ID, "device", 1)
        await service.on_imu_capabilities(CONNECTION_ID, complete_capabilities(), 2)
        await service.on_imu_sample(
            CONNECTION_ID,
            imu_sample(ImuSensorType.ACCELEROMETER, 1, 1_000),
            3,
        )
        await service.on_imu_sample(
            CONNECTION_ID,
            imu_sample(ImuSensorType.GYROSCOPE, 1, 1_100),
            4,
        )
        await service.wait_until_started_or_done()
        with pytest.raises(TimeoutError):
            await service.wait(0.001)
        output = await service.finish()
        assert output is not None and output.exists()

    asyncio.run(scenario())


def test_sample_inactivity_watchdog_fails_and_removes_partial(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = FakeRuntime()
        service = ImuCalibrationService(  # type: ignore[arg-type]
            runtime,
            tmp_path,
            capture_id=CAPTURE_ID,
        )
        await service.on_connection_started(CONNECTION_ID, "device", 1)
        await service.on_imu_capabilities(CONNECTION_ID, complete_capabilities(), 2)
        await service.on_imu_sample(
            CONNECTION_ID,
            imu_sample(ImuSensorType.ACCELEROMETER, 1, 1_000),
            3,
        )
        await service.watchdog(timeout_seconds=0.01)

        assert service.phase is CapturePhase.FAILED
        assert not service.writer.partial_dir.exists()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["disconnect", "channel", "malformed"])
def test_service_failure_discards_partial(tmp_path: Path, failure: str) -> None:
    async def scenario() -> None:
        runtime = FakeRuntime()
        service = ImuCalibrationService(  # type: ignore[arg-type]
            runtime,
            tmp_path,
            capture_id=CAPTURE_ID,
        )
        await service.on_connection_started(CONNECTION_ID, "device", 1)
        await service.on_imu_capabilities(CONNECTION_ID, complete_capabilities(), 2)
        await service.on_imu_sample(
            CONNECTION_ID,
            imu_sample(ImuSensorType.ACCELEROMETER, 1, 1_000),
            3,
        )
        if failure == "disconnect":
            await service.on_connection_state(CONNECTION_ID, "disconnected", 4)
        elif failure == "channel":
            await service.on_imu_channel_closed(CONNECTION_ID)
        else:
            await service.on_imu_message_rejected(CONNECTION_ID)

        assert service.phase is CapturePhase.FAILED
        assert not service.writer.partial_dir.exists()
        assert not service.writer.final_dir.exists()

    asyncio.run(scenario())


def test_lightweight_api_has_no_recording_routes(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    service = ImuCalibrationService(  # type: ignore[arg-type]
        runtime,
        tmp_path,
        capture_id=CAPTURE_ID,
    )
    app = create_app(runtime, service, None)  # type: ignore[arg-type]
    with TestClient(app) as client:
        assert client.get("/api/v1/health").json()["service"] == "imu-calibration-capture"
        assert client.get("/api/v1/recordings/status").status_code == 404
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404
    assert runtime.closed


def test_capture_module_does_not_import_qt_or_recording_control() -> None:
    source_root = Path(__file__).parents[1] / "ui" / "imu_calibration"
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_root.glob("*.py"))
    assert "PyQt" not in source
    assert "recording_control" not in source
