from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from .webrtc_models import ImuSample, ImuSensorType

_RATE_WINDOW_NS = 2_000_000_000


@dataclass(frozen=True, slots=True)
class ImuTelemetryPoint:
    sequence: int
    sensor_type: ImuSensorType
    sensor_event_monotonic_ns: int
    received_at_client_monotonic_ns: int
    values: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class ImuTelemetrySnapshot:
    revision: int = 0
    connection_session_id: str | None = None
    accelerometer: tuple[ImuTelemetryPoint, ...] = ()
    gyroscope: tuple[ImuTelemetryPoint, ...] = ()
    accelerometer_rate_hz: float = 0.0
    gyroscope_rate_hz: float = 0.0
    latest_callback_latency_ms: float | None = None
    sequence_gap_count: int = 0
    duplicate_count: int = 0
    out_of_order_count: int = 0
    buffer_eviction_count: int = 0


class ImuTelemetryRuntime:
    """Keep a bounded raw IMU trace for monitoring without pose estimation."""

    def __init__(
        self,
        *,
        maximum_points_per_sensor: int = 600,
        clock: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if maximum_points_per_sensor < 2:
            raise ValueError("maximum_points_per_sensor must be at least two")
        self._maximum_points_per_sensor = maximum_points_per_sensor
        self._clock = clock
        self._lock = threading.Lock()
        self._accelerometer: deque[ImuTelemetryPoint] = deque(
            maxlen=maximum_points_per_sensor
        )
        self._gyroscope: deque[ImuTelemetryPoint] = deque(
            maxlen=maximum_points_per_sensor
        )
        self._last_sequence: dict[ImuSensorType, int] = {}
        self._revision = 0
        self._sequence_gap_count = 0
        self._duplicate_count = 0
        self._out_of_order_count = 0
        self._latest_callback_latency_ms: float | None = None
        self._connection_session_id: str | None = None
        self._buffer_eviction_count = 0

    async def submit_imu_sample(
        self,
        *,
        connection_session_id: str,
        sample: ImuSample,
        received_at_client_monotonic_ns: int,
    ) -> None:
        with self._lock:
            if connection_session_id != self._connection_session_id:
                self._reset_connection_locked(connection_session_id)
            previous = self._last_sequence.get(sample.sensor_type)
            if previous is not None:
                if sample.sequence_number == previous:
                    self._duplicate_count += 1
                elif sample.sequence_number < previous:
                    self._out_of_order_count += 1
                else:
                    self._sequence_gap_count += max(
                        0, sample.sequence_number - previous - 1
                    )
            if previous is None or sample.sequence_number > previous:
                self._last_sequence[sample.sensor_type] = sample.sequence_number

            points = self._points_for(sample.sensor_type)
            if len(points) == self._maximum_points_per_sensor:
                self._buffer_eviction_count += 1
            self._revision += 1
            points.append(
                ImuTelemetryPoint(
                    sequence=self._revision,
                    sensor_type=sample.sensor_type,
                    sensor_event_monotonic_ns=sample.sensor_event_monotonic_ns,
                    received_at_client_monotonic_ns=received_at_client_monotonic_ns,
                    values=sample.values,
                )
            )
            callback_delta_ns = (
                sample.received_at_elapsed_realtime_ns
                - sample.sensor_event_monotonic_ns
            )
            self._latest_callback_latency_ms = callback_delta_ns / 1_000_000

    def snapshot(self, now_ns: int | None = None) -> ImuTelemetrySnapshot:
        current_ns = self._clock() if now_ns is None else now_ns
        with self._lock:
            accelerometer = tuple(self._accelerometer)
            gyroscope = tuple(self._gyroscope)
            return ImuTelemetrySnapshot(
                revision=self._revision,
                connection_session_id=self._connection_session_id,
                accelerometer=accelerometer,
                gyroscope=gyroscope,
                accelerometer_rate_hz=_sample_rate(accelerometer, current_ns),
                gyroscope_rate_hz=_sample_rate(gyroscope, current_ns),
                latest_callback_latency_ms=self._latest_callback_latency_ms,
                sequence_gap_count=self._sequence_gap_count,
                duplicate_count=self._duplicate_count,
                out_of_order_count=self._out_of_order_count,
                buffer_eviction_count=self._buffer_eviction_count,
            )

    def samples_after(self, sequence: int) -> tuple[ImuTelemetryPoint, ...]:
        with self._lock:
            points = (*self._accelerometer, *self._gyroscope)
            return tuple(
                sorted(
                    (point for point in points if point.sequence > sequence),
                    key=lambda item: item.sequence,
                )
            )

    async def close(self) -> None:
        return None

    def _points_for(self, sensor_type: ImuSensorType) -> deque[ImuTelemetryPoint]:
        if sensor_type is ImuSensorType.ACCELEROMETER:
            return self._accelerometer
        return self._gyroscope

    def _reset_connection_locked(self, connection_session_id: str) -> None:
        self._connection_session_id = connection_session_id
        self._accelerometer.clear()
        self._gyroscope.clear()
        self._last_sequence.clear()
        self._sequence_gap_count = 0
        self._duplicate_count = 0
        self._out_of_order_count = 0
        self._buffer_eviction_count = 0
        self._latest_callback_latency_ms = None


def _sample_rate(points: tuple[ImuTelemetryPoint, ...], now_ns: int) -> float:
    if len(points) < 2 or now_ns - points[-1].received_at_client_monotonic_ns > _RATE_WINDOW_NS:
        return 0.0
    cutoff = points[-1].sensor_event_monotonic_ns - _RATE_WINDOW_NS
    recent = [
        point.sensor_event_monotonic_ns
        for point in points
        if point.sensor_event_monotonic_ns >= cutoff
    ]
    if len(recent) < 2 or recent[-1] <= recent[0]:
        return 0.0
    return round((len(recent) - 1) * 1_000_000_000 / (recent[-1] - recent[0]), 3)
