from __future__ import annotations

import asyncio
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from ahrs.filters import Madgwick

from .webrtc_models import ImuSample, ImuSensorType


@dataclass(frozen=True, slots=True)
class ImuPoseSnapshot:
    session_id: str | None
    quaternion_wxyz: tuple[float, float, float, float]
    roll_degrees: float
    pitch_degrees: float
    yaw_degrees: float
    accelerometer_mps2: tuple[float, float, float] | None
    gyroscope_radps: tuple[float, float, float] | None
    samples_received: int
    samples_processed: int
    queue_overflow_count: int
    recent_rate_hz: float
    latest_sample_age_ms: float | None


@dataclass(frozen=True, slots=True)
class _QueuedSample:
    session_id: str
    sample: ImuSample
    received_at_client_monotonic_ns: int


class ImuPreviewRuntime:
    """Bounded Madgwick pose preview; VIO remains the authoritative future pose source."""

    def __init__(self, *, queue_size: int = 512, beta: float = 0.05) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self._queue: queue.Queue[_QueuedSample | None] = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._filter = Madgwick(frequency=100.0, beta=beta)
        self._quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._session_id: str | None = None
        self._accelerometer: tuple[float, float, float] | None = None
        self._gyroscope: tuple[float, float, float] | None = None
        self._last_gyro_timestamp_ns: int | None = None
        self._latest_received_at_ns: int | None = None
        self._samples_received = 0
        self._samples_processed = 0
        self._queue_overflow_count = 0
        self._processed_at_ns: deque[int] = deque(maxlen=400)
        self._closed = False
        self._worker = threading.Thread(
            target=self._run,
            name="imu-preview",
            daemon=False,
        )
        self._worker.start()

    async def submit_imu_sample(
        self,
        *,
        session_id: str,
        sample: ImuSample,
        received_at_client_monotonic_ns: int,
    ) -> None:
        queued = _QueuedSample(session_id, sample, received_at_client_monotonic_ns)
        with self._lock:
            self._samples_received += 1
        try:
            self._queue.put_nowait(queued)
        except queue.Full:
            with self._lock:
                self._queue_overflow_count += 1

    def snapshot(self) -> ImuPoseSnapshot:
        now_ns = time.perf_counter_ns()
        with self._lock:
            cutoff_ns = now_ns - 2_000_000_000
            recent = tuple(value for value in self._processed_at_ns if value >= cutoff_ns)
            rate_hz = 0.0
            if len(recent) > 1 and recent[-1] > recent[0]:
                rate_hz = (len(recent) - 1) * 1_000_000_000 / (recent[-1] - recent[0])
            quaternion = tuple(float(value) for value in self._quaternion)
            roll, pitch, yaw = _quaternion_to_euler_degrees(quaternion)
            age_ms = None
            if self._latest_received_at_ns is not None:
                age_ms = max(0.0, (now_ns - self._latest_received_at_ns) / 1_000_000)
            return ImuPoseSnapshot(
                session_id=self._session_id,
                quaternion_wxyz=quaternion,
                roll_degrees=roll,
                pitch_degrees=pitch,
                yaw_degrees=yaw,
                accelerometer_mps2=self._accelerometer,
                gyroscope_radps=self._gyroscope,
                samples_received=self._samples_received,
                samples_processed=self._samples_processed,
                queue_overflow_count=self._queue_overflow_count,
                recent_rate_hz=round(rate_hz, 3),
                latest_sample_age_ms=round(age_ms, 3) if age_ms is not None else None,
            )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        while True:
            try:
                self._queue.put_nowait(None)
                break
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    continue
        await asyncio.to_thread(self._worker.join, 5.0)
        if self._worker.is_alive():
            raise TimeoutError("IMU preview worker did not stop")

    def _run(self) -> None:
        while True:
            queued = self._queue.get()
            if queued is None:
                return
            sample = queued.sample
            with self._lock:
                self._session_id = queued.session_id
                self._latest_received_at_ns = queued.received_at_client_monotonic_ns
                if sample.sensor_type is ImuSensorType.ACCELEROMETER:
                    self._accelerometer = sample.values
                else:
                    self._gyroscope = sample.values
                    self._update_orientation(sample)
                self._samples_processed += 1
                self._processed_at_ns.append(time.perf_counter_ns())

    def _update_orientation(self, gyroscope: ImuSample) -> None:
        if self._accelerometer is None:
            self._last_gyro_timestamp_ns = gyroscope.sensor_event_monotonic_ns
            return
        previous_ns = self._last_gyro_timestamp_ns
        self._last_gyro_timestamp_ns = gyroscope.sensor_event_monotonic_ns
        if previous_ns is None or gyroscope.sensor_event_monotonic_ns <= previous_ns:
            return
        delta_seconds = (gyroscope.sensor_event_monotonic_ns - previous_ns) / 1_000_000_000
        if not 0.0001 <= delta_seconds <= 0.2:
            return
        updated = self._filter.updateIMU(
            self._quaternion,
            gyr=np.asarray(gyroscope.values, dtype=np.float64),
            acc=np.asarray(self._accelerometer, dtype=np.float64),
            dt=delta_seconds,
        )
        if updated is not None and np.isfinite(updated).all():
            self._quaternion = np.asarray(updated, dtype=np.float64)


def _quaternion_to_euler_degrees(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch_term = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(pitch_term)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))
