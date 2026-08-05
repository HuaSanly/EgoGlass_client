from __future__ import annotations

import asyncio
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from ahrs.common.orientation import acc2q
from ahrs.filters import Madgwick

from .webrtc_models import ImuSample, ImuSensorType

_RATE_WINDOW_NS = 2_000_000_000
_INITIAL_ACCELEROMETER_SAMPLES = 20


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
        self._reference_quaternion: np.ndarray | None = None
        self._orientation_initialized = False
        self._session_id: str | None = None
        self._accelerometer: tuple[float, float, float] | None = None
        self._accelerometer_window: deque[np.ndarray] = deque(
            maxlen=_INITIAL_ACCELEROMETER_SAMPLES
        )
        self._gyroscope: tuple[float, float, float] | None = None
        self._last_gyro_timestamp_ns: int | None = None
        self._latest_received_at_ns: int | None = None
        self._samples_received = 0
        self._samples_processed = 0
        self._queue_overflow_count = 0
        self._gyroscope_timing: deque[tuple[int, int]] = deque(maxlen=400)
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
            rate_hz = self._recent_gyroscope_rate_hz(now_ns)
            quaternion = tuple(
                float(value)
                for value in _relative_quaternion(
                    self._reference_quaternion,
                    self._quaternion,
                )
            )
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

    def reset_orientation(self) -> None:
        """Use the current fused pose as the preview's new relative origin."""

        with self._lock:
            self._reference_quaternion = (
                _normalized_quaternion(self._quaternion)
                if self._orientation_initialized
                else None
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
                if self._session_id != queued.session_id:
                    self._begin_session(queued.session_id)
                self._latest_received_at_ns = queued.received_at_client_monotonic_ns
                if sample.sensor_type is ImuSensorType.ACCELEROMETER:
                    self._accelerometer = sample.values
                    self._accelerometer_window.append(
                        np.asarray(sample.values, dtype=np.float64)
                    )
                else:
                    self._gyroscope = sample.values
                    self._record_gyroscope_timing(
                        sample.sensor_event_monotonic_ns,
                        queued.received_at_client_monotonic_ns,
                    )
                    self._update_orientation(sample)
                self._samples_processed += 1

    def _begin_session(self, session_id: str) -> None:
        self._session_id = session_id
        self._quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._reference_quaternion = None
        self._orientation_initialized = False
        self._accelerometer = None
        self._accelerometer_window.clear()
        self._gyroscope = None
        self._last_gyro_timestamp_ns = None
        self._gyroscope_timing.clear()

    def _record_gyroscope_timing(
        self,
        sensor_timestamp_ns: int,
        received_at_client_monotonic_ns: int,
    ) -> None:
        if (
            self._gyroscope_timing
            and sensor_timestamp_ns <= self._gyroscope_timing[-1][0]
        ):
            return
        self._gyroscope_timing.append(
            (sensor_timestamp_ns, received_at_client_monotonic_ns)
        )

    def _recent_gyroscope_rate_hz(self, now_ns: int) -> float:
        if not self._gyroscope_timing:
            return 0.0
        latest_sensor_ns, latest_received_ns = self._gyroscope_timing[-1]
        if now_ns - latest_received_ns > _RATE_WINDOW_NS:
            return 0.0
        cutoff_ns = latest_sensor_ns - _RATE_WINDOW_NS
        recent = tuple(
            sensor_ns
            for sensor_ns, _received_ns in self._gyroscope_timing
            if sensor_ns >= cutoff_ns
        )
        if len(recent) < 2 or recent[-1] <= recent[0]:
            return 0.0
        return (len(recent) - 1) * 1_000_000_000 / (recent[-1] - recent[0])

    def _update_orientation(self, gyroscope: ImuSample) -> None:
        current_ns = gyroscope.sensor_event_monotonic_ns
        previous_ns = self._last_gyro_timestamp_ns
        if previous_ns is not None and current_ns <= previous_ns:
            return
        self._last_gyro_timestamp_ns = current_ns
        if self._accelerometer is None:
            return
        if previous_ns is None:
            return
        if not self._orientation_initialized:
            if len(self._accelerometer_window) < _INITIAL_ACCELEROMETER_SAMPLES:
                return
            initial = acc2q(np.mean(tuple(self._accelerometer_window), axis=0))
            if not np.isfinite(initial).all():
                return
            self._quaternion = _normalized_quaternion(initial)
            self._reference_quaternion = self._quaternion.copy()
            self._orientation_initialized = True
            return
        delta_seconds = (current_ns - previous_ns) / 1_000_000_000
        if not 0.0001 <= delta_seconds <= 0.2:
            return
        updated = self._filter.updateIMU(
            self._quaternion,
            gyr=np.asarray(gyroscope.values, dtype=np.float64),
            acc=np.asarray(self._accelerometer, dtype=np.float64),
            dt=delta_seconds,
        )
        if updated is not None and np.isfinite(updated).all():
            self._quaternion = _normalized_quaternion(updated)


def _relative_quaternion(
    reference: np.ndarray | None,
    current: np.ndarray,
) -> np.ndarray:
    if reference is None:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    reference_inverse = _normalized_quaternion(reference).copy()
    reference_inverse[1:] *= -1.0
    return _normalized_quaternion(
        _quaternion_product(reference_inverse, _normalized_quaternion(current))
    )


def _normalized_quaternion(quaternion: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return values / norm


def _quaternion_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def _quaternion_to_euler_degrees(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch_term = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(pitch_term)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))
