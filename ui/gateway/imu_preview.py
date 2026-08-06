from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from sensor_preprocessing import (
    ImuOrientationConfig,
    ImuOrientationFilter,
    SensorCalibration,
    quaternion_to_euler_degrees,
)

from .webrtc_models import ImuSample, ImuSensorType

_RATE_WINDOW_NS = 2_000_000_000


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

    def __init__(
        self,
        *,
        queue_size: int = 512,
        beta: float | None = None,
        orientation_config: ImuOrientationConfig | None = None,
        calibration: SensorCalibration | None = None,
    ) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self._queue: queue.Queue[_QueuedSample | None] = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        config = orientation_config or ImuOrientationConfig()
        if beta is not None:
            config = config.model_copy(update={"madgwick_beta": beta})
        self._orientation = ImuOrientationFilter(
            config,
            **_calibration_filter_kwargs(calibration),
        )
        self._session_id: str | None = None
        self._accelerometer: tuple[float, float, float] | None = None
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
            quaternion = self._orientation.relative_quaternion_wxyz
            roll, pitch, yaw = quaternion_to_euler_degrees(quaternion)
            age_ms = None
            if self._latest_received_at_ns is not None:
                age_ms = max(0.0, (now_ns - self._latest_received_at_ns) / 1_000_000)
            return ImuPoseSnapshot(
                session_id=self._session_id,
                quaternion_wxyz=quaternion,
                roll_degrees=roll,
                pitch_degrees=pitch,
                yaw_degrees=yaw,
                accelerometer_mps2=self._orientation.accelerometer_mps2,
                gyroscope_radps=self._orientation.gyroscope_radps,
                samples_received=self._samples_received,
                samples_processed=self._samples_processed,
                queue_overflow_count=self._queue_overflow_count,
                recent_rate_hz=round(rate_hz, 3),
                latest_sample_age_ms=round(age_ms, 3) if age_ms is not None else None,
            )

    # These narrow aliases keep older diagnostics/tests able to inspect the
    # runtime without maintaining a second fusion implementation.
    @property
    def _quaternion(self) -> np.ndarray:
        return self._orientation._quaternion

    @_quaternion.setter
    def _quaternion(self, value: np.ndarray) -> None:
        self._orientation._quaternion = np.asarray(value, dtype=np.float64)
        self._orientation._smoothed_quaternion = self._orientation._quaternion.copy()

    @property
    def _orientation_initialized(self) -> bool:
        return self._orientation.initialized

    @_orientation_initialized.setter
    def _orientation_initialized(self, value: bool) -> None:
        self._orientation._initialized = bool(value)

    @property
    def _reference_quaternion(self) -> np.ndarray | None:
        return self._orientation._reference_quaternion

    @_reference_quaternion.setter
    def _reference_quaternion(self, value: np.ndarray | None) -> None:
        self._orientation._reference_quaternion = value

    @property
    def _last_gyro_timestamp_ns(self) -> int | None:
        return self._orientation.last_gyro_timestamp_ns

    @_last_gyro_timestamp_ns.setter
    def _last_gyro_timestamp_ns(self, value: int | None) -> None:
        self._orientation._last_gyro_timestamp_ns = value

    @property
    def _accelerometer(self) -> tuple[float, float, float] | None:
        return self._orientation.accelerometer_mps2

    @_accelerometer.setter
    def _accelerometer(self, value: tuple[float, float, float] | None) -> None:
        self._orientation._accelerometer = (
            None if value is None else np.asarray(value, dtype=np.float64)
        )

    def _update_orientation(self, gyroscope: ImuSample) -> None:
        self._orientation.process(
            gyroscope.sensor_type,
            gyroscope.values,
            gyroscope.sensor_event_monotonic_ns,
        )

    def reset_orientation(self) -> None:
        """Use the current fused pose as the preview's new relative origin."""

        with self._lock:
            self._orientation.reset_reference()

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
                if sample.sensor_type is ImuSensorType.GYROSCOPE:
                    self._record_gyroscope_timing(
                        sample.sensor_event_monotonic_ns,
                        queued.received_at_client_monotonic_ns,
                    )
                self._orientation.process(
                    sample.sensor_type,
                    sample.values,
                    sample.sensor_event_monotonic_ns,
                )
                self._samples_processed += 1

    def _begin_session(self, session_id: str) -> None:
        self._session_id = session_id
        self._orientation.reset()
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

    @classmethod
    def from_sensor_config(
        cls,
        config_path: str,
        *,
        queue_size: int = 512,
    ) -> ImuPreviewRuntime:
        from sensor_preprocessing import SensorPreprocessingConfig

        config = SensorPreprocessingConfig.load(config_path)
        return cls(
            queue_size=queue_size,
            orientation_config=config.imu_orientation,
            calibration=SensorCalibration.load(config.calibration_file),
        )


def _calibration_filter_kwargs(
    calibration: SensorCalibration | None,
) -> dict[str, object]:
    if calibration is None:
        return {}
    imu = calibration.imu
    return {
        "raw_imu_to_body_axes": imu.raw_imu_to_body_axes,
        "gyroscope_bias_rad_s": imu.gyroscope_bias_rad_s,
        "accelerometer_bias_m_s2": imu.accelerometer_bias_m_s2,
        "accelerometer_scale": imu.accelerometer_scale,
    }
