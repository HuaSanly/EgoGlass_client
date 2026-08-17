from __future__ import annotations

import csv
import os
import queue
import re
import shutil
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO

from ui.gateway.webrtc_models import ImuSample, ImuSensorType

IMU_HEADER = ("sensor_type", "sequence", "timestamp_ns", "x", "y", "z")
_STOP = object()


@dataclass(frozen=True, slots=True)
class ImuWriterStats:
    rows: int = 0
    accelerometer_rows: int = 0
    gyroscope_rows: int = 0
    sequence_gaps: int = 0
    queue_size: int = 0
    bytes_written: int = 0
    first_timestamp_ns: int | None = None
    last_timestamp_ns: int | None = None
    accelerometer_rate_hz: float = 0.0
    gyroscope_rate_hz: float = 0.0


class ImuCaptureWriter:
    """Batch IMU rows to a private partial file and publish atomically."""

    def __init__(
        self,
        root: Path,
        *,
        capture_id: str | None = None,
        queue_size: int = 8192,
        batch_size: int = 1000,
        flush_interval_seconds: float = 1.0,
        fsync_interval_seconds: float = 30.0,
    ) -> None:
        if queue_size < 1 or batch_size < 1:
            raise ValueError("queue_size and batch_size must be positive")
        self.root = root.expanduser().resolve()
        self.capture_id = capture_id or uuid.uuid4().hex
        if re.fullmatch(r"[0-9a-f]{32}", self.capture_id) is None:
            raise ValueError("capture_id must be a UUID4 hex string")
        self.partial_dir = self.root / f".imu-capture-{self.capture_id}.partial"
        self.final_dir = self.root / self.capture_id
        self.partial_path = self.partial_dir / "imu.csv"
        self.final_path = self.final_dir / "imu.csv"
        self._queue: queue.Queue[object] = queue.Queue(maxsize=queue_size)
        self._batch_size = batch_size
        self._flush_interval_seconds = flush_interval_seconds
        self._fsync_interval_seconds = fsync_interval_seconds
        self._thread: threading.Thread | None = None
        self._file: TextIO | None = None
        self._lock = threading.Lock()
        self._started = False
        self._finished = False
        self._error: BaseException | None = None
        self._last_sequence: dict[ImuSensorType, int] = {}
        self._first_timestamp: dict[ImuSensorType, int] = {}
        self._last_timestamp: dict[ImuSensorType, int] = {}
        self._stats = ImuWriterStats()
        self._last_fsync_at = time.monotonic()

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    @property
    def stats(self) -> ImuWriterStats:
        with self._lock:
            return replace(self._stats, queue_size=self._queue.qsize())

    def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("writer already started")
            if self.final_dir.exists() or self.partial_dir.exists():
                raise FileExistsError(f"capture directory already exists: {self.capture_id}")
            self.root.mkdir(parents=True, exist_ok=True)
            self.partial_dir.mkdir()
            handle = self.partial_path.open("w", encoding="utf-8", newline="")
            csv.writer(handle, lineterminator="\n").writerow(IMU_HEADER)
            handle.flush()
            self._file = handle
            self._started = True
            self._thread = threading.Thread(
                target=self._run,
                name="imu-csv-writer",
                daemon=True,
            )
            self._thread.start()

    def append(self, sample: ImuSample) -> None:
        with self._lock:
            if not self._started or self._finished:
                raise RuntimeError("writer is not accepting samples")
            if self._error is not None:
                raise RuntimeError("writer failed") from self._error
            previous_sequence = self._last_sequence.get(sample.sensor_type)
            if previous_sequence is not None:
                if sample.sequence_number <= previous_sequence:
                    raise ValueError(f"{sample.sensor_type} sequence must be strictly increasing")
                gaps = sample.sequence_number - previous_sequence - 1
            else:
                gaps = 0
            previous_timestamp = self._last_timestamp.get(sample.sensor_type)
            if (
                previous_timestamp is not None
                and sample.sensor_event_monotonic_ns <= previous_timestamp
            ):
                raise ValueError(f"{sample.sensor_type} timestamp must be strictly increasing")
            try:
                self._queue.put_nowait(sample)
            except queue.Full as error:
                self._error = error
                raise RuntimeError("IMU writer queue overflow") from error
            self._last_sequence[sample.sensor_type] = sample.sequence_number
            self._first_timestamp.setdefault(
                sample.sensor_type,
                sample.sensor_event_monotonic_ns,
            )
            self._last_timestamp[sample.sensor_type] = sample.sensor_event_monotonic_ns
            accelerometer_rows = self._stats.accelerometer_rows + (
                sample.sensor_type is ImuSensorType.ACCELEROMETER
            )
            gyroscope_rows = self._stats.gyroscope_rows + (
                sample.sensor_type is ImuSensorType.GYROSCOPE
            )
            self._stats = ImuWriterStats(
                rows=self._stats.rows + 1,
                accelerometer_rows=accelerometer_rows,
                gyroscope_rows=gyroscope_rows,
                sequence_gaps=self._stats.sequence_gaps + gaps,
                bytes_written=self._stats.bytes_written,
                first_timestamp_ns=(
                    sample.sensor_event_monotonic_ns
                    if self._stats.first_timestamp_ns is None
                    else min(
                        self._stats.first_timestamp_ns,
                        sample.sensor_event_monotonic_ns,
                    )
                ),
                last_timestamp_ns=(
                    sample.sensor_event_monotonic_ns
                    if self._stats.last_timestamp_ns is None
                    else max(
                        self._stats.last_timestamp_ns,
                        sample.sensor_event_monotonic_ns,
                    )
                ),
                accelerometer_rate_hz=self._sensor_rate(
                    ImuSensorType.ACCELEROMETER,
                    accelerometer_rows,
                ),
                gyroscope_rate_hz=self._sensor_rate(
                    ImuSensorType.GYROSCOPE,
                    gyroscope_rows,
                ),
            )

    def finish(self, *, publish: bool) -> Path | None:
        with self._lock:
            if not self._started or self._finished:
                return self.final_path if self.final_path.exists() else None
            self._finished = True
            thread = self._thread
        self._queue.put(_STOP)
        if thread is not None:
            thread.join()
        with self._lock:
            error = self._error
        if error is not None or not publish:
            self._discard()
            if error is not None:
                raise RuntimeError("IMU CSV writer failed") from error
            return None
        if self.stats.accelerometer_rows == 0 or self.stats.gyroscope_rows == 0:
            self._discard()
            raise ValueError("both accelerometer and gyroscope samples are required")
        self.final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.partial_dir, self.final_dir)
        return self.final_path

    def discard(self) -> None:
        self._discard()

    def _run(self) -> None:
        handle = self._file
        if handle is None:
            return
        writer = csv.writer(handle, lineterminator="\n")
        batch: list[ImuSample] = []
        try:
            while True:
                try:
                    item = self._queue.get(timeout=self._flush_interval_seconds)
                except queue.Empty:
                    item = None
                if item is _STOP:
                    self._write_batch(writer, handle, batch, force_fsync=True)
                    break
                if item is not None:
                    batch.append(item)  # type: ignore[arg-type]
                if len(batch) >= self._batch_size or (item is None and batch):
                    self._write_batch(writer, handle, batch)
                    batch.clear()
        except BaseException as error:
            with self._lock:
                self._error = error
        finally:
            with suppress(Exception):
                handle.close()

    def _write_batch(
        self,
        writer: object,
        handle: TextIO,
        batch: list[ImuSample],
        *,
        force_fsync: bool = False,
    ) -> None:
        if not batch:
            if force_fsync:
                handle.flush()
                os.fsync(handle.fileno())
            return
        for sample in batch:
            writer.writerow(  # type: ignore[attr-defined]
                (
                    sample.sensor_type.value,
                    sample.sequence_number,
                    sample.sensor_event_monotonic_ns,
                    *sample.values,
                )
            )
        handle.flush()
        now = time.monotonic()
        if force_fsync or now - self._last_fsync_at >= self._fsync_interval_seconds:
            os.fsync(handle.fileno())
            self._last_fsync_at = now
        with self._lock, suppress(Exception):
            self._stats = replace(self._stats, bytes_written=handle.tell())

    def _discard(self) -> None:
        with self._lock:
            thread = self._thread if self._thread is not None and self._thread.is_alive() else None
        if thread is not None:
            self._queue.put(_STOP)
            thread.join()
        if self.partial_dir.exists():
            shutil.rmtree(self.partial_dir)

    def _sensor_rate(self, sensor_type: ImuSensorType, count: int) -> float:
        first = self._first_timestamp.get(sensor_type)
        last = self._last_timestamp.get(sensor_type)
        if count < 2 or first is None or last is None or last <= first:
            return 0.0
        return round((count - 1) * 1_000_000_000 / (last - first), 3)
