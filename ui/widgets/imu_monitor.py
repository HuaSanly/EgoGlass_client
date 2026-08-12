from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, InfoBadge, SimpleCardWidget, StrongBodyLabel


@dataclass(frozen=True, slots=True)
class ImuChartSample:
    sensor_type: str
    sequence_number: int
    recording_time_ns: int
    received_at_client_monotonic_ns: int
    values: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class ImuMonitorStats:
    sample_rate_hz: float = 0.0
    latest_latency_ms: float | None = None
    sequence_gaps: int = 0
    duplicate_samples: int = 0
    out_of_order_samples: int = 0
    queue_overflows: int = 0
    recording_window_ns: int = 0
    csv_state: str = "idle"


class ImuMonitorWidget(SimpleCardWidget):
    """Bounded raw accelerometer and gyroscope monitor for the capture console."""

    _COLORS = ("#0f6cbd", "#16a085", "#d1495b")

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        maximum_samples: int = 600,
    ) -> None:
        super().__init__(parent)
        if maximum_samples < 2:
            raise ValueError("maximum_samples must be at least two")
        self.setObjectName("imuRawMonitor")
        self.setMinimumWidth(380)
        self._maximum_samples = maximum_samples
        self._samples: dict[str, deque[ImuChartSample]] = {
            "accelerometer": deque(maxlen=maximum_samples),
            "gyroscope": deque(maxlen=maximum_samples),
        }
        self._last_keys: set[tuple[str, int]] = set()
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(9)

        header = QHBoxLayout()
        header.addWidget(StrongBodyLabel("原始 IMU", self))
        header.addStretch(1)
        self.rate_badge = InfoBadge.info("0.0 Hz", self)
        self.latency_badge = InfoBadge.info("延迟 --", self)
        header.addWidget(self.rate_badge)
        header.addWidget(self.latency_badge)
        root.addLayout(header)

        self.accelerometer_plot = self._create_plot("加速度", "m/s²")
        self.gyroscope_plot = self._create_plot("角速度", "rad/s")
        root.addWidget(self.accelerometer_plot)
        root.addWidget(self.gyroscope_plot)
        self._curves = {
            "accelerometer": tuple(
                self.accelerometer_plot.plot(pen=pg.mkPen(color, width=1.5), name=axis)
                for axis, color in zip(("X", "Y", "Z"), self._COLORS, strict=True)
            ),
            "gyroscope": tuple(
                self.gyroscope_plot.plot(pen=pg.mkPen(color, width=1.5), name=axis)
                for axis, color in zip(("X", "Y", "Z"), self._COLORS, strict=True)
            ),
        }

        stats = QGridLayout()
        stats.setHorizontalSpacing(12)
        stats.setVerticalSpacing(4)
        self.sequence_label = CaptionLabel("丢序 0  ·  重复 0  ·  乱序 0", self)
        self.overflow_label = CaptionLabel("队列溢出 0", self)
        self.window_label = CaptionLabel("录制时间窗 00:00.000", self)
        self.csv_label = CaptionLabel("CSV 等待写入", self)
        stats.addWidget(self.sequence_label, 0, 0)
        stats.addWidget(self.overflow_label, 0, 1, Qt.AlignmentFlag.AlignRight)
        stats.addWidget(self.window_label, 1, 0)
        stats.addWidget(self.csv_label, 1, 1, Qt.AlignmentFlag.AlignRight)
        root.addLayout(stats)

    def _create_plot(self, title: str, unit: str) -> pg.PlotWidget:
        plot = pg.PlotWidget(self)
        plot.setObjectName(f"imu{title}Plot")
        plot.setMinimumHeight(150)
        plot.setBackground(None)
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()
        plot.showGrid(x=True, y=True, alpha=0.18)
        plot.setLabel("left", unit)
        plot.setLabel("bottom", "时间", units="s")
        plot.addLegend(offset=(8, 6), colCount=3)
        plot.getAxis("left").setTextPen("#667085")
        plot.getAxis("bottom").setTextPen("#667085")
        plot.getAxis("left").setPen("#d0d5dd")
        plot.getAxis("bottom").setPen("#d0d5dd")
        plot.setTitle(title, color="#344054", size="9pt")
        return plot

    def append_samples(self, samples: Iterable[ImuChartSample]) -> int:
        appended = 0
        for sample in samples:
            sensor = _normalize_sensor_type(sample.sensor_type)
            if sensor not in self._samples:
                continue
            key = (sensor, sample.sequence_number)
            if key in self._last_keys:
                continue
            self._samples[sensor].append(sample)
            self._last_keys.add(key)
            appended += 1
        if len(self._last_keys) > self._maximum_samples * 3:
            self._last_keys = {
                (sensor, sample.sequence_number)
                for sensor, values in self._samples.items()
                for sample in values
            }
        if appended:
            self._refresh_curves()
        return appended

    def set_stats(self, stats: ImuMonitorStats) -> None:
        self.rate_badge.setText(f"{stats.sample_rate_hz:.1f} Hz")
        self.latency_badge.setText(
            "延迟 --"
            if stats.latest_latency_ms is None
            else f"延迟 {stats.latest_latency_ms:.1f} ms"
        )
        self.sequence_label.setText(
            f"丢序 {stats.sequence_gaps}  ·  重复 {stats.duplicate_samples}  ·  "
            f"乱序 {stats.out_of_order_samples}"
        )
        self.overflow_label.setText(f"队列溢出 {stats.queue_overflows}")
        self.window_label.setText(
            f"录制时间窗 {_clock_ns(stats.recording_window_ns)}"
        )
        self.csv_label.setText(_csv_state_text(stats.csv_state))

    def clear_samples(self) -> None:
        for values in self._samples.values():
            values.clear()
        self._last_keys.clear()
        self._refresh_curves()

    def sample_count(self, sensor_type: str) -> int:
        return len(self._samples[_normalize_sensor_type(sensor_type)])

    def _refresh_curves(self) -> None:
        for sensor, samples in self._samples.items():
            if not samples:
                for curve in self._curves[sensor]:
                    curve.setData([], [])
                continue
            origin_ns = samples[0].recording_time_ns
            x = [(sample.recording_time_ns - origin_ns) / 1_000_000_000 for sample in samples]
            for axis, curve in enumerate(self._curves[sensor]):
                curve.setData(x, [sample.values[axis] for sample in samples])


def _normalize_sensor_type(value: str) -> str:
    normalized = str(value).rsplit(".", 1)[-1].lower()
    aliases = {"accel": "accelerometer", "gyro": "gyroscope"}
    return aliases.get(normalized, normalized)


def _clock_ns(duration_ns: int) -> str:
    total_ms = max(0, duration_ns) // 1_000_000
    minutes, remainder = divmod(total_ms, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _csv_state_text(state: str) -> str:
    return {
        "idle": "CSV 等待写入",
        "writing": "CSV 正在写入",
        "finalizing": "CSV 正在校验",
        "complete": "CSV 已完成",
        "error": "CSV 写入失败",
    }.get(str(state).lower(), f"CSV {state}")
