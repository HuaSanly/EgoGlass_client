"""Canonical IMU sample exchanged with preprocessing services."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ImuSensor(StrEnum):
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"


@dataclass(frozen=True, slots=True)
class ImuPacket:
    """One sensor sample with both device and client timing."""

    session_id: str
    sensor: ImuSensor
    sequence_number: int
    sensor_event_ns: int
    received_at_ns: int
    values: tuple[float, float, float]
    accuracy: int = -1

    def __post_init__(self) -> None:
        if self.sequence_number < 0 or self.sensor_event_ns < 0 or self.received_at_ns < 0:
            raise ValueError("IMU sequence and timestamps must be non-negative")
        if len(self.values) != 3:
            raise ValueError("IMU values must contain exactly three components")
