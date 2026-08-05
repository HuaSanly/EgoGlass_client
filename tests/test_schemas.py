from __future__ import annotations

import numpy as np
import pytest

from schemas import FramePacket, ImuPacket, ImuSensor, PlaybackFrame
from ui.gateway.live_frames import LiveFrame
from ui.gateway.webrtc_models import ImuSample, ImuSensorType
from ui.presentation import render_hand_tracking_overlay


def _rgb() -> np.ndarray:
    image = np.zeros((4, 3, 3), dtype=np.uint8)
    image.setflags(write=False)
    return image


def test_frame_packet_requires_immutable_rgb() -> None:
    packet = FramePacket(
        session_id="session",
        stream_id="camera",
        frame_index=1,
        captured_at_ns=None,
        received_at_ns=2,
        pts_ns=3,
        image_rgb=_rgb(),
    )

    assert (packet.width, packet.height) == (3, 4)

    with pytest.raises(ValueError, match="read-only"):
        FramePacket(
            session_id="session",
            stream_id="camera",
            frame_index=1,
            captured_at_ns=None,
            received_at_ns=2,
            pts_ns=3,
            image_rgb=np.zeros((4, 3, 3), dtype=np.uint8),
        )


def test_imu_packet_and_playback_frame_have_stable_identity() -> None:
    imu = ImuPacket("session", ImuSensor.GYROSCOPE, 4, 5, 6, (1.0, 2.0, 3.0))
    frame = PlaybackFrame("session", "clip", 4, 5, 6, _rgb())

    assert imu.sensor is ImuSensor.GYROSCOPE
    assert frame.frame_index == imu.sequence_number


def test_presentation_adapter_uses_public_hand_schema() -> None:
    assert callable(render_hand_tracking_overlay)


def test_gateway_adapters_convert_to_public_frame_and_imu_schemas() -> None:
    image = _rgb()
    frame_packet = LiveFrame(
        "session",
        "stream",
        7,
        8,
        9,
        image,
        10,
    ).to_frame_packet()
    imu_packet = ImuSample(
        sensor_type=ImuSensorType.GYROSCOPE,
        android_sensor_type=4,
        sequence_number=11,
        sensor_event_monotonic_ns=12,
        received_at_elapsed_realtime_ns=13,
        accuracy=3,
        values=(1.0, 2.0, 3.0),
    ).to_packet("session")

    assert isinstance(frame_packet, FramePacket)
    assert frame_packet.pts_ns == 10
    assert isinstance(imu_packet, ImuPacket)
    assert imu_packet.sensor is ImuSensor.GYROSCOPE
