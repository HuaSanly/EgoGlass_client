from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from schemas.recording import (
    CameraFrameRow,
    ImuSensorType,
    RecordingImuRow,
    RecordingOutput,
)
from ui.gateway.capture_recording import (
    CaptureRecordingReader,
    CaptureRecordingWriter,
    StagedCameraFrame,
    StagedImuSample,
)


def write_h264_video(
    path: Path,
    *,
    width: int = 32,
    height: int = 24,
    frame_count: int = 2,
    fps: int = 10,
) -> tuple[tuple[int, int, int], ...]:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "tune": "zerolatency"}
        for index in range(frame_count):
            pixels = np.full((height, width, 3), (index * 40) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    with av.open(str(path), mode="r") as container:
        decoded = tuple(container.decode(container.streams.video[0]))
    assert len(decoded) == frame_count
    return tuple(
        (frame.pts, frame.time_base.numerator, frame.time_base.denominator)
        for frame in decoded
        if frame.pts is not None and frame.time_base is not None
    )


def staged_camera_frames(
    video_index: Sequence[tuple[int, int, int]],
    *,
    first_device_ns: int = 100_000_000,
    frame_period_ns: int = 100_000_000,
) -> tuple[StagedCameraFrame, ...]:
    return tuple(
        StagedCameraFrame(
            row=CameraFrameRow(
                frame_idx=index,
                frame_id=1_000 + index,
                rokid_timestamp_ns=(5_000 + index * 100) * 1_000_000,
                device_monotonic_ns=first_device_ns + index * frame_period_ns,
            ),
            mp4_pts=pts,
            mp4_time_base_num=time_base_num,
            mp4_time_base_den=time_base_den,
            received_at_client_monotonic_ns=1_000_000_000 + index * frame_period_ns,
        )
        for index, (pts, time_base_num, time_base_den) in enumerate(video_index)
    )


def append_covering_imu(
    writer: CaptureRecordingWriter,
    camera_frames: Sequence[StagedCameraFrame],
) -> None:
    first_ns = camera_frames[0].row.device_monotonic_ns - 1
    last_ns = camera_frames[-1].row.device_monotonic_ns + 1
    for sensor in (ImuSensorType.ACCELEROMETER, ImuSensorType.GYROSCOPE):
        for sequence, timestamp_ns in enumerate((first_ns, last_ns)):
            writer.append_imu(
                StagedImuSample(
                    row=RecordingImuRow(
                        sensor_type=sensor,
                        sequence=sequence,
                        timestamp_ns=timestamp_ns,
                        x=float(sequence),
                        y=float(sequence + 1),
                        z=float(sequence + 2),
                    ),
                    received_at_client_monotonic_ns=(
                        camera_frames[0].received_at_client_monotonic_ns
                        if sequence == 0
                        else camera_frames[-1].received_at_client_monotonic_ns
                    ),
                )
            )


def create_recording(
    root: Path,
    *,
    recording_id: str = "a" * 32,
    width: int = 32,
    height: int = 24,
    frame_count: int = 2,
) -> CaptureRecordingReader:
    writer = CaptureRecordingWriter.create(
        root,
        recording_id=recording_id,
        video_profile=RecordingOutput(width=width, height=height, fps=10.0),
    )
    video_index = write_h264_video(
        writer.video_path,
        width=width,
        height=height,
        frame_count=frame_count,
    )
    frames = staged_camera_frames(video_index)
    append_covering_imu(writer, frames)
    return writer.finalize(frames)
