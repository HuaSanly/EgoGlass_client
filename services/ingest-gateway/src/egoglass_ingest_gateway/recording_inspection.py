from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import av


@dataclass(frozen=True)
class Mp4Inspection:
    path: str
    video_codec: str
    width: int
    height: int
    nominal_fps: float
    decoded_frames: int

    def as_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def inspect_recording(path: Path) -> Mp4Inspection:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.suffix.casefold() != ".mp4":
        raise ValueError("recording must use the .mp4 extension")
    try:
        with av.open(str(resolved), mode="r") as container:
            video_streams = list(container.streams.video)
            if len(video_streams) != 1:
                raise ValueError("recording must contain exactly one video stream")
            stream = video_streams[0]
            codec_name = stream.codec_context.name
            if codec_name != "h264":
                raise ValueError(f"expected H.264 video, found {codec_name or 'unknown'}")
            dimensions = (stream.codec_context.width, stream.codec_context.height)
            if dimensions != (1920, 1080):
                raise ValueError(
                    f"expected 1920x1080 video, found {dimensions[0]}x{dimensions[1]}"
                )
            rate = stream.average_rate or stream.base_rate
            if rate is None:
                raise ValueError("recording has no nominal frame rate")
            nominal_fps = float(rate)
            if abs(nominal_fps - 30.0) > 0.05:
                raise ValueError(f"expected nominal 30 FPS, found {nominal_fps:.3f}")
            decoded_frames = sum(1 for _frame in container.decode(stream))
            if decoded_frames == 0:
                raise ValueError("recording contains no decodable video frames")
    except av.FFmpegError as error:
        raise ValueError(f"recording is not a finalized playable MP4: {error}") from error
    return Mp4Inspection(
        path=str(resolved),
        video_codec=codec_name,
        width=dimensions[0],
        height=dimensions[1],
        nominal_fps=round(nominal_fps, 3),
        decoded_frames=decoded_frames,
    )
