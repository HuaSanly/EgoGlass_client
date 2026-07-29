from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import av


@dataclass(frozen=True)
class Mp4Inspection:
    path: str
    video_codec: str
    width: int
    height: int
    average_fps: float
    presentation_span_seconds: float
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
            if dimensions != (1280, 720):
                raise ValueError(
                    f"expected 1280x720 video, found {dimensions[0]}x{dimensions[1]}"
                )
            presentation_times: list[Fraction] = []
            for frame in container.decode(stream):
                if frame.pts is None or frame.time_base is None:
                    raise ValueError("recording frame is missing exact PTS")
                presentation_times.append(Fraction(frame.pts) * frame.time_base)
            decoded_frames = len(presentation_times)
            if not presentation_times:
                raise ValueError("recording contains no decodable video frames")
            if any(
                current <= previous
                for previous, current in zip(
                    presentation_times,
                    presentation_times[1:],
                    strict=False,
                )
            ):
                raise ValueError("recording frame PTS must strictly increase")
            presentation_span = presentation_times[-1] - presentation_times[0]
            if decoded_frames == 1:
                rate = stream.average_rate or stream.base_rate
                if rate is None or rate <= 0:
                    raise ValueError("recording has no measurable frame rate")
                average_fps = float(rate)
            else:
                if presentation_span <= 0:
                    raise ValueError("recording has no presentation time span")
                average_fps = float(Fraction(decoded_frames - 1) / presentation_span)
    except av.FFmpegError as error:
        raise ValueError(f"recording is not a finalized playable MP4: {error}") from error
    return Mp4Inspection(
        path=str(resolved),
        video_codec=codec_name,
        width=dimensions[0],
        height=dimensions[1],
        average_fps=round(average_fps, 3),
        presentation_span_seconds=round(float(presentation_span), 6),
        decoded_frames=decoded_frames,
    )
