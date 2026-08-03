from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from pathlib import Path

import av
from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QImage, QPainter


@dataclass(frozen=True, slots=True)
class ThumbnailResult:
    session_id: str
    clip_id: str
    image: QImage | None
    error: str | None = None


class VideoThumbnailService:
    """Decode bounded first-frame thumbnails away from the Qt main thread."""

    def __init__(self, *, workers: int = 2) -> None:
        if workers < 1:
            raise ValueError("thumbnail worker count must be positive")
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="video-thumbnail",
        )
        self._futures: dict[
            tuple[str, str], concurrent.futures.Future[ThumbnailResult]
        ] = {}
        self._completed: set[tuple[str, str]] = set()
        self._closed = False

    def request(self, session_id: str, clip_id: str, session_directory: Path) -> None:
        key = (session_id, clip_id)
        if self._closed or key in self._futures or key in self._completed:
            return
        self._futures[key] = self._executor.submit(
            _decode_thumbnail,
            session_id,
            clip_id,
            session_directory,
        )

    def take_completed(self) -> tuple[ThumbnailResult, ...]:
        completed: list[ThumbnailResult] = []
        for key, future in tuple(self._futures.items()):
            if not future.done():
                continue
            del self._futures[key]
            self._completed.add(key)
            try:
                completed.append(future.result())
            except Exception as error:
                completed.append(ThumbnailResult(*key, image=None, error=str(error)))
        return tuple(completed)

    def invalidate(self) -> None:
        self._completed.clear()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for future in self._futures.values():
            future.cancel()
        self._futures.clear()
        self._executor.shutdown(wait=True, cancel_futures=True)


def _decode_thumbnail(
    session_id: str,
    clip_id: str,
    session_directory: Path,
) -> ThumbnailResult:
    path = _clip_media_path(session_directory, clip_id)
    if path is None:
        return ThumbnailResult(session_id, clip_id, None, "找不到视频文件")
    try:
        with av.open(str(path), mode="r") as container:
            stream = container.streams.video[0]
            frame = next(container.decode(stream), None)
            if frame is None:
                raise ValueError("视频没有可解码帧")
            source = frame.to_image()
            source_image = QImage(
                source.tobytes("raw", "RGB"),
                source.width,
                source.height,
                source.width * 3,
                QImage.Format.Format_RGB888,
            ).copy()
        return ThumbnailResult(
            session_id,
            clip_id,
            _letterbox_4_by_3(source_image, 320, 240),
        )
    except Exception as error:
        return ThumbnailResult(session_id, clip_id, None, str(error))


def _clip_media_path(session_directory: Path, clip_id: str) -> Path | None:
    for path in (
        session_directory / "media" / f"{clip_id}.mp4",
        session_directory / f"{clip_id}.mp4",
    ):
        if path.is_file():
            return path
    return None


def _letterbox_4_by_3(image: QImage, width: int, height: int) -> QImage:
    output = QImage(width, height, QImage.Format.Format_RGB888)
    output.fill(QColor("#11161d"))
    scaled = image.scaled(
        width,
        height,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    painter = QPainter(output)
    painter.drawImage(
        QRect((width - scaled.width()) // 2, (height - scaled.height()) // 2,
              scaled.width(), scaled.height()),
        scaled,
    )
    painter.end()
    return output
