from __future__ import annotations

import asyncio
from fractions import Fraction
from pathlib import Path

import av
from aiortc import MediaStreamError

from egoglass_ingest_gateway.adapters.webrtc import WebRtcVideoRecordingSource
from egoglass_ingest_gateway.recording import RecordingRuntime
from egoglass_ingest_gateway.recording_inspection import inspect_recording

SESSION_ID = "e" * 32


class SyntheticFullHdTrack:
    kind = "video"

    def __init__(self) -> None:
        self._pts = [90_000, 180_000, 0]
        self._index = 0

    async def recv(self) -> av.VideoFrame:
        if self._index == len(self._pts):
            raise MediaStreamError
        frame = av.VideoFrame(1920, 1080, "yuv420p")
        frame.pts = self._pts[self._index]
        frame.time_base = Fraction(1, 90_000)
        self._index += 1
        return frame


class SyntheticSource:
    def subscribe(self, *, buffered: bool) -> SyntheticFullHdTrack:
        assert buffered is True
        return SyntheticFullHdTrack()


async def no_countdown_delay(seconds: float) -> None:
    assert seconds == 3.0


def test_real_pyav_path_publishes_only_playable_full_hd_h264_mp4(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        runtime = RecordingRuntime(
            tmp_path,
            lambda: asyncio.sleep(
                0,
                result=WebRtcVideoRecordingSource(
                    SESSION_ID,
                    SyntheticSource(),
                    1920,
                    1080,
                ),
            ),
            sleep=no_countdown_delay,
        )
        await runtime.start()
        for _attempt in range(100):
            library = await runtime.library()
            if library.sessions:
                break
            await asyncio.sleep(0.01)
        else:
            status = await runtime.status()
            raise AssertionError(
                f"synthetic recording was not finalized: {status.model_dump()}"
            )

        clip = library.sessions[0].clips[0]
        path = await runtime.media_path(SESSION_ID, clip.clip_id)
        assert path is not None
        inspection = inspect_recording(path)
        assert inspection.video_codec == "h264"
        assert (inspection.width, inspection.height) == (1920, 1080)
        assert inspection.nominal_fps == 30.0
        assert inspection.decoded_frames == 3
        assert not list(tmp_path.rglob("*.part.mp4"))
        renamed_library = await runtime.rename_session(SESSION_ID, "Eval capture")
        assert renamed_library.sessions[0].display_name == "Eval capture"
        assert path.is_file()
        deleted_library = await runtime.delete_clip(SESSION_ID, clip.clip_id)
        assert deleted_library.sessions == []
        assert not path.exists()
        assert not list(tmp_path.rglob("*"))
        await runtime.close()

    asyncio.run(scenario())
