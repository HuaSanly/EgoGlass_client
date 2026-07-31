from __future__ import annotations

import asyncio
import json
from fractions import Fraction
from pathlib import Path

import av
from aiortc import MediaStreamError

from ingest_gateway.adapters.webrtc import WebRtcVideoRecordingSource
from ingest_gateway.recording import RecordingRuntime
from ingest_gateway.recording_inspection import inspect_recording

SESSION_ID = "e" * 32


class SyntheticFourByThreeTrack:
    kind = "video"

    def __init__(self) -> None:
        self._pts = [90_000, 180_000, 0]
        self._index = 0

    async def recv(self) -> av.VideoFrame:
        if self._index == len(self._pts):
            raise MediaStreamError
        frame = av.VideoFrame(640, 480, "yuv420p")
        frame.pts = self._pts[self._index]
        frame.time_base = Fraction(1, 90_000)
        self._index += 1
        return frame


class SyntheticSource:
    def subscribe(self, *, buffered: bool) -> SyntheticFourByThreeTrack:
        assert buffered is True
        return SyntheticFourByThreeTrack()


async def no_countdown_delay(seconds: float) -> None:
    assert seconds == 3.0


def test_real_pyav_path_publishes_playable_four_by_three_h264_mp4(
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
                    640,
                    480,
                    1,
                ),
            ),
            sleep=no_countdown_delay,
            session_id_factory=lambda: SESSION_ID,
        )
        await runtime.start()
        for _attempt in range(100):
            library = await runtime.library()
            if library.sessions and library.sessions[0].clips:
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
        assert (inspection.width, inspection.height) == (640, 480)
        assert inspection.average_fps > 0
        assert inspection.presentation_span_seconds > 0
        assert inspection.decoded_frames == 3
        assert not list(tmp_path.rglob("*.part.mp4"))
        renamed_library = await runtime.rename_session(SESSION_ID, "Eval capture")
        assert renamed_library.sessions[0].display_name == "Eval capture"
        assert path.is_file()
        await runtime.session_command("new")
        deleted_library = await runtime.delete_clip(SESSION_ID, clip.clip_id)
        assert deleted_library.sessions[0].clips == []
        assert not path.exists()
        assert (tmp_path / SESSION_ID / "telemetry" / "telemetry.sqlite").is_file()
        assert (await runtime.delete_session(SESSION_ID)).sessions == []
        assert not (tmp_path / SESSION_ID).exists()
        await runtime.close()

    asyncio.run(scenario())


def test_legacy_full_hd_manifest_remains_visible_after_hd_profile_change(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        session_id = "a" * 32
        clip_id = "b" * 32
        session_directory = tmp_path / session_id
        session_directory.mkdir()
        media_path = session_directory / f"{clip_id}.mp4"
        media_path.write_bytes(b"legacy-recording")
        (session_directory / "session.json").write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "started_at_unix_ms": 1,
                    "display_name": "Legacy 1080p",
                    "clips": [
                        {
                            "clip_id": clip_id,
                            "recorded_at_unix_ms": 1,
                            "ended_at_unix_ms": 2,
                            "duration_ms": 1,
                            "width": 1920,
                            "height": 1080,
                            "fps": 30,
                            "file_size_bytes": media_path.stat().st_size,
                            "media_url": (
                                f"/api/v1/recordings/media/{session_id}/{clip_id}"
                            ),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        runtime = RecordingRuntime(
            tmp_path,
            lambda: asyncio.sleep(0, result=None),
        )

        library = await runtime.library()
        assert library.sessions[0].display_name == "Legacy 1080p"
        clip = library.sessions[0].clips[0]
        assert (clip.width, clip.height) == (1920, 1080)
        assert await runtime.media_path(session_id, clip_id) == media_path

    asyncio.run(scenario())
