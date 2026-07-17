from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RecordingState(StrEnum):
    UNAVAILABLE = "unavailable"
    READY = "ready"
    COUNTDOWN = "countdown"
    RECORDING = "recording"
    FINALIZING = "finalizing"
    ERROR = "error"


class RecordingOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    width: Literal[1920] = 1920
    height: Literal[1080] = 1080
    fps: Literal[30] = 30
    container: Literal["mp4"] = "mp4"
    video_codec: Literal["h264"] = "h264"


class RecordingCommandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["start", "stop"]


class RecordingStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    state: RecordingState
    detail: str = Field(default="", max_length=256)
    session_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    clip_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    countdown_started_at_unix_ms: int | None = Field(default=None, ge=0)
    recording_starts_at_unix_ms: int | None = Field(default=None, ge=0)
    recording_started_at_unix_ms: int | None = Field(default=None, ge=0)
    recording_duration_ms: int = Field(default=0, ge=0)
    output: RecordingOutput = Field(default_factory=RecordingOutput)


class RecordingClip(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    recorded_at_unix_ms: int = Field(ge=0)
    ended_at_unix_ms: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    width: Literal[1920] = 1920
    height: Literal[1080] = 1080
    fps: Literal[30] = 30
    file_size_bytes: int = Field(gt=0)
    media_url: str = Field(pattern=r"^/api/v1/recordings/media/[0-9a-f]{32}/[0-9a-f]{32}$")


class RecordingSession(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    started_at_unix_ms: int = Field(ge=0)
    clips: list[RecordingClip]


class RecordingLibrary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    sessions: list[RecordingSession]
