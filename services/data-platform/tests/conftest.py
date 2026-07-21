from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

SESSION_ID = "1" * 32
CLIP_ID = "2" * 32


@pytest.fixture
def recordings_root(tmp_path: Path) -> Path:
    create_capture_session(tmp_path)
    return tmp_path


def create_capture_session(
    root: Path,
    *,
    session_id: str = SESSION_ID,
    clip_id: str = CLIP_ID,
    state: str = "complete",
    frame_count: int = 300,
    fps: float = 30.0,
    with_frame_index: bool = True,
) -> Path:
    session_directory = root / session_id
    media_directory = session_directory / "media"
    telemetry_directory = session_directory / "telemetry"
    (session_directory / "annotations").mkdir(parents=True, exist_ok=True)
    (session_directory / "derived").mkdir(parents=True, exist_ok=True)
    media_directory.mkdir(parents=True, exist_ok=True)
    telemetry_directory.mkdir(parents=True, exist_ok=True)
    media_path = media_directory / f"{clip_id}.mp4"
    media_path.write_bytes(b"synthetic-mp4-fixture")
    manifest = {
        "schema_version": "1.0",
        "contract_id": "capture-session-v1",
        "session_id": session_id,
        "display_name": "2026-07-21 10-00-00",
        "display_name_source": "timestamp_default",
        "lifecycle": {
            "state": state,
            "start_reason": "first_recording_request",
            "started_at_unix_ns": 1_784_600_000_000_000_000,
            "ended_at_unix_ns": (
                None if state in {"active", "finalizing"} else 1_784_600_010_000_000_000
            ),
            "end_reason": None if state in {"active", "finalizing"} else "client_shutdown",
        },
        "session_time_origin": {
            "status": "established",
            "clock_id": "glasses_elapsed_realtime_ns",
            "origin_elapsed_realtime_ns": 1_000_000,
            "origin_event": "first_imu_sample",
        },
        "imu_capture_policy": "continuous_while_session_active",
        "provenance": {},
        "storage": {},
        "clips": [
            {
                "clip_id": clip_id,
                "state": "complete",
                "relative_media_path": f"media/{clip_id}.mp4",
                "requested_at_session_time_ns": 0,
                "started_at_session_time_ns": 1_000_000_000,
                "ended_at_session_time_ns": 11_000_000_000,
                "video_profile": {
                    "container": "mp4",
                    "codec": "h264",
                    "width": 1280,
                    "height": 720,
                    "nominal_fps": fps,
                },
                "frame_count": frame_count,
                "sha256": hashlib.sha256(media_path.read_bytes()).hexdigest(),
            }
        ],
    }
    (session_directory / "session.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    if with_frame_index:
        with sqlite3.connect(telemetry_directory / "telemetry.sqlite") as database:
            database.execute(
                """
                CREATE TABLE video_frame_index (
                    clip_id TEXT NOT NULL,
                    frame_index INTEGER NOT NULL,
                    mp4_pts INTEGER NOT NULL,
                    mp4_time_base_numerator INTEGER NOT NULL,
                    mp4_time_base_denominator INTEGER NOT NULL,
                    session_time_ns INTEGER
                )
                """
            )
            database.executemany(
                "INSERT INTO video_frame_index VALUES (?, ?, ?, 1, 90000, ?)",
                (
                    (clip_id, frame, frame * 3000, 1_000_000_000 + frame * 33_333_333)
                    for frame in range(frame_count)
                ),
            )
    return session_directory


def complete_episode(*, episode_id: str = "3" * 32) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "clip_id": CLIP_ID,
        "start_frame_index": 30,
        "end_frame_index_exclusive": 240,
        "source_strategy": "manual",
        "labels": {
            "task_id": "place-cup",
            "instruction": "拿起杯子并放到托盘中",
            "verb": "放置",
            "object": "杯子",
            "target": "托盘",
            "hand": "right",
            "outcome": "success",
            "quality_flags": [],
            "notes": "",
        },
        "phases": [
            {
                "phase_id": "4" * 32,
                "start_frame_index": 30,
                "end_frame_index_exclusive": 90,
                "phase": "approach",
                "action_verb": "接近",
                "active_hand": "right",
                "object": "杯子",
            },
            {
                "phase_id": "5" * 32,
                "start_frame_index": 90,
                "end_frame_index_exclusive": 240,
                "phase": "manipulate",
                "action_verb": "拿起并放置",
                "active_hand": "right",
                "object": "杯子",
            },
        ],
    }
