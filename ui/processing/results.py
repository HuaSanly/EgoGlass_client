from __future__ import annotations

import json
import sqlite3
from pathlib import Path

_SCHEMA_VERSION = "1"


class ProcessingResultStore:
    """Frame-addressable structured results for processed-video playback."""

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path).expanduser().resolve()
        self.read_only = read_only
        if read_only:
            if not self.path.is_file():
                raise FileNotFoundError(self.path)
            self._validate()
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize()

    def put(self, result: dict[str, object]) -> None:
        if self.read_only:
            raise RuntimeError("result store is read-only")
        clip_id = _require_string(result, "sequence_id")
        frame_index = _require_integer(result, "frame_index")
        session_time_ns = _require_integer(result, "session_time_ns")
        payload = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO frame_results(clip_id, frame_index, session_time_ns, result_json)
                VALUES (?, ?, ?, ?)
                """,
                (clip_id, frame_index, session_time_ns, payload),
            )

    def result_for_frame(
        self,
        clip_id: str,
        frame_index: int,
        session_time_ns: int,
        *,
        hold_previous_frames: int = 0,
    ) -> dict[str, object] | None:
        if hold_previous_frames < 0:
            raise ValueError("hold_previous_frames cannot be negative")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT frame_index, session_time_ns, result_json
                FROM frame_results
                WHERE clip_id = ? AND frame_index <= ? AND session_time_ns <= ?
                ORDER BY frame_index DESC, session_time_ns DESC LIMIT 1
                """,
                (clip_id, frame_index, session_time_ns),
            ).fetchone()
        if row is None or frame_index - row["frame_index"] > hold_previous_frames:
            return None
        return json.loads(row["result_json"])

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM frame_results").fetchone()[0])

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS frame_results (
                    clip_id TEXT NOT NULL,
                    frame_index INTEGER NOT NULL CHECK(frame_index >= 0),
                    session_time_ns INTEGER NOT NULL CHECK(session_time_ns >= 0),
                    result_json TEXT NOT NULL,
                    PRIMARY KEY(clip_id, frame_index, session_time_ns)
                );
                CREATE INDEX IF NOT EXISTS frame_results_time
                    ON frame_results(clip_id, session_time_ns);
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES ('schema_version', ?)",
                (_SCHEMA_VERSION,),
            )

    def _validate(self) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
        if row is None or row["value"] != _SCHEMA_VERSION:
            raise RuntimeError("unsupported processing result schema")

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            connection = sqlite3.connect(
                f"{self.path.as_uri()}?mode=ro", uri=True, timeout=5.0
            )
            connection.execute("PRAGMA query_only = ON")
        else:
            connection = sqlite3.connect(self.path, timeout=5.0)
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
        connection.row_factory = sqlite3.Row
        return connection


def _require_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"result {key} must be a non-empty string")
    return value


def _require_integer(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"result {key} must be a non-negative integer")
    return value
