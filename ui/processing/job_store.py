from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

from .models import (
    ACTIVE_JOB_STATES,
    TERMINAL_JOB_STATES,
    ProcessingJob,
    ProcessingJobState,
    ProcessingPreset,
)

_SCHEMA_VERSION = "5"
_MISTAGGED_SCHEMA_VERSION = "6"
_JOB_COLUMNS = frozenset(
    {
        "job_id",
        "session_id",
        "clip_id",
        "preset_json",
        "state",
        "created_at_unix_ns",
        "updated_at_unix_ns",
        "progress_current",
        "progress_total",
        "detail",
        "run_id",
        "retry_of_job_id",
        "started_at_unix_ns",
        "finished_at_unix_ns",
        "configuration_revision",
        "configuration_sha256_json",
        "configuration_snapshot_json",
        "task_profile_id",
        "task_profile_snapshot_json",
    }
)


class ProcessingJobStore:
    """Persistent, process-local queue with explicit crash recovery."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self.recover_interrupted()

    def enqueue(
        self,
        session_id: str,
        clip_id: str | None,
        preset: ProcessingPreset,
        *,
        retry_of_job_id: str | None = None,
        configuration_revision: int = 0,
        configuration_sha256_by_file: tuple[tuple[str, str], ...] = (),
        configuration_snapshot_json: str = "{}",
        task_profile_id: str | None = None,
        task_profile_snapshot_json: str = "{}",
    ) -> ProcessingJob:
        _validate_identifier(session_id, "session_id")
        if clip_id is not None:
            _validate_identifier(clip_id, "clip_id")
        if task_profile_id is not None:
            _validate_identifier(task_profile_id, "task_profile_id")
        _validate_configuration_snapshot(configuration_snapshot_json)
        _validate_configuration_snapshot(task_profile_snapshot_json)
        now_ns = time.time_ns()
        job_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, session_id, clip_id, preset_json, state,
                    created_at_unix_ns, updated_at_unix_ns, detail, retry_of_job_id,
                    configuration_revision, configuration_sha256_json
                    , configuration_snapshot_json, task_profile_id,
                    task_profile_snapshot_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    session_id,
                    clip_id,
                    _preset_json(preset),
                    ProcessingJobState.QUEUED.value,
                    now_ns,
                    now_ns,
                    "等待处理",
                    retry_of_job_id,
                    configuration_revision,
                    json.dumps(
                        dict(configuration_sha256_by_file),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    configuration_snapshot_json,
                    task_profile_id,
                    task_profile_snapshot_json,
                ),
            )
        return self.require(job_id)

    def claim_next(self) -> ProcessingJob | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT job_id FROM jobs WHERE state = ? ORDER BY created_at_unix_ns LIMIT 1",
                (ProcessingJobState.QUEUED.value,),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            now_ns = time.time_ns()
            connection.execute(
                """
                UPDATE jobs SET state = ?, updated_at_unix_ns = ?,
                    started_at_unix_ns = ?, detail = ?
                WHERE job_id = ? AND state = ?
                """,
                (
                    ProcessingJobState.PREPARING.value,
                    now_ns,
                    now_ns,
                    "正在校验会话",
                    row["job_id"],
                    ProcessingJobState.QUEUED.value,
                ),
            )
            connection.commit()
            return self.require(str(row["job_id"]))

    def start_run(self, job_id: str, run_id: str, total: int) -> ProcessingJob:
        return self._transition(
            job_id,
            {ProcessingJobState.PREPARING},
            ProcessingJobState.RUNNING,
            run_id=run_id,
            progress_current=0,
            progress_total=max(0, total),
            detail="正在处理",
        )

    def update_progress(self, job_id: str, current: int, total: int, detail: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET progress_current = ?, progress_total = ?, detail = ?, updated_at_unix_ns = ?
                WHERE job_id = ? AND state IN (?, ?)
                """,
                (
                    max(0, current),
                    max(0, total),
                    detail,
                    time.time_ns(),
                    job_id,
                    ProcessingJobState.RUNNING.value,
                    ProcessingJobState.CANCELING.value,
                ),
            )

    def complete(self, job_id: str, detail: str = "处理完成") -> ProcessingJob:
        job = self.require(job_id)
        return self._transition(
            job_id,
            {ProcessingJobState.RUNNING, ProcessingJobState.CANCELING},
            ProcessingJobState.COMPLETED,
            progress_current=job.progress_total,
            detail=detail,
        )

    def partial(self, job_id: str, detail: str = "部分完成，世界坐标不完整") -> ProcessingJob:
        job = self.require(job_id)
        return self._transition(
            job_id,
            {ProcessingJobState.RUNNING, ProcessingJobState.CANCELING},
            ProcessingJobState.PARTIAL,
            progress_current=job.progress_total,
            detail=detail,
        )

    def fail(self, job_id: str, detail: str) -> ProcessingJob:
        return self._transition(
            job_id,
            {ProcessingJobState.PREPARING, ProcessingJobState.RUNNING},
            ProcessingJobState.FAILED,
            detail=detail,
        )

    def request_cancel(self, job_id: str) -> ProcessingJob:
        job = self.require(job_id)
        if job.state is ProcessingJobState.QUEUED:
            return self._transition(
                job_id,
                {ProcessingJobState.QUEUED},
                ProcessingJobState.CANCELED,
                detail="已取消",
            )
        if job.state in {ProcessingJobState.PREPARING, ProcessingJobState.RUNNING}:
            return self._transition(
                job_id,
                {job.state},
                ProcessingJobState.CANCELING,
                detail="正在取消",
            )
        return job

    def mark_canceled(self, job_id: str) -> ProcessingJob:
        return self._transition(
            job_id,
            {ProcessingJobState.CANCELING},
            ProcessingJobState.CANCELED,
            detail="已取消",
        )

    def retry(
        self,
        job_id: str,
        *,
        configuration_revision: int = 0,
        configuration_sha256_by_file: tuple[tuple[str, str], ...] = (),
        configuration_snapshot_json: str = "{}",
        task_profile_snapshot_json: str | None = None,
    ) -> ProcessingJob:
        job = self.require(job_id)
        if job.state not in {
            ProcessingJobState.FAILED,
            ProcessingJobState.INTERRUPTED,
            ProcessingJobState.CANCELED,
            ProcessingJobState.PARTIAL,
        }:
            raise ValueError("only partial, failed, interrupted, or canceled jobs can be retried")
        return self.enqueue(
            job.session_id,
            job.clip_id,
            job.preset,
            retry_of_job_id=job.job_id,
            configuration_revision=configuration_revision,
            configuration_sha256_by_file=configuration_sha256_by_file,
            configuration_snapshot_json=configuration_snapshot_json,
            task_profile_id=job.task_profile_id,
            task_profile_snapshot_json=(
                job.task_profile_snapshot_json
                if task_profile_snapshot_json is None
                else task_profile_snapshot_json
            ),
        )

    def recover_interrupted(self) -> int:
        values = tuple(state.value for state in ACTIVE_JOB_STATES)
        placeholders = ",".join("?" for _ in values)
        now_ns = time.time_ns()
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE jobs SET state = ?, detail = ?, updated_at_unix_ns = ?,
                    finished_at_unix_ns = ?
                WHERE state IN ({placeholders})
                """,
                (
                    ProcessingJobState.INTERRUPTED.value,
                    "客户端退出导致任务中断，请手动重试",
                    now_ns,
                    now_ns,
                    *values,
                ),
            )
            return cursor.rowcount

    def require(self, job_id: str) -> ProcessingJob:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown processing job {job_id!r}")
        return _job_from_row(row)

    def list_jobs(self, limit: int = 200) -> tuple[ProcessingJob, ...]:
        if limit < 1:
            raise ValueError("job limit must be positive")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at_unix_ns DESC LIMIT ?", (limit,)
            ).fetchall()
        return tuple(_job_from_row(row) for row in rows)

    def setting(self, key: str, default: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = ?", (f"setting:{key}",)
            ).fetchone()
        return default if row is None else str(row["value"])

    def set_setting(self, key: str, value: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO metadata(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (f"setting:{key}", value),
            )

    def _transition(
        self,
        job_id: str,
        expected: set[ProcessingJobState],
        state: ProcessingJobState,
        **changes: object,
    ) -> ProcessingJob:
        now_ns = time.time_ns()
        assignments = ["state = ?", "updated_at_unix_ns = ?"]
        values: list[object] = [state.value, now_ns]
        if state in TERMINAL_JOB_STATES:
            assignments.append("finished_at_unix_ns = ?")
            values.append(now_ns)
        for key, value in changes.items():
            if key not in {
                "run_id",
                "progress_current",
                "progress_total",
                "detail",
                "started_at_unix_ns",
                "finished_at_unix_ns",
            }:
                raise ValueError(f"unsupported job field {key!r}")
            assignments.append(f"{key} = ?")
            values.append(value)
        expected_values = tuple(item.value for item in expected)
        placeholders = ",".join("?" for _ in expected_values)
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE jobs SET {', '.join(assignments)} WHERE job_id = ? "
                f"AND state IN ({placeholders})",
                (*values, job_id, *expected_values),
            )
            if cursor.rowcount != 1:
                current = self.require(job_id)
                raise RuntimeError(
                    f"cannot transition job {job_id} from {current.state.value} to {state.value}"
                )
        return self.require(job_id)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    clip_id TEXT,
                    preset_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at_unix_ns INTEGER NOT NULL,
                    updated_at_unix_ns INTEGER NOT NULL,
                    progress_current INTEGER NOT NULL DEFAULT 0,
                    progress_total INTEGER NOT NULL DEFAULT 0,
                    detail TEXT NOT NULL DEFAULT '',
                    run_id TEXT,
                    retry_of_job_id TEXT REFERENCES jobs(job_id),
                    started_at_unix_ns INTEGER,
                    finished_at_unix_ns INTEGER,
                    configuration_revision INTEGER NOT NULL DEFAULT 0,
                    configuration_sha256_json TEXT NOT NULL DEFAULT '{}',
                    configuration_snapshot_json TEXT NOT NULL DEFAULT '{}'
                    , task_profile_id TEXT,
                    task_profile_snapshot_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS jobs_state_created
                    ON jobs(state, created_at_unix_ns);
                """
            )
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES ('schema_version', ?)",
                    (_SCHEMA_VERSION,),
                )
            elif row["value"] in {"1", "2", "3", "4"}:
                columns = {
                    str(item["name"])
                    for item in connection.execute("PRAGMA table_info(jobs)").fetchall()
                }
                if "started_at_unix_ns" not in columns:
                    connection.execute("ALTER TABLE jobs ADD COLUMN started_at_unix_ns INTEGER")
                if "finished_at_unix_ns" not in columns:
                    connection.execute("ALTER TABLE jobs ADD COLUMN finished_at_unix_ns INTEGER")
                if "configuration_revision" not in columns:
                    connection.execute(
                        "ALTER TABLE jobs ADD COLUMN configuration_revision "
                        "INTEGER NOT NULL DEFAULT 0"
                    )
                if "configuration_sha256_json" not in columns:
                    connection.execute(
                        "ALTER TABLE jobs ADD COLUMN configuration_sha256_json "
                        "TEXT NOT NULL DEFAULT '{}'"
                    )
                if "configuration_snapshot_json" not in columns:
                    connection.execute(
                        "ALTER TABLE jobs ADD COLUMN configuration_snapshot_json "
                        "TEXT NOT NULL DEFAULT '{}'"
                    )
                if "task_profile_id" not in columns:
                    connection.execute("ALTER TABLE jobs ADD COLUMN task_profile_id TEXT")
                if "task_profile_snapshot_json" not in columns:
                    connection.execute(
                        "ALTER TABLE jobs ADD COLUMN task_profile_snapshot_json "
                        "TEXT NOT NULL DEFAULT '{}'"
                    )
                connection.execute(
                    "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
                    (_SCHEMA_VERSION,),
                )
            elif row["value"] == _MISTAGGED_SCHEMA_VERSION:
                columns = {
                    str(item["name"])
                    for item in connection.execute("PRAGMA table_info(jobs)").fetchall()
                }
                if columns != _JOB_COLUMNS:
                    raise RuntimeError("unsupported video-processing queue schema")
                # Version 6 was briefly written by a development build without
                # changing the v5 table. Normalize only that exact known shape.
                connection.execute(
                    "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
                    (_SCHEMA_VERSION,),
                )
            elif row["value"] != _SCHEMA_VERSION:
                raise RuntimeError("unsupported video-processing queue schema")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _preset_json(preset: ProcessingPreset) -> str:
    return json.dumps(
        {
            "preset_id": preset.preset_id,
            "display_name": preset.display_name,
            "inference_stride_frames": preset.inference_stride_frames,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _job_from_row(row: sqlite3.Row) -> ProcessingJob:
    payload = json.loads(row["preset_json"])
    configuration_hashes = json.loads(row["configuration_sha256_json"])
    if not isinstance(configuration_hashes, dict):
        raise RuntimeError("invalid processing job configuration provenance")
    return ProcessingJob(
        job_id=row["job_id"],
        session_id=row["session_id"],
        clip_id=row["clip_id"],
        preset=ProcessingPreset(**payload),
        state=ProcessingJobState(row["state"]),
        created_at_unix_ns=row["created_at_unix_ns"],
        updated_at_unix_ns=row["updated_at_unix_ns"],
        progress_current=row["progress_current"],
        progress_total=row["progress_total"],
        detail=row["detail"],
        run_id=row["run_id"],
        retry_of_job_id=row["retry_of_job_id"],
        started_at_unix_ns=row["started_at_unix_ns"],
        finished_at_unix_ns=row["finished_at_unix_ns"],
        configuration_revision=row["configuration_revision"],
        configuration_sha256_by_file=tuple(
            sorted((str(key), str(value)) for key, value in configuration_hashes.items())
        ),
        configuration_snapshot_json=row["configuration_snapshot_json"],
        task_profile_id=(
            str(row["task_profile_id"]) if row["task_profile_id"] is not None else None
        ),
        task_profile_snapshot_json=row["task_profile_snapshot_json"],
    )


def _validate_identifier(value: str, name: str) -> None:
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_.-"
    if not value.strip() or any(character not in allowed for character in value.lower()):
        raise ValueError(f"{name} contains unsupported characters")


def _validate_configuration_snapshot(value: str) -> None:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("configuration snapshot must be valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("configuration snapshot must be a JSON object")
