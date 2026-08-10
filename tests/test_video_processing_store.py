from __future__ import annotations

import sqlite3

from ui.processing import (
    ProcessingJobState,
    ProcessingJobStore,
    ProcessingPreset,
    ProcessingResultStore,
)


def test_job_queue_persists_history_and_recovers_active_jobs(tmp_path) -> None:
    path = tmp_path / ".processing" / "jobs.sqlite3"
    store = ProcessingJobStore(path)
    first = store.enqueue("a" * 32, None, ProcessingPreset())
    claimed = store.claim_next()
    assert claimed is not None and claimed.job_id == first.job_id
    store.start_run(first.job_id, "run-1", 10)
    store.update_progress(first.job_id, 3, 10, "3/10")

    recovered = ProcessingJobStore(path)
    interrupted = recovered.require(first.job_id)
    assert interrupted.state is ProcessingJobState.INTERRUPTED
    assert interrupted.progress_current == 3
    assert interrupted.started_at_unix_ns is not None
    assert interrupted.finished_at_unix_ns is not None
    assert interrupted.elapsed_seconds >= 0

    retry = recovered.retry(first.job_id)
    assert retry.retry_of_job_id == first.job_id
    assert retry.state is ProcessingJobState.QUEUED
    assert len(recovered.list_jobs()) == 2


def test_queue_cancel_distinguishes_waiting_and_running_jobs(tmp_path) -> None:
    store = ProcessingJobStore(tmp_path / "jobs.sqlite3")
    waiting = store.enqueue("a" * 32, None, ProcessingPreset())
    assert store.request_cancel(waiting.job_id).state is ProcessingJobState.CANCELED

    running = store.enqueue("b" * 32, None, ProcessingPreset())
    store.claim_next()
    store.start_run(running.job_id, "run-2", 5)
    assert store.request_cancel(running.job_id).state is ProcessingJobState.CANCELING
    assert store.mark_canceled(running.job_id).state is ProcessingJobState.CANCELED


def test_job_queue_persists_submission_configuration_provenance(tmp_path) -> None:
    path = tmp_path / "jobs.sqlite3"
    store = ProcessingJobStore(path)

    job = store.enqueue(
        "a" * 32,
        None,
        ProcessingPreset(),
        configuration_revision=7,
        configuration_sha256_by_file=(("hand-tracking.yaml", "abc123"),),
        configuration_snapshot_json='{"hand_tracking": {"detector": "mediapipe"}}',
        task_profile_id="default-manipulation",
        task_profile_snapshot_json='{"display_name": "old"}',
    )

    restarted = ProcessingJobStore(path).require(job.job_id)
    assert restarted.configuration_revision == 7
    assert restarted.configuration_sha256_by_file == (("hand-tracking.yaml", "abc123"),)
    assert restarted.configuration_snapshot_json == ('{"hand_tracking": {"detector": "mediapipe"}}')
    assert restarted.task_profile_id == "default-manipulation"
    assert restarted.task_profile_snapshot_json == '{"display_name": "old"}'

    store.request_cancel(job.job_id)
    retried = store.retry(
        job.job_id,
        task_profile_snapshot_json='{"display_name": "current"}',
    )
    assert retried.task_profile_snapshot_json == '{"display_name": "current"}'


def test_completed_run_wins_a_cancel_requested_after_its_last_frame(tmp_path) -> None:
    store = ProcessingJobStore(tmp_path / "jobs.sqlite3")
    job = store.enqueue("a" * 32, None, ProcessingPreset())
    store.claim_next()
    store.start_run(job.job_id, "run-complete", 2)
    store.update_progress(job.job_id, 2, 2, "all frames written")
    assert store.request_cancel(job.job_id).state is ProcessingJobState.CANCELING

    completed = store.complete(job.job_id)

    assert completed.state is ProcessingJobState.COMPLETED
    assert completed.progress_current == completed.progress_total == 2


def test_partial_job_is_terminal_viewable_and_retryable(tmp_path) -> None:
    store = ProcessingJobStore(tmp_path / "jobs.sqlite3")
    job = store.enqueue("a" * 32, None, ProcessingPreset())
    store.claim_next()
    store.start_run(job.job_id, "run-partial", 2)

    partial = store.partial(job.job_id, "VIO coverage 50%")
    retried = store.retry(partial.job_id)

    assert partial.state is ProcessingJobState.PARTIAL
    assert partial.progress_current == partial.progress_total == 2
    assert retried.retry_of_job_id == partial.job_id


def test_queue_migrates_v1_timestamps_without_losing_history(tmp_path) -> None:
    path = tmp_path / "jobs.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO metadata VALUES ('schema_version', '1');
            CREATE TABLE jobs (
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
                retry_of_job_id TEXT REFERENCES jobs(job_id)
            );
            """
        )

    ProcessingJobStore(path)

    with sqlite3.connect(path) as connection:
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(jobs)")}
        version = connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
    assert {
        "started_at_unix_ns",
        "finished_at_unix_ns",
        "configuration_revision",
        "configuration_sha256_json",
        "configuration_snapshot_json",
        "task_profile_id",
        "task_profile_snapshot_json",
    } <= columns
    assert version == ("5",)


def test_queue_recovers_known_mistagged_v6_without_losing_history(tmp_path) -> None:
    path = tmp_path / "jobs.sqlite3"
    store = ProcessingJobStore(path)
    job = store.enqueue(
        "a" * 32,
        None,
        ProcessingPreset(),
        configuration_revision=9,
        configuration_sha256_by_file=(("offline-hand-tracking.yaml", "sha"),),
        configuration_snapshot_json='{"offline_hand_tracking": {"detector": "vitpose"}}',
    )
    store.set_setting("auto_enqueue", "true")

    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE metadata SET value = '6' WHERE key = 'schema_version'"
        )

    reopened = ProcessingJobStore(path)
    recovered = reopened.require(job.job_id)
    assert recovered.configuration_revision == 9
    assert recovered.configuration_sha256_by_file == (("offline-hand-tracking.yaml", "sha"),)
    assert recovered.configuration_snapshot_json == (
        '{"offline_hand_tracking": {"detector": "vitpose"}}'
    )
    assert reopened.setting("auto_enqueue", "missing") == "true"

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone() == ("5",)
        assert connection.execute("SELECT count(*) FROM jobs").fetchone() == (1,)


def test_queue_rejects_unknown_v6_shape(tmp_path) -> None:
    path = tmp_path / "jobs.sqlite3"
    store = ProcessingJobStore(path)
    store.enqueue("a" * 32, None, ProcessingPreset())
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE jobs ADD COLUMN unknown_field TEXT")
        connection.execute(
            "UPDATE metadata SET value = '6' WHERE key = 'schema_version'"
        )

    try:
        ProcessingJobStore(path)
    except RuntimeError as error:
        assert str(error) == "unsupported video-processing queue schema"
    else:
        raise AssertionError("unknown v6 queue shape was accepted")


def test_results_are_addressed_by_clip_frame_and_session_time(tmp_path) -> None:
    path = tmp_path / "results.sqlite"
    store = ProcessingResultStore(path)
    result = {
        "session_id": "a" * 32,
        "sequence_id": "b" * 32,
        "frame_index": 6,
        "session_time_ns": 200,
        "hands": [],
    }
    store.put(result)

    reader = ProcessingResultStore(path, read_only=True)
    assert reader.result_for_frame("b" * 32, 6, 200) == result
    assert reader.result_for_frame("b" * 32, 7, 250) is None
    assert reader.result_for_frame("b" * 32, 7, 250, hold_previous_frames=1) == result
    assert reader.result_for_frame("b" * 32, 7, 199, hold_previous_frames=1) is None


def test_v2_results_keep_raw_and_final_streams_separate(tmp_path) -> None:
    path = tmp_path / "results.sqlite"
    store = ProcessingResultStore(path)
    raw = {
        "session_id": "a" * 32,
        "sequence_id": "b" * 32,
        "frame_index": 0,
        "session_time_ns": 100,
        "hands": [{"temporal_source": "observed"}],
    }
    final = {
        **raw,
        "hands": [{"temporal_source": "interpolated"}],
    }
    store.put_raw(raw)
    store.put_final(final)

    reader = ProcessingResultStore(path, read_only=True)
    assert reader.schema_version == "2"
    assert reader.count(raw=True) == 1
    assert reader.raw_result_for_frame("b" * 32, 0, 100) == raw
    assert reader.result_for_frame("b" * 32, 0, 100) == final


def test_v1_results_remain_readable_without_raw_table(tmp_path) -> None:
    path = tmp_path / "v1.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO metadata VALUES ('schema_version', '1');
            CREATE TABLE frame_results (
                clip_id TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                session_time_ns INTEGER NOT NULL,
                result_json TEXT NOT NULL,
                PRIMARY KEY(clip_id, frame_index, session_time_ns)
            );
            INSERT INTO frame_results VALUES ('clip', 0, 100, '{"hands": []}');
            """
        )

    reader = ProcessingResultStore(path, read_only=True)
    assert reader.schema_version == "1"
    assert reader.raw_result_for_frame("clip", 0, 100) is None
    assert reader.result_for_frame("clip", 0, 100) == {"hands": []}
