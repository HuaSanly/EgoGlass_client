from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sqlite3
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .adapters.mp4_recorder import RecordedVideoFrame
from .recording_models import CaptureSessionQuality
from .webrtc_matcher import FrameMetadataMatch
from .webrtc_models import ImuCapabilities, ImuSample, VideoFrameMetadata

TELEMETRY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CachedConnection:
    connection_session_id: str
    device_session_id: str
    state: str
    observed_at_client_monotonic_ns: int


@dataclass(frozen=True)
class ClipTimeBounds:
    started_at_session_time_ns: int | None
    ended_at_session_time_ns: int | None


@dataclass(frozen=True)
class CaptureDerivationResult:
    origin_elapsed_realtime_ns: int | None
    origin_event: str | None
    clip_time_bounds: dict[str, ClipTimeBounds]
    quality: CaptureSessionQuality


@dataclass(frozen=True)
class _DatabaseWrite:
    method: str
    args: tuple[object, ...]


class CaptureSessionWriter:
    """Move bounded, batched SQLite writes off the WebRTC event loop."""

    def __init__(
        self,
        database: CaptureSessionDatabase,
        *,
        max_queue_size: int = 32_768,
        max_batch_size: int = 512,
    ) -> None:
        if max_queue_size < 1 or max_batch_size < 1:
            raise ValueError("telemetry queue and batch sizes must be positive")
        self.database = database
        self._queue: asyncio.Queue[_DatabaseWrite | None] = asyncio.Queue(max_queue_size)
        self._max_batch_size = max_batch_size
        self._accepting = True
        self._closed = False
        self._failure: Exception | None = None
        self.overflow_count = 0
        self._task = asyncio.create_task(self._run())

    def enqueue(self, method: str, *args: object) -> bool:
        if not self._accepting or self._failure is not None:
            return False
        try:
            self._queue.put_nowait(_DatabaseWrite(method, args))
        except asyncio.QueueFull:
            self.overflow_count += 1
            return False
        return True

    async def flush(self) -> None:
        await self._queue.join()
        self._raise_if_failed()
        await asyncio.to_thread(self.database.flush)

    async def finalize(self) -> CaptureDerivationResult:
        if self._closed:
            raise RuntimeError("capture telemetry writer is already finalized")
        self._accepting = False
        await self.flush()
        result = await asyncio.to_thread(
            self.database.finalize_derivations,
            self.overflow_count,
        )
        self._queue.put_nowait(None)
        await self._task
        self._raise_if_failed()
        await asyncio.to_thread(self.database.checkpoint_and_close)
        self._closed = True
        return result

    async def _run(self) -> None:
        while True:
            first = await self._queue.get()
            if first is None:
                self._queue.task_done()
                return
            writes = [first]
            while len(writes) < self._max_batch_size:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    self._queue.task_done()
                    break
                writes.append(item)
            try:
                await asyncio.to_thread(self.database.apply_batch, writes)
            except Exception as error:
                self._failure = error
                self._accepting = False
            finally:
                for _write in writes:
                    self._queue.task_done()
            if self._failure is not None:
                while True:
                    try:
                        self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    self._queue.task_done()

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError("capture telemetry writer failed") from self._failure


class CaptureSessionDatabase:
    """SQLite representation of capture-session-v1 logical records."""

    def __init__(self, session_id: str, path: Path) -> None:
        self.session_id = session_id
        self.path = path
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS session_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS session_events (
                event_row_id INTEGER PRIMARY KEY,
                schema_version TEXT NOT NULL DEFAULT '1.0',
                record_type TEXT NOT NULL DEFAULT 'session_event',
                session_id TEXT NOT NULL,
                event_id TEXT NOT NULL UNIQUE,
                event_type TEXT NOT NULL,
                occurred_at_client_perf_counter_ns INTEGER NOT NULL,
                occurred_at_elapsed_realtime_ns INTEGER,
                session_time_ns INTEGER,
                clip_id TEXT,
                details_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS connection_segments (
                segment_id INTEGER PRIMARY KEY,
                connection_session_id TEXT NOT NULL,
                device_session_id TEXT NOT NULL,
                started_at_client_perf_counter_ns INTEGER NOT NULL,
                ended_at_client_perf_counter_ns INTEGER,
                end_state TEXT
            );
            CREATE TABLE IF NOT EXISTS imu_capabilities (
                capability_id INTEGER PRIMARY KEY,
                connection_session_id TEXT NOT NULL,
                received_at_client_perf_counter_ns INTEGER NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS imu_samples (
                sample_id INTEGER PRIMARY KEY,
                schema_version TEXT NOT NULL DEFAULT '1.0',
                record_type TEXT NOT NULL DEFAULT 'imu_sample',
                session_id TEXT NOT NULL,
                connection_session_id TEXT NOT NULL,
                sensor_type TEXT NOT NULL,
                android_sensor_type INTEGER NOT NULL,
                sequence_number INTEGER NOT NULL,
                sensor_event_monotonic_ns INTEGER NOT NULL,
                received_at_elapsed_realtime_ns INTEGER NOT NULL,
                received_at_client_perf_counter_ns INTEGER NOT NULL,
                alignment_status TEXT NOT NULL DEFAULT 'pending',
                session_time_ns INTEGER,
                timestamp_uncertainty_ns INTEGER,
                clock_mapping_segment_id TEXT,
                accuracy INTEGER NOT NULL,
                value_x REAL NOT NULL,
                value_y REAL NOT NULL,
                value_z REAL NOT NULL,
                unit TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS imu_samples_timeline
                ON imu_samples(sensor_event_monotonic_ns, sample_id);
            CREATE TABLE IF NOT EXISTS video_frame_metadata_raw (
                metadata_row_id INTEGER PRIMARY KEY,
                schema_version TEXT NOT NULL DEFAULT '1.0',
                record_type TEXT NOT NULL DEFAULT 'video_frame_metadata',
                session_id TEXT NOT NULL,
                video_frame_metadata_id TEXT NOT NULL UNIQUE,
                connection_session_id TEXT NOT NULL,
                camera_start_generation INTEGER NOT NULL,
                frame_id INTEGER NOT NULL,
                captured_at_rokid_sdk_ms INTEGER NOT NULL,
                received_at_elapsed_realtime_ns INTEGER NOT NULL,
                video_at_monotonic_ns INTEGER NOT NULL,
                rtp_timestamp_90khz INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                rotation_degrees INTEGER NOT NULL,
                capture_config_id TEXT NOT NULL,
                received_at_client_perf_counter_ns INTEGER NOT NULL,
                ingest_status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS video_frame_matches (
                match_id INTEGER PRIMARY KEY,
                connection_session_id TEXT NOT NULL,
                camera_start_generation INTEGER NOT NULL,
                frame_id INTEGER NOT NULL,
                captured_at_rokid_sdk_ms INTEGER NOT NULL,
                received_at_elapsed_realtime_ns INTEGER NOT NULL,
                video_at_monotonic_ns INTEGER NOT NULL,
                rtp_timestamp_90khz INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                rotation_degrees INTEGER NOT NULL,
                capture_config_id TEXT NOT NULL,
                decoded_frame_index INTEGER,
                decoded_frame_pts INTEGER NOT NULL,
                decoded_frame_time_base_num INTEGER,
                decoded_frame_time_base_den INTEGER,
                received_at_client_perf_counter_ns INTEGER,
                timestamp_match_error_90khz INTEGER NOT NULL,
                UNIQUE(connection_session_id, camera_start_generation, frame_id)
            );
            CREATE INDEX IF NOT EXISTS video_frame_matches_source_pts
                ON video_frame_matches(
                    decoded_frame_pts,
                    decoded_frame_time_base_num,
                    decoded_frame_time_base_den
                );
            CREATE TABLE IF NOT EXISTS video_frame_index (
                video_frame_row_id INTEGER PRIMARY KEY,
                schema_version TEXT NOT NULL DEFAULT '1.0',
                record_type TEXT NOT NULL DEFAULT 'video_frame_index',
                session_id TEXT NOT NULL,
                clip_id TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                mp4_pts INTEGER NOT NULL,
                mp4_time_base_numerator INTEGER NOT NULL,
                mp4_time_base_denominator INTEGER NOT NULL,
                video_frame_metadata_id TEXT,
                frame_id INTEGER,
                captured_at_rokid_sdk_ms INTEGER,
                received_at_elapsed_realtime_ns INTEGER,
                video_at_monotonic_ns INTEGER,
                rtp_timestamp_90khz INTEGER,
                received_at_client_perf_counter_ns INTEGER NOT NULL,
                alignment_status TEXT NOT NULL DEFAULT 'pending',
                session_time_ns INTEGER,
                timestamp_uncertainty_ns INTEGER,
                clock_mapping_segment_id TEXT,
                metadata_match_status TEXT NOT NULL,
                source_frame_pts INTEGER,
                source_frame_time_base_num INTEGER,
                source_frame_time_base_den INTEGER,
                frame_metadata_match_id INTEGER UNIQUE
                    REFERENCES video_frame_matches(match_id),
                UNIQUE(clip_id, frame_index)
            );
            CREATE TABLE IF NOT EXISTS clock_mapping_segments (
                mapping_row_id INTEGER PRIMARY KEY,
                schema_version TEXT NOT NULL DEFAULT '1.0',
                record_type TEXT NOT NULL DEFAULT 'clock_mapping_segment',
                session_id TEXT NOT NULL,
                connection_session_id TEXT NOT NULL,
                source_instance_id TEXT NOT NULL,
                segment_index INTEGER NOT NULL,
                clock_mapping_segment_id TEXT NOT NULL UNIQUE,
                source_clock_id TEXT NOT NULL,
                target_clock_id TEXT NOT NULL,
                source_from INTEGER NOT NULL,
                source_to INTEGER NOT NULL,
                target_from_ns INTEGER NOT NULL,
                target_to_ns INTEGER NOT NULL,
                scale_target_ns_per_source_unit REAL NOT NULL,
                offset_target_ns REAL NOT NULL,
                uncertainty_ns INTEGER NOT NULL,
                fit_method TEXT NOT NULL,
                sample_count INTEGER NOT NULL,
                residual_p95_ns INTEGER NOT NULL,
                residual_max_ns INTEGER NOT NULL,
                status TEXT NOT NULL,
                estimator_state TEXT NOT NULL,
                segment_reason TEXT NOT NULL,
                outlier_count INTEGER NOT NULL
            );
            """
        )
        self._connection.executemany(
            "INSERT OR REPLACE INTO session_metadata(key, value) VALUES (?, ?)",
            (
                ("telemetry_schema_version", str(TELEMETRY_SCHEMA_VERSION)),
                ("session_id", self.session_id),
            ),
        )
        self._connection.commit()

    def apply_batch(self, writes: Sequence[_DatabaseWrite]) -> None:
        try:
            self._connection.execute("BEGIN")
            for write in writes:
                getattr(self, write.method)(*write.args)
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def flush(self) -> None:
        self._connection.commit()

    def begin_clip_delete(self, clip_id: str) -> CaptureSessionQuality:
        self._connection.execute("BEGIN")
        self._connection.execute(
            "DELETE FROM video_frame_index WHERE clip_id = ?",
            (clip_id,),
        )
        self._connection.execute(
            "DELETE FROM session_events WHERE clip_id = ?",
            (clip_id,),
        )
        return self.quality()

    def commit_clip_delete(self) -> None:
        self._connection.commit()

    def rollback_clip_delete(self) -> None:
        self._connection.rollback()

    def record_event(
        self,
        event_id: str,
        event_type: str,
        client_perf_counter_ns: int,
        elapsed_realtime_ns: int | None,
        clip_id: str | None,
        details: dict[str, object],
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO session_events(
                session_id, event_id, event_type,
                occurred_at_client_perf_counter_ns,
                occurred_at_elapsed_realtime_ns, session_time_ns,
                clip_id, details_json
            ) VALUES (?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            (
                self.session_id,
                event_id,
                event_type,
                client_perf_counter_ns,
                elapsed_realtime_ns,
                clip_id,
                json.dumps(details, sort_keys=True, separators=(",", ":")),
            ),
        )

    def begin_connection(self, connection: CachedConnection, started_at_ns: int) -> None:
        row = self._connection.execute(
            """
            SELECT segment_id FROM connection_segments
            WHERE connection_session_id = ?
              AND ended_at_client_perf_counter_ns IS NULL
            ORDER BY segment_id DESC LIMIT 1
            """,
            (connection.connection_session_id,),
        ).fetchone()
        if row is None:
            self._connection.execute(
                """
                INSERT INTO connection_segments(
                    connection_session_id, device_session_id,
                    started_at_client_perf_counter_ns
                ) VALUES (?, ?, ?)
                """,
                (
                    connection.connection_session_id,
                    connection.device_session_id,
                    started_at_ns,
                ),
            )

    def end_connection(
        self,
        connection_session_id: str,
        ended_at_ns: int,
        state: str,
    ) -> None:
        self._connection.execute(
            """
            UPDATE connection_segments
            SET ended_at_client_perf_counter_ns = ?, end_state = ?
            WHERE segment_id = (
                SELECT segment_id FROM connection_segments
                WHERE connection_session_id = ?
                  AND ended_at_client_perf_counter_ns IS NULL
                ORDER BY segment_id DESC LIMIT 1
            )
            """,
            (ended_at_ns, state, connection_session_id),
        )

    def record_capabilities(
        self,
        connection_session_id: str,
        capabilities: ImuCapabilities,
        received_at_client_perf_counter_ns: int,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO imu_capabilities(
                connection_session_id, received_at_client_perf_counter_ns,
                payload_json
            ) VALUES (?, ?, ?)
            """,
            (
                connection_session_id,
                received_at_client_perf_counter_ns,
                json.dumps(
                    capabilities.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        )

    def record_imu_sample(
        self,
        connection_session_id: str,
        sample: ImuSample,
        received_at_client_perf_counter_ns: int,
    ) -> None:
        unit = "m_s2" if sample.sensor_type.value == "accelerometer" else "rad_s"
        self._connection.execute(
            """
            INSERT INTO imu_samples(
                session_id, connection_session_id, sensor_type,
                android_sensor_type, sequence_number,
                sensor_event_monotonic_ns, received_at_elapsed_realtime_ns,
                received_at_client_perf_counter_ns, accuracy,
                value_x, value_y, value_z, unit
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.session_id,
                connection_session_id,
                sample.sensor_type.value,
                sample.android_sensor_type,
                sample.sequence_number,
                sample.sensor_event_monotonic_ns,
                sample.received_at_elapsed_realtime_ns,
                received_at_client_perf_counter_ns,
                sample.accuracy,
                *sample.values,
                unit,
            ),
        )

    def record_frame_match(
        self,
        connection_session_id: str,
        match: FrameMetadataMatch,
    ) -> None:
        metadata = match.metadata
        self._connection.execute(
            """
            INSERT INTO video_frame_matches(
                connection_session_id, camera_start_generation, frame_id,
                captured_at_rokid_sdk_ms,
                received_at_elapsed_realtime_ns, video_at_monotonic_ns,
                rtp_timestamp_90khz, width, height, rotation_degrees,
                capture_config_id, decoded_frame_index, decoded_frame_pts,
                decoded_frame_time_base_num, decoded_frame_time_base_den,
                received_at_client_perf_counter_ns,
                timestamp_match_error_90khz
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                connection_session_id,
                metadata.camera_start_generation,
                metadata.frame_id,
                metadata.captured_at_rokid_sdk_ms,
                metadata.received_at_elapsed_realtime_ns,
                metadata.video_at_monotonic_ns,
                metadata.rtp_timestamp_90khz,
                metadata.width,
                metadata.height,
                metadata.rotation_degrees,
                metadata.capture_config_id,
                match.decoded_frame_index,
                match.decoded_frame_pts,
                match.decoded_frame_time_base_num,
                match.decoded_frame_time_base_den,
                match.decoded_frame_received_at_client_monotonic_ns,
                match.timestamp_match_error_90khz,
            ),
        )

    def record_video_frame_metadata(
        self,
        connection_session_id: str,
        metadata: VideoFrameMetadata,
        received_at_client_perf_counter_ns: int,
        camera_start_generation: int,
        ingest_status: str,
    ) -> None:
        if camera_start_generation != metadata.camera_start_generation:
            raise ValueError("camera start generation does not match frame metadata")
        next_row_id = self._connection.execute(
            "SELECT COALESCE(MAX(metadata_row_id), 0) + 1 FROM video_frame_metadata_raw"
        ).fetchone()[0]
        metadata_id = f"meta-{connection_session_id[:12]}-{camera_start_generation}-{next_row_id}"
        self._connection.execute(
            """
            INSERT INTO video_frame_metadata_raw(
                session_id, video_frame_metadata_id, connection_session_id,
                camera_start_generation, frame_id, captured_at_rokid_sdk_ms,
                received_at_elapsed_realtime_ns, video_at_monotonic_ns,
                rtp_timestamp_90khz, width, height, rotation_degrees,
                capture_config_id, received_at_client_perf_counter_ns,
                ingest_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.session_id,
                metadata_id,
                connection_session_id,
                camera_start_generation,
                metadata.frame_id,
                metadata.captured_at_rokid_sdk_ms,
                metadata.received_at_elapsed_realtime_ns,
                metadata.video_at_monotonic_ns,
                metadata.rtp_timestamp_90khz,
                metadata.width,
                metadata.height,
                metadata.rotation_degrees,
                metadata.capture_config_id,
                received_at_client_perf_counter_ns,
                ingest_status,
            ),
        )

    def record_clip_frames(
        self,
        clip_id: str,
        connection_session_id: str,
        camera_start_generation: int,
        frames: Sequence[RecordedVideoFrame],
        expected_frame_count: int,
    ) -> None:
        if len(frames) != expected_frame_count:
            raise ValueError("every encoded frame must have exact muxed MP4 timing")
        for frame in frames:
            match = self._find_frame_match(
                connection_session_id,
                camera_start_generation,
                frame,
            )
            self._insert_video_frame(clip_id, frame, match)

    def _find_frame_match(
        self,
        connection_session_id: str,
        camera_start_generation: int,
        frame: RecordedVideoFrame,
    ) -> sqlite3.Row | None:
        if frame.source_frame_pts is None:
            return None
        if (
            frame.source_frame_time_base_num is None
            or frame.source_frame_time_base_den is None
        ):
            timing_predicate = """
              decoded_frame_pts = ?
              AND decoded_frame_time_base_num IS NULL
              AND decoded_frame_time_base_den IS NULL
            """
            timing_parameters = (frame.source_frame_pts,)
        else:
            timing_predicate = """
              decoded_frame_time_base_num IS NOT NULL
              AND decoded_frame_time_base_den IS NOT NULL
              AND decoded_frame_pts * decoded_frame_time_base_num * ?
                  = ? * ? * decoded_frame_time_base_den
            """
            timing_parameters = (
                frame.source_frame_time_base_den,
                frame.source_frame_pts,
                frame.source_frame_time_base_num,
            )
        self._connection.row_factory = sqlite3.Row
        row = self._connection.execute(
            f"""
            SELECT * FROM video_frame_matches
            WHERE connection_session_id = ?
              AND camera_start_generation = ?
              AND {timing_predicate}
              AND match_id NOT IN (
                  SELECT frame_metadata_match_id FROM video_frame_index
                  WHERE frame_metadata_match_id IS NOT NULL
              )
            ORDER BY match_id LIMIT 1
            """,
            (
                connection_session_id,
                camera_start_generation,
                *timing_parameters,
            ),
        ).fetchone()
        self._connection.row_factory = None
        return row

    def _insert_video_frame(
        self,
        clip_id: str,
        frame: RecordedVideoFrame,
        match: sqlite3.Row | None,
    ) -> None:
        metadata_id = None
        if match is not None:
            metadata_row = self._connection.execute(
                """
                SELECT video_frame_metadata_id FROM video_frame_metadata_raw
                WHERE connection_session_id = ? AND frame_id = ?
                  AND camera_start_generation = ?
                  AND ingest_status = 'accepted'
                ORDER BY metadata_row_id DESC LIMIT 1
                """,
                (
                    match["connection_session_id"],
                    match["frame_id"],
                    match["camera_start_generation"],
                ),
            ).fetchone()
            if metadata_row is None:
                match = None
            else:
                metadata_id = metadata_row[0]
        error = None if match is None else int(match["timestamp_match_error_90khz"])
        status = "unmatched" if error is None else ("exact" if error == 0 else "within_tolerance")
        received_at_client_ns = frame.received_at_client_perf_counter_ns
        if match is not None and match["received_at_client_perf_counter_ns"] is not None:
            received_at_client_ns = int(match["received_at_client_perf_counter_ns"])
        self._connection.execute(
            """
            INSERT INTO video_frame_index(
                session_id, clip_id, frame_index, mp4_pts,
                mp4_time_base_numerator, mp4_time_base_denominator,
                video_frame_metadata_id, frame_id, captured_at_rokid_sdk_ms,
                received_at_elapsed_realtime_ns, video_at_monotonic_ns,
                rtp_timestamp_90khz, received_at_client_perf_counter_ns,
                metadata_match_status, source_frame_pts,
                source_frame_time_base_num, source_frame_time_base_den,
                frame_metadata_match_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.session_id,
                clip_id,
                frame.frame_index,
                frame.mp4_pts,
                frame.mp4_time_base_num,
                frame.mp4_time_base_den,
                metadata_id,
                None if match is None else match["frame_id"],
                None if match is None else match["captured_at_rokid_sdk_ms"],
                None if match is None else match["received_at_elapsed_realtime_ns"],
                None if match is None else match["video_at_monotonic_ns"],
                None if match is None else match["rtp_timestamp_90khz"],
                received_at_client_ns,
                status,
                frame.source_frame_pts,
                frame.source_frame_time_base_num,
                frame.source_frame_time_base_den,
                None if match is None else match["match_id"],
            ),
        )

    def finalize_derivations(
        self,
        telemetry_queue_overflow_count: int,
    ) -> CaptureDerivationResult:
        self._connection.execute("BEGIN")
        try:
            self._rebuild_clock_mappings()
            origin, origin_event = self._derive_session_time()
            self._connection.executemany(
                "INSERT OR REPLACE INTO session_metadata(key, value) VALUES (?, ?)",
                (
                    ("origin_elapsed_realtime_ns", "" if origin is None else str(origin)),
                    ("origin_event", "" if origin_event is None else origin_event),
                    ("telemetry_queue_overflow_count", str(telemetry_queue_overflow_count)),
                ),
            )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise
        return CaptureDerivationResult(
            origin_elapsed_realtime_ns=origin,
            origin_event=origin_event,
            clip_time_bounds=self._clip_bounds(),
            quality=self.quality(telemetry_queue_overflow_count),
        )

    def _rebuild_clock_mappings(self) -> None:
        self._connection.execute("DELETE FROM clock_mapping_segments")
        camera_rows = self._connection.execute(
            """
            SELECT connection_session_id,
                   'camera-' || camera_start_generation,
                   captured_at_rokid_sdk_ms,
                   received_at_elapsed_realtime_ns
            FROM video_frame_metadata_raw
            WHERE ingest_status = 'accepted'
            ORDER BY metadata_row_id
            """
        ).fetchall()
        self._write_mapping_groups(
            camera_rows,
            source_clock_id="rokid_sdk_ms",
            nominal_scale=1_000_000.0,
        )
        imu_rows = self._connection.execute(
            """
            SELECT connection_session_id, sensor_type,
                   sensor_event_monotonic_ns,
                   received_at_elapsed_realtime_ns
            FROM imu_samples ORDER BY sample_id
            """
        ).fetchall()
        self._write_mapping_groups(
            imu_rows,
            source_clock_id="sensor_event_monotonic_ns",
            nominal_scale=1.0,
        )

    def _write_mapping_groups(
        self,
        rows: Iterable[tuple[str, str, int, int]],
        *,
        source_clock_id: str,
        nominal_scale: float,
    ) -> None:
        groups: dict[tuple[str, str], list[tuple[int, int]]] = {}
        for connection_id, source_instance_id, source_time, target_time in rows:
            groups.setdefault((connection_id, source_instance_id), []).append(
                (source_time, target_time)
            )
        for (connection_id, source_instance_id), observations in groups.items():
            segments = _split_clock_observations(observations)
            for index, (segment, discontinuous) in enumerate(segments):
                fit = _fit_robust_affine(segment, nominal_scale)
                group_key = f"{connection_id}:{source_instance_id}"
                group_digest = hashlib.sha256(group_key.encode("utf-8")).hexdigest()[:12]
                mapping_id = f"map-{source_clock_id[:8]}-{group_digest}-{index}"
                status = "rejected" if fit.rejected else "provisional"
                state = (
                    "discontinuous"
                    if discontinuous
                    else ("estimated" if fit.inlier_count >= 10 else "estimating")
                )
                reason = "timestamp_discontinuity" if discontinuous else "camera_start"
                if source_clock_id == "sensor_event_monotonic_ns":
                    reason = "session_start" if not discontinuous else reason
                self._connection.execute(
                    """
                    INSERT INTO clock_mapping_segments(
                        session_id, connection_session_id, source_instance_id,
                        segment_index, clock_mapping_segment_id, source_clock_id,
                        target_clock_id, source_from, source_to,
                        target_from_ns, target_to_ns,
                        scale_target_ns_per_source_unit, offset_target_ns,
                        uncertainty_ns, fit_method, sample_count,
                        residual_p95_ns, residual_max_ns, status,
                        estimator_state, segment_reason, outlier_count
                    ) VALUES (?, ?, ?, ?, ?, ?, 'glasses_elapsed_realtime_ns',
                              ?, ?, ?, ?, ?, ?, ?, 'affine_robust', ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.session_id,
                        connection_id,
                        source_instance_id,
                        index,
                        mapping_id,
                        source_clock_id,
                        segment[0][0],
                        segment[-1][0],
                        min(target for _source, target in segment),
                        max(target for _source, target in segment),
                        fit.scale,
                        fit.offset,
                        fit.uncertainty_ns,
                        fit.inlier_count,
                        fit.residual_p95_ns,
                        fit.residual_max_ns,
                        status,
                        state,
                        reason,
                        fit.outlier_count,
                    ),
                )

    def _derive_session_time(self) -> tuple[int | None, str | None]:
        mappings = self._mapping_rows()
        mapped_imu_times: list[int] = []
        for connection_id, sensor_type, source_time in self._connection.execute(
            """
            SELECT connection_session_id, sensor_type,
                   sensor_event_monotonic_ns FROM imu_samples
            """
        ).fetchall():
            mapping = _find_mapping(
                mappings,
                connection_id,
                sensor_type,
                "sensor_event_monotonic_ns",
                source_time,
            )
            if mapping is not None:
                if mapping["status"] == "rejected":
                    continue
                mapped_imu_times.append(
                    round(float(mapping["scale"]) * source_time + float(mapping["offset"]))
                )
        if mapped_imu_times:
            origin = min(mapped_imu_times)
            origin_event = "first_imu_sample"
        else:
            video_origin = self._connection.execute(
                "SELECT MIN(received_at_elapsed_realtime_ns) FROM video_frame_matches"
            ).fetchone()[0]
            origin = None if video_origin is None else int(video_origin)
            origin_event = None if origin is None else "first_video_frame"
        if origin is None:
            return None, None

        for row in self._connection.execute(
            """
            SELECT sample_id, connection_session_id, sensor_type,
                   sensor_event_monotonic_ns FROM imu_samples
            """
        ).fetchall():
            mapping = _find_mapping(
                mappings,
                row[1],
                row[2],
                "sensor_event_monotonic_ns",
                row[3],
            )
            if mapping is None:
                continue
            if mapping["status"] == "rejected":
                self._connection.execute(
                    """
                    UPDATE imu_samples SET alignment_status = 'rejected',
                        session_time_ns = NULL, timestamp_uncertainty_ns = NULL,
                        clock_mapping_segment_id = ? WHERE sample_id = ?
                    """,
                    (mapping["id"], row[0]),
                )
                continue
            target = round(float(mapping["scale"]) * row[3] + float(mapping["offset"]))
            self._connection.execute(
                """
                UPDATE imu_samples SET alignment_status = 'mapped', session_time_ns = ?,
                    timestamp_uncertainty_ns = ?, clock_mapping_segment_id = ?
                WHERE sample_id = ?
                """,
                (
                    max(0, target - origin),
                    int(mapping["uncertainty"]),
                    mapping["id"],
                    row[0],
                ),
            )
        frame_rows = self._connection.execute(
            """
            SELECT video_frame_row_id, video_frame_metadata_id,
                   captured_at_rokid_sdk_ms FROM video_frame_index
            WHERE video_frame_metadata_id IS NOT NULL
            """
        ).fetchall()
        for video_row_id, metadata_id, source_time in frame_rows:
            metadata_row = self._connection.execute(
                """
                SELECT connection_session_id, camera_start_generation
                FROM video_frame_metadata_raw
                WHERE video_frame_metadata_id = ?
                """,
                (metadata_id,),
            ).fetchone()
            if metadata_row is None:
                continue
            mapping = _find_mapping(
                mappings,
                metadata_row[0],
                f"camera-{metadata_row[1]}",
                "rokid_sdk_ms",
                source_time,
            )
            if mapping is None:
                continue
            if mapping["status"] == "rejected":
                self._connection.execute(
                    """
                    UPDATE video_frame_index SET alignment_status = 'rejected',
                        session_time_ns = NULL, timestamp_uncertainty_ns = NULL,
                        clock_mapping_segment_id = ? WHERE video_frame_row_id = ?
                    """,
                    (mapping["id"], video_row_id),
                )
                continue
            target = round(
                float(mapping["scale"]) * source_time + float(mapping["offset"])
            )
            self._connection.execute(
                """
                UPDATE video_frame_index SET alignment_status = 'mapped', session_time_ns = ?,
                    timestamp_uncertainty_ns = ?, clock_mapping_segment_id = ?
                WHERE video_frame_row_id = ?
                """,
                (
                    max(0, target - origin),
                    int(mapping["uncertainty"]),
                    mapping["id"],
                    video_row_id,
                ),
            )
        self._connection.execute(
            """
            UPDATE session_events
            SET session_time_ns = MAX(0, occurred_at_elapsed_realtime_ns - ?)
            WHERE occurred_at_elapsed_realtime_ns IS NOT NULL
            """,
            (origin,),
        )
        return origin, origin_event

    def _mapping_rows(self) -> list[dict[str, object]]:
        rows = self._connection.execute(
            """
            SELECT connection_session_id, source_instance_id,
                   source_clock_id, source_from,
                   source_to, scale_target_ns_per_source_unit,
                   offset_target_ns, uncertainty_ns,
                   clock_mapping_segment_id, status
            FROM clock_mapping_segments
            """
        ).fetchall()
        return [
            {
                "connection": row[0],
                "source_instance": row[1],
                "source_clock": row[2],
                "source_from": row[3],
                "source_to": row[4],
                "scale": row[5],
                "offset": row[6],
                "uncertainty": row[7],
                "id": row[8],
                "status": row[9],
            }
            for row in rows
        ]

    def _clip_bounds(self) -> dict[str, ClipTimeBounds]:
        return {
            clip_id: ClipTimeBounds(started, ended)
            for clip_id, started, ended in self._connection.execute(
                """
                SELECT clip_id, MIN(session_time_ns), MAX(session_time_ns)
                FROM video_frame_index GROUP BY clip_id
                """
            ).fetchall()
        }

    def quality(self, telemetry_queue_overflow_count: int | None = None) -> CaptureSessionQuality:
        if telemetry_queue_overflow_count is None:
            row = self._connection.execute(
                "SELECT value FROM session_metadata WHERE key = 'telemetry_queue_overflow_count'"
            ).fetchone()
            telemetry_queue_overflow_count = 0 if row is None else int(row[0])
        sensor_counts = dict(
            self._connection.execute(
                "SELECT sensor_type, COUNT(*) FROM imu_samples GROUP BY sensor_type"
            ).fetchall()
        )
        gap_count, duplicate_count, out_of_order_count = _sequence_quality(
            self._connection.execute(
                """
                SELECT connection_session_id, sensor_type, sequence_number
                FROM imu_samples ORDER BY sample_id
                """
            ).fetchall()
        )
        counts = self._connection.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM imu_samples),
                (SELECT COUNT(*) FROM connection_segments),
                (SELECT COUNT(*) FROM video_frame_matches),
                (SELECT COUNT(*) FROM video_frame_metadata_raw),
                (SELECT COUNT(*) FROM video_frame_metadata_raw metadata
                 WHERE NOT EXISTS (
                    SELECT 1 FROM video_frame_index frame
                    WHERE frame.video_frame_metadata_id = metadata.video_frame_metadata_id
                 )),
                (SELECT COUNT(*) FROM video_frame_index),
                (SELECT COUNT(*) FROM video_frame_index
                 WHERE metadata_match_status IN ('exact', 'within_tolerance')),
                (SELECT COUNT(*) FROM clock_mapping_segments),
                (SELECT COUNT(*) FROM clock_mapping_segments WHERE status = 'rejected'),
                (SELECT MAX(uncertainty_ns) FROM clock_mapping_segments),
                (SELECT COUNT(*) FROM imu_samples WHERE alignment_status != 'mapped')
            """
        ).fetchone()
        assert counts is not None
        recorded = int(counts[5])
        recorded_matched = int(counts[6])
        return CaptureSessionQuality(
            imu_sample_count=int(counts[0]),
            accelerometer_sample_count=int(sensor_counts.get("accelerometer", 0)),
            gyroscope_sample_count=int(sensor_counts.get("gyroscope", 0)),
            imu_sequence_gap_count=gap_count,
            imu_duplicate_sample_count=duplicate_count,
            imu_out_of_order_sample_count=out_of_order_count,
            telemetry_queue_overflow_count=telemetry_queue_overflow_count,
            connection_segment_count=int(counts[1]),
            matched_video_frame_count=int(counts[2]),
            video_metadata_count=int(counts[3]),
            unmatched_video_metadata_count=int(counts[4]),
            recorded_video_frame_count=recorded,
            recorded_video_frame_metadata_match_count=recorded_matched,
            metadata_match_coverage=(
                None if not recorded else round(recorded_matched / recorded, 6)
            ),
            timestamp_mapping_segment_count=int(counts[7]),
            rejected_clock_mapping_segment_count=int(counts[8]),
            timestamp_max_uncertainty_ns=(
                None if counts[9] is None else int(counts[9])
            ),
            unaligned_imu_sample_count=int(counts[10]),
        )

    def quality_counts(self) -> tuple[int, int, int]:
        gap_count, duplicate_count, out_of_order_count = _sequence_quality(
            self._connection.execute(
                """
                SELECT connection_session_id, sensor_type, sequence_number
                FROM imu_samples ORDER BY sample_id
                """
            ).fetchall()
        )
        return gap_count, duplicate_count, out_of_order_count

    def checkpoint_and_close(self) -> None:
        self._connection.commit()
        self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._connection.close()


def read_capture_quality(path: Path) -> CaptureSessionQuality:
    if not path.is_file():
        return CaptureSessionQuality()
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        database = CaptureSessionDatabase.__new__(CaptureSessionDatabase)
        database.path = path
        database._connection = connection
        return database.quality()
    except (sqlite3.Error, ValueError):
        return CaptureSessionQuality()
    finally:
        connection.close()


@dataclass(frozen=True)
class _AffineFit:
    scale: float
    offset: float
    uncertainty_ns: int
    residual_p95_ns: int
    residual_max_ns: int
    inlier_count: int
    outlier_count: int
    rejected: bool


def _split_clock_observations(
    observations: Sequence[tuple[int, int]],
) -> list[tuple[list[tuple[int, int]], bool]]:
    segments: list[tuple[list[tuple[int, int]], bool]] = []
    for observation in observations:
        previous = None if not segments else segments[-1][0][-1]
        discontinuity = previous is not None and (
            observation[0] < previous[0]
            or observation[1] < previous[1]
            or observation[1] - previous[1] > 5_000_000_000
        )
        if not segments or discontinuity:
            segments.append(([observation], bool(segments)))
        else:
            segments[-1][0].append(observation)
    return segments


def _fit_robust_affine(
    observations: Sequence[tuple[int, int]],
    nominal_scale: float,
) -> _AffineFit:
    initial_scale, initial_offset = _fit_initial_robust(observations, nominal_scale)
    residuals = [
        target - (initial_scale * source + initial_offset)
        for source, target in observations
    ]
    median_residual = statistics.median(residuals)
    mad = statistics.median(abs(value - median_residual) for value in residuals)
    threshold = max(5_000_000.0, 6.0 * 1.4826 * mad)
    inliers = [
        observation
        for observation, residual in zip(observations, residuals, strict=True)
        if abs(residual - median_residual) <= threshold
    ]
    if not inliers:
        inliers = list(observations)
    scale, offset = _fit_affine(inliers, nominal_scale)
    final_residuals = sorted(
        abs(target - (scale * source + offset)) for source, target in inliers
    )
    p95_index = max(0, math.ceil(0.95 * len(final_residuals)) - 1)
    p95 = round(final_residuals[p95_index])
    maximum = round(final_residuals[-1])
    quantization_uncertainty = 500_000 if nominal_scale == 1_000_000.0 else 0
    uncertainty = max(
        p95,
        quantization_uncertainty,
        10_000_000 if len(inliers) < 3 else 0,
    )
    scale_ratio = scale / nominal_scale
    rejected = len(inliers) >= 2 and not 0.95 <= scale_ratio <= 1.05
    return _AffineFit(
        scale=scale,
        offset=offset,
        uncertainty_ns=uncertainty,
        residual_p95_ns=p95,
        residual_max_ns=maximum,
        inlier_count=len(inliers),
        outlier_count=len(observations) - len(inliers),
        rejected=rejected,
    )


def _fit_affine(
    observations: Sequence[tuple[int, int]],
    default_scale: float,
) -> tuple[float, float]:
    source_origin, target_origin = observations[0]
    source_offsets = [source - source_origin for source, _target in observations]
    target_offsets = [target - target_origin for _source, target in observations]
    denominator = sum(offset * offset for offset in source_offsets)
    if denominator == 0:
        scale = default_scale
    else:
        scale = sum(
            source * target
            for source, target in zip(source_offsets, target_offsets, strict=True)
        ) / denominator
    return scale, float(target_origin) - scale * float(source_origin)


def _fit_initial_robust(
    observations: Sequence[tuple[int, int]],
    default_scale: float,
) -> tuple[float, float]:
    slopes = [
        (right_target - left_target) / (right_source - left_source)
        for (left_source, left_target), (right_source, right_target) in zip(
            observations,
            observations[1:],
            strict=False,
        )
        if right_source > left_source
    ]
    scale = statistics.median(slopes) if slopes else default_scale
    offsets = [target - scale * source for source, target in observations]
    return scale, float(statistics.median(offsets))


def _find_mapping(
    mappings: Sequence[dict[str, object]],
    connection_session_id: str,
    source_instance_id: str,
    source_clock_id: str,
    source_time: int,
) -> dict[str, object] | None:
    for mapping in mappings:
        if (
            mapping["connection"] == connection_session_id
            and mapping["source_instance"] == source_instance_id
            and mapping["source_clock"] == source_clock_id
            and int(mapping["source_from"]) <= source_time <= int(mapping["source_to"])
        ):
            return mapping
    return None


def _sequence_quality(
    rows: Iterable[tuple[str, str, int]],
) -> tuple[int, int, int]:
    previous: dict[tuple[str, str], int] = {}
    gaps = 0
    duplicates = 0
    out_of_order = 0
    for connection_session_id, sensor_type, sequence_number in rows:
        key = (connection_session_id, sensor_type)
        last = previous.get(key)
        if last is None:
            previous[key] = sequence_number
        elif sequence_number == last:
            duplicates += 1
        elif sequence_number < last:
            out_of_order += 1
        else:
            gaps += max(0, sequence_number - last - 1)
            previous[key] = sequence_number
    return gaps, duplicates, out_of_order
