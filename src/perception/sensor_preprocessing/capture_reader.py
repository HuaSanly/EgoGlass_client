"""只读加载已完成的 capture-session-v1 会话。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .models import (
    AlignmentStatus,
    CaptureClipRef,
    CaptureSessionRef,
    ImuSensorType,
    MetadataMatchStatus,
    Mp4Timestamp,
    RawFrameRef,
    RawImuSample,
    StoredAlignment,
)

_TELEMETRY_SCHEMA_VERSION = "1"
_HASH_CHUNK_BYTES = 1024 * 1024


class CaptureSessionReadError(ValueError):
    """采集会话缺失、损坏或不符合预处理输入契约。"""


class _LifecycleView(BaseModel):
    """Manifest 生命周期视图。"""

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    state: Literal["active", "finalizing", "complete", "incomplete"]


class _StorageView(BaseModel):
    """Manifest 存储路径视图。"""

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    telemetry_database_path: str = "telemetry/telemetry.sqlite"


class _VideoProfileView(BaseModel):
    """完整 clip 的视频参数视图。"""

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    container: Literal["mp4"]
    codec: Literal["h264"]
    width: int = Field(ge=1, le=8192)
    height: int = Field(ge=1, le=8192)
    nominal_fps: float = Field(gt=0, le=240)


class _ClipView(BaseModel):
    """Manifest clip 字段视图。"""

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    clip_id: str = Field(min_length=1)
    state: Literal["preparing", "recording", "complete", "incomplete", "cancelled"]
    relative_media_path: str = Field(min_length=1)
    video_profile: _VideoProfileView
    frame_count: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class _ManifestView(BaseModel):
    """预处理阶段需要的 capture-session-v1 字段子集。"""

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    schema_version: Literal["1.0"]
    contract_id: Literal["capture-session-v1"]
    session_id: str = Field(min_length=1)
    lifecycle: _LifecycleView
    storage: _StorageView = Field(default_factory=_StorageView)
    clips: list[_ClipView]


class _AlignmentRowView(BaseModel):
    """Telemetry 行中已经存储的时间对齐字段。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    alignment_status: Literal["pending", "mapped"]
    session_time_ns: int | None
    timestamp_uncertainty_ns: int | None
    clock_mapping_segment_id: str | None


class _FrameRowView(_AlignmentRowView):
    """帧索引 JOIN 结果的严格类型视图，拒绝 SQLite 的隐式类型漂移。"""

    video_frame_row_id: int
    session_id: str
    clip_id: str
    frame_index: int
    mp4_pts: int
    mp4_time_base_numerator: int
    mp4_time_base_denominator: int
    video_frame_metadata_id: str | None
    frame_metadata_match_id: int | None
    timestamp_match_error_90khz: int | None
    indexed_frame_id: int | None
    indexed_captured_at_rokid_sdk_ms: int | None
    indexed_received_at_elapsed_realtime_ns: int | None
    indexed_video_at_monotonic_ns: int | None
    indexed_rtp_timestamp_90khz: int | None
    received_at_client_perf_counter_ns: int
    metadata_match_status: Literal["exact", "within_tolerance", "unmatched"]
    source_frame_pts: int | None
    source_frame_time_base_num: int | None
    source_frame_time_base_den: int | None
    metadata_session_id: str | None
    metadata_ingest_status: Literal["accepted"] | None
    connection_session_id: str | None
    camera_start_generation: int | None
    metadata_frame_id: int | None
    metadata_captured_at_rokid_sdk_ms: int | None
    metadata_received_at_elapsed_realtime_ns: int | None
    metadata_video_at_monotonic_ns: int | None
    metadata_rtp_timestamp_90khz: int | None
    metadata_received_at_client_perf_counter_ns: int | None
    width: int | None
    height: int | None
    rotation_degrees: int | None
    capture_config_id: str | None
    match_connection_session_id: str | None
    match_camera_start_generation: int | None
    match_frame_id: int | None
    match_captured_at_rokid_sdk_ms: int | None
    match_received_at_elapsed_realtime_ns: int | None
    match_video_at_monotonic_ns: int | None
    match_rtp_timestamp_90khz: int | None
    match_width: int | None
    match_height: int | None
    match_rotation_degrees: int | None
    match_capture_config_id: str | None
    match_received_at_client_perf_counter_ns: int | None


class _ImuRowView(_AlignmentRowView):
    """IMU 表记录的严格类型视图。"""

    sample_id: int
    session_id: str
    connection_session_id: str
    sensor_type: Literal["accelerometer", "gyroscope"]
    android_sensor_type: int
    sequence_number: int
    sensor_event_monotonic_ns: int
    received_at_elapsed_realtime_ns: int
    received_at_client_perf_counter_ns: int
    accuracy: int
    value_x: float
    value_y: float
    value_z: float
    unit: str


@dataclass(frozen=True, slots=True)
class CaptureSessionReader:
    """从已验证会话中流式读取原始帧索引和 IMU 样本。"""

    session: CaptureSessionRef

    @classmethod
    def open(
        cls,
        session_directory: str | Path,
        *,
        verify_media_hashes: bool = True,
    ) -> CaptureSessionReader:
        """验证会话 manifest、媒体文件和 telemetry 数据库并创建 reader。

        Args:
            session_directory: 包含 ``session.json`` 的会话目录。
            verify_media_hashes: 是否计算完整 MP4 的 SHA256 并与 manifest 比较。

        Returns:
            可以重复创建只读迭代器的 ``CaptureSessionReader``。

        Raises:
            CaptureSessionReadError: 输入不是完整、可验证的 capture-session-v1。
        """

        directory = _resolve_session_directory(session_directory)
        manifest = _read_manifest(directory / "session.json")
        if manifest.lifecycle.state != "complete":
            raise CaptureSessionReadError("capture session must be complete")
        if directory.name != manifest.session_id:
            raise CaptureSessionReadError("session directory name does not match session_id")

        database_path = _resolve_session_file(
            directory,
            manifest.storage.telemetry_database_path,
            description="telemetry database",
        )
        complete_clips = tuple(clip for clip in manifest.clips if clip.state == "complete")
        if not complete_clips:
            raise CaptureSessionReadError("capture session has no complete clips")
        if len({clip.clip_id for clip in complete_clips}) != len(complete_clips):
            raise CaptureSessionReadError("capture session contains duplicate clip ids")

        clips: list[CaptureClipRef] = []
        for clip in complete_clips:
            if clip.frame_count is None or clip.frame_count < 1 or clip.sha256 is None:
                raise CaptureSessionReadError(
                    f"complete clip {clip.clip_id!r} requires frame_count and sha256"
                )
            media_path = _resolve_session_file(
                directory,
                clip.relative_media_path,
                description=f"media for clip {clip.clip_id}",
            )
            if verify_media_hashes and _sha256_file(media_path) != clip.sha256:
                raise CaptureSessionReadError(f"media hash mismatch for clip {clip.clip_id!r}")
            clips.append(
                CaptureClipRef(
                    clip_id=clip.clip_id,
                    media_path=media_path,
                    frame_count=clip.frame_count,
                    sha256=clip.sha256,
                    width=clip.video_profile.width,
                    height=clip.video_profile.height,
                    nominal_fps=clip.video_profile.nominal_fps,
                )
            )

        session = CaptureSessionRef(
            session_id=manifest.session_id,
            session_directory=directory,
            telemetry_database_path=database_path,
            clips=tuple(clips),
        )
        _validate_database(session)
        return cls(session=session)

    def iter_frames(self, clip_id: str) -> Iterator[RawFrameRef]:
        """按 ``frame_index`` 顺序流式读取指定 clip 的帧引用。"""

        clip = self._require_clip(clip_id)
        connection = _open_read_only(self.session.telemetry_database_path)
        try:
            rows = connection.execute(
                """
                SELECT
                    frame.video_frame_row_id,
                    frame.session_id,
                    frame.clip_id,
                    frame.frame_index,
                    frame.mp4_pts,
                    frame.mp4_time_base_numerator,
                    frame.mp4_time_base_denominator,
                    frame.video_frame_metadata_id,
                    frame.frame_metadata_match_id,
                    frame.frame_id AS indexed_frame_id,
                    frame.captured_at_rokid_sdk_ms AS indexed_captured_at_rokid_sdk_ms,
                    frame.received_at_elapsed_realtime_ns
                        AS indexed_received_at_elapsed_realtime_ns,
                    frame.video_at_monotonic_ns AS indexed_video_at_monotonic_ns,
                    frame.rtp_timestamp_90khz AS indexed_rtp_timestamp_90khz,
                    frame.received_at_client_perf_counter_ns,
                    frame.alignment_status,
                    frame.session_time_ns,
                    frame.timestamp_uncertainty_ns,
                    frame.clock_mapping_segment_id,
                    frame.metadata_match_status,
                    frame.source_frame_pts,
                    frame.source_frame_time_base_num,
                    frame.source_frame_time_base_den,
                    metadata.session_id AS metadata_session_id,
                    metadata.ingest_status AS metadata_ingest_status,
                    metadata.connection_session_id,
                    metadata.camera_start_generation,
                    metadata.frame_id AS metadata_frame_id,
                    metadata.captured_at_rokid_sdk_ms
                        AS metadata_captured_at_rokid_sdk_ms,
                    metadata.received_at_elapsed_realtime_ns
                        AS metadata_received_at_elapsed_realtime_ns,
                    metadata.video_at_monotonic_ns AS metadata_video_at_monotonic_ns,
                    metadata.rtp_timestamp_90khz AS metadata_rtp_timestamp_90khz,
                    metadata.received_at_client_perf_counter_ns
                        AS metadata_received_at_client_perf_counter_ns,
                    metadata.width,
                    metadata.height,
                    metadata.rotation_degrees,
                    metadata.capture_config_id,
                    match.connection_session_id AS match_connection_session_id,
                    match.camera_start_generation AS match_camera_start_generation,
                    match.frame_id AS match_frame_id,
                    match.captured_at_rokid_sdk_ms AS match_captured_at_rokid_sdk_ms,
                    match.received_at_elapsed_realtime_ns
                        AS match_received_at_elapsed_realtime_ns,
                    match.video_at_monotonic_ns AS match_video_at_monotonic_ns,
                    match.rtp_timestamp_90khz AS match_rtp_timestamp_90khz,
                    match.width AS match_width,
                    match.height AS match_height,
                    match.rotation_degrees AS match_rotation_degrees,
                    match.capture_config_id AS match_capture_config_id,
                    match.received_at_client_perf_counter_ns
                        AS match_received_at_client_perf_counter_ns,
                    match.timestamp_match_error_90khz
                FROM video_frame_index AS frame
                LEFT JOIN video_frame_metadata_raw AS metadata
                    ON metadata.video_frame_metadata_id = frame.video_frame_metadata_id
                LEFT JOIN video_frame_matches AS match
                    ON match.match_id = frame.frame_metadata_match_id
                WHERE frame.clip_id = ?
                ORDER BY frame.frame_index
                """,
                (clip_id,),
            )
            for row in rows:
                frame = _frame_from_row(row, clip.media_path)
                if frame.session_id != self.session.session_id:
                    raise CaptureSessionReadError("frame session_id does not match manifest")
                if frame.clip_id != clip_id:
                    raise CaptureSessionReadError("frame clip_id does not match requested clip")
                yield frame
        except sqlite3.Error as exc:
            raise CaptureSessionReadError("failed to read video frame index") from exc
        finally:
            connection.close()

    def iter_imu_samples(self) -> Iterator[RawImuSample]:
        """按数据库 ``sample_id`` 顺序流式读取原始 IMU，保留乱序证据。"""

        connection = _open_read_only(self.session.telemetry_database_path)
        try:
            rows = connection.execute(
                """
                SELECT
                    sample_id,
                    session_id,
                    connection_session_id,
                    sensor_type,
                    android_sensor_type,
                    sequence_number,
                    sensor_event_monotonic_ns,
                    received_at_elapsed_realtime_ns,
                    received_at_client_perf_counter_ns,
                    alignment_status,
                    session_time_ns,
                    timestamp_uncertainty_ns,
                    clock_mapping_segment_id,
                    accuracy,
                    value_x,
                    value_y,
                    value_z,
                    unit
                FROM imu_samples
                ORDER BY sample_id
                """
            )
            for row in rows:
                sample = _imu_from_row(row)
                if sample.session_id != self.session.session_id:
                    raise CaptureSessionReadError("IMU session_id does not match manifest")
                yield sample
        except sqlite3.Error as exc:
            raise CaptureSessionReadError("failed to read IMU samples") from exc
        finally:
            connection.close()

    def _require_clip(self, clip_id: str) -> CaptureClipRef:
        """返回已验证 clip；未知 ID 立即失败，避免静默返回空序列。"""

        for clip in self.session.clips:
            if clip.clip_id == clip_id:
                return clip
        raise KeyError(f"unknown complete clip {clip_id!r}")


def _resolve_session_directory(session_directory: str | Path) -> Path:
    """解析并验证会话根目录。"""

    try:
        directory = Path(session_directory).resolve(strict=True)
    except OSError as exc:
        raise CaptureSessionReadError("capture session directory does not exist") from exc
    if not directory.is_dir():
        raise CaptureSessionReadError("capture session path is not a directory")
    return directory


def _read_manifest(manifest_path: Path) -> _ManifestView:
    """读取 JSON 并验证预处理所需的 manifest 字段。"""

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return _ManifestView.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise CaptureSessionReadError("invalid capture-session-v1 manifest") from exc


def _resolve_session_file(root: Path, relative_path: str, *, description: str) -> Path:
    """解析 manifest 相对路径并阻止绝对路径、目录穿越和外部符号链接。"""

    path = Path(relative_path)
    if path.is_absolute():
        raise CaptureSessionReadError(f"{description} path must be relative")
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root):
        raise CaptureSessionReadError(f"{description} path escapes the session directory")
    if not candidate.is_file():
        raise CaptureSessionReadError(f"{description} file does not exist")
    return candidate


def _sha256_file(path: Path) -> str:
    """分块计算文件 SHA256，避免把大型 MP4 一次载入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_uncheckpointed_database(database_path: Path) -> None:
    """拒绝 immutable 模式会忽略的非空 WAL 或 rollback journal。"""

    for suffix in ("-wal", "-journal"):
        sidecar = database_path.with_name(f"{database_path.name}{suffix}")
        try:
            if sidecar.is_file() and sidecar.stat().st_size > 0:
                raise CaptureSessionReadError(
                    f"telemetry database has uncheckpointed {suffix[1:]} data"
                )
        except OSError as exc:
            raise CaptureSessionReadError("cannot inspect telemetry database sidecars") from exc


def _open_read_only(database_path: Path) -> sqlite3.Connection:
    """以 SQLite 只读不可变模式打开数据库，且不在采集目录创建 WAL/SHM。"""

    _reject_uncheckpointed_database(database_path)
    try:
        connection = sqlite3.connect(
            f"{database_path.as_uri()}?mode=ro&immutable=1",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection
    except sqlite3.Error as exc:
        raise CaptureSessionReadError("cannot open telemetry database read-only") from exc


def _validate_database(session: CaptureSessionRef) -> None:
    """验证 telemetry 身份、schema 版本以及每个完整 clip 的帧数。"""

    try:
        with closing(_open_read_only(session.telemetry_database_path)) as connection:
            metadata = dict(connection.execute("SELECT key, value FROM session_metadata"))
            if metadata.get("telemetry_schema_version") != _TELEMETRY_SCHEMA_VERSION:
                raise CaptureSessionReadError("unsupported telemetry schema version")
            if metadata.get("session_id") != session.session_id:
                raise CaptureSessionReadError("telemetry session_id does not match manifest")
            index_stats = {
                row[0]: (row[1], row[2], row[3], row[4])
                for row in connection.execute(
                    """
                    SELECT clip_id, COUNT(*), COUNT(DISTINCT frame_index),
                           MIN(frame_index), MAX(frame_index)
                    FROM video_frame_index
                    GROUP BY clip_id
                    """
                )
            }
            for clip in session.clips:
                count, distinct_count, first_index, last_index = index_stats.get(
                    clip.clip_id,
                    (0, 0, None, None),
                )
                if count != clip.frame_count:
                    raise CaptureSessionReadError(
                        f"frame index count does not match manifest for clip {clip.clip_id!r}"
                    )
                if (
                    distinct_count != clip.frame_count
                    or first_index != 0
                    or last_index != clip.frame_count - 1
                ):
                    raise CaptureSessionReadError(
                        f"frame indices are not contiguous for clip {clip.clip_id!r}"
                    )
    except sqlite3.Error as exc:
        raise CaptureSessionReadError("invalid telemetry database schema") from exc


def _stored_alignment_from_row(row: _AlignmentRowView) -> StoredAlignment:
    """从 SQLite row 保留采集时已有的对齐字段。"""

    return StoredAlignment(
        status=AlignmentStatus(row.alignment_status),
        session_time_ns=row.session_time_ns,
        uncertainty_ns=row.timestamp_uncertainty_ns,
        clock_mapping_segment_id=row.clock_mapping_segment_id,
    )


def _optional_source_timestamp(row: _FrameRowView) -> Mp4Timestamp | None:
    """把全空 source PTS 转为 None，并拒绝部分缺失的时间基。"""

    values = (
        row.source_frame_pts,
        row.source_frame_time_base_num,
        row.source_frame_time_base_den,
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise CaptureSessionReadError("source frame timestamp is partially missing")
    return Mp4Timestamp(
        pts=values[0],
        time_base_numerator=values[1],
        time_base_denominator=values[2],
    )


def _frame_from_row(row: sqlite3.Row, media_path: Path) -> RawFrameRef:
    """把一行帧索引转换为不可变原始帧引用。"""

    try:
        data = _FrameRowView.model_validate(dict(row))
        _validate_frame_metadata_consistency(data)
        return RawFrameRef(
            video_frame_row_id=data.video_frame_row_id,
            session_id=data.session_id,
            clip_id=data.clip_id,
            frame_index=data.frame_index,
            media_path=media_path,
            mp4_timestamp=Mp4Timestamp(
                pts=data.mp4_pts,
                time_base_numerator=data.mp4_time_base_numerator,
                time_base_denominator=data.mp4_time_base_denominator,
            ),
            metadata_match_status=MetadataMatchStatus(data.metadata_match_status),
            video_frame_metadata_id=data.video_frame_metadata_id,
            frame_metadata_match_id=data.frame_metadata_match_id,
            timestamp_match_error_90khz=data.timestamp_match_error_90khz,
            connection_session_id=data.connection_session_id,
            camera_start_generation=data.camera_start_generation,
            frame_id=data.indexed_frame_id,
            captured_at_rokid_sdk_ms=data.indexed_captured_at_rokid_sdk_ms,
            received_at_elapsed_realtime_ns=(
                data.indexed_received_at_elapsed_realtime_ns
            ),
            video_at_monotonic_ns=data.indexed_video_at_monotonic_ns,
            rtp_timestamp_90khz=data.indexed_rtp_timestamp_90khz,
            received_at_client_perf_counter_ns=data.received_at_client_perf_counter_ns,
            metadata_received_at_client_perf_counter_ns=(
                data.metadata_received_at_client_perf_counter_ns
            ),
            width=data.width,
            height=data.height,
            rotation_degrees=data.rotation_degrees,
            capture_config_id=data.capture_config_id,
            source_frame_timestamp=_optional_source_timestamp(data),
            stored_alignment=_stored_alignment_from_row(data),
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise CaptureSessionReadError("invalid video frame index row") from exc


def _validate_frame_metadata_consistency(row: _FrameRowView) -> None:
    """核对帧索引中的冗余相机字段与原始相机元数据，拒绝损坏的 JOIN。"""

    if row.video_frame_metadata_id is None:
        return
    if row.metadata_session_id != row.session_id:
        raise ValueError("frame metadata session does not match frame session")
    if row.metadata_ingest_status != "accepted":
        raise ValueError("frame metadata was not accepted by ingest")
    if row.connection_session_id != row.match_connection_session_id:
        raise ValueError("frame metadata connection does not match frame match")
    duplicated_values = (
        (row.indexed_frame_id, row.metadata_frame_id, row.match_frame_id),
        (row.camera_start_generation, row.match_camera_start_generation),
        (
            row.indexed_captured_at_rokid_sdk_ms,
            row.metadata_captured_at_rokid_sdk_ms,
            row.match_captured_at_rokid_sdk_ms,
        ),
        (
            row.indexed_received_at_elapsed_realtime_ns,
            row.metadata_received_at_elapsed_realtime_ns,
            row.match_received_at_elapsed_realtime_ns,
        ),
        (
            row.indexed_video_at_monotonic_ns,
            row.metadata_video_at_monotonic_ns,
            row.match_video_at_monotonic_ns,
        ),
        (
            row.indexed_rtp_timestamp_90khz,
            row.metadata_rtp_timestamp_90khz,
            row.match_rtp_timestamp_90khz,
        ),
        (row.width, row.match_width),
        (row.height, row.match_height),
        (row.rotation_degrees, row.match_rotation_degrees),
        (row.capture_config_id, row.match_capture_config_id),
    )
    if any(len(set(values)) != 1 for values in duplicated_values):
        raise ValueError("frame metadata does not match frame index")
    if (
        row.match_received_at_client_perf_counter_ns is not None
        and row.match_received_at_client_perf_counter_ns
        != row.received_at_client_perf_counter_ns
    ):
        raise ValueError("matched frame receipt time does not match frame index")


def _imu_from_row(row: sqlite3.Row) -> RawImuSample:
    """把一行 IMU 表记录转换为不可变原始样本。"""

    try:
        data = _ImuRowView.model_validate(dict(row))
        return RawImuSample(
            sample_id=data.sample_id,
            session_id=data.session_id,
            connection_session_id=data.connection_session_id,
            sensor_type=ImuSensorType(data.sensor_type),
            android_sensor_type=data.android_sensor_type,
            sequence_number=data.sequence_number,
            sensor_event_monotonic_ns=data.sensor_event_monotonic_ns,
            received_at_elapsed_realtime_ns=data.received_at_elapsed_realtime_ns,
            received_at_client_perf_counter_ns=data.received_at_client_perf_counter_ns,
            accuracy=data.accuracy,
            values=(data.value_x, data.value_y, data.value_z),
            unit=data.unit,
            stored_alignment=_stored_alignment_from_row(data),
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise CaptureSessionReadError("invalid IMU sample row") from exc
