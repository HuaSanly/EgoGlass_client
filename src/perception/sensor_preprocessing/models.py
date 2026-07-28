"""传感器预处理阶段使用的基础数据模型。

本模块只定义数据及其不变量，不读取文件、不执行时间拟合，也不运行感知算法。
"""

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from math import isfinite
from pathlib import Path


class TimeStatus(StrEnum):
    """时间戳换算结果的可信状态。

    ``VERIFIED`` 表示映射关系已经通过设备文档或实机测试验证；
    ``ESTIMATED`` 表示时间来自拟合或回调时间等近似方法；
    ``UNAVAILABLE`` 表示当前没有足够证据换算到统一时间轴。
    """

    VERIFIED = "verified"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class TimeEstimate:
    """一个原始时间戳及其到会话时间轴的换算结果。
    Args:
        source_clock_id: 原始时钟名称，名称中应体现时钟来源和单位。
        source_timestamp: 原始时间戳整数，不在此处修改或覆盖。
        status: 本次时间换算的可信状态。
        session_time_ns: 换算后的会话单调时间，单位为纳秒；不可用时为 ``None``。
        uncertainty_ns: 换算误差的保守上界，单位为纳秒；不可用时为 ``None``。
        clock_mapping_id: 所用时钟映射的唯一标识；不可用时为 ``None``。
    该对象不可变，保证后续阶段不能意外覆盖原始时间或换算结果。
    """

    source_clock_id: str
    source_timestamp: int
    status: TimeStatus
    session_time_ns: int | None = None
    uncertainty_ns: int | None = None
    clock_mapping_id: str | None = None

    def __post_init__(self) -> None:
        """在数据类创建后检查时间字段是否自洽。
        Inputs:
            不接收额外参数，检查当前实例中的全部字段。
        Returns:
            ``None``。检查通过后对象即可使用。
        Raises:
            ValueError: 原始时钟为空、时间为负数，或者状态与派生字段不匹配。
        """

        if not self.source_clock_id.strip():
            raise ValueError("source_clock_id cannot be empty")
        if self.source_timestamp < 0:
            raise ValueError("source_timestamp cannot be negative")
        if not isinstance(self.status, TimeStatus):
            raise TypeError("status must be a TimeStatus")

        derived = (
            self.session_time_ns,
            self.uncertainty_ns,
            self.clock_mapping_id,
        )
        if self.status is TimeStatus.UNAVAILABLE:
            if any(value is not None for value in derived):
                raise ValueError("unavailable time cannot contain derived values")
            return

        if (
            self.session_time_ns is None
            or self.uncertainty_ns is None
            or self.clock_mapping_id is None
        ):
            raise ValueError("resolved time requires time, uncertainty, and mapping id")
        if not self.clock_mapping_id.strip():
            raise ValueError("clock_mapping_id cannot be empty")
        if self.session_time_ns < 0 or self.uncertainty_ns < 0:
            raise ValueError("resolved time values cannot be negative")


@dataclass(frozen=True, slots=True)
class Mp4Timestamp:
    """使用整数 PTS 和有理数 time base 表示一个 MP4 展示时间。
    Args:
        pts: MP4 帧的 Presentation Timestamp，可用于精确定位解码帧。
        time_base_numerator: time base 的分子，必须大于零。
        time_base_denominator: time base 的分母，必须大于零。
    不把时间立即转换为浮点数，避免长视频中的累计精度损失。
    """

    pts: int
    time_base_numerator: int
    time_base_denominator: int

    def __post_init__(self) -> None:
        """检查 MP4 time base 是否能够构成合法的正有理数。
        Inputs:
            不接收额外参数，读取当前实例中的 time base 分子和分母。
        Returns:
            ``None``。
        Raises:
            ValueError: time base 的分子或分母小于等于零。
        """

        if self.time_base_numerator <= 0:
            raise ValueError("time-base numerator must be positive")
        if self.time_base_denominator <= 0:
            raise ValueError("time-base denominator must be positive")

    @property
    def presentation_time_seconds(self) -> Fraction:
        """计算该 PTS 在视频时间轴上的精确秒数。
        Returns:
            ``Fraction``：值为 ``pts * numerator / denominator``，不会产生浮点误差。
        """

        return Fraction(
            self.pts * self.time_base_numerator,
            self.time_base_denominator,
        )


class MetadataMatchStatus(StrEnum):
    """MP4 帧和眼镜端相机元数据之间的匹配结果。"""

    EXACT = "exact"
    WITHIN_TOLERANCE = "within_tolerance"
    UNMATCHED = "unmatched"


class ImuSensorType(StrEnum):
    """当前预处理契约支持的 IMU 传感器类型。"""

    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"


class AlignmentStatus(StrEnum):
    """采集数据库中时间对齐字段的状态。"""

    PENDING = "pending"
    MAPPED = "mapped"


@dataclass(frozen=True, slots=True)
class StoredAlignment:
    """采集数据库中已有的时间对齐结果，不在读取阶段重新计算。"""

    status: AlignmentStatus
    session_time_ns: int | None
    uncertainty_ns: int | None
    clock_mapping_segment_id: str | None

    def __post_init__(self) -> None:
        """检查原始对齐字段是全部缺失或全部存在，禁止部分结果。"""

        if not isinstance(self.status, AlignmentStatus):
            raise TypeError("status must be an AlignmentStatus")
        derived = (
            self.session_time_ns,
            self.uncertainty_ns,
            self.clock_mapping_segment_id,
        )
        if self.status is AlignmentStatus.PENDING and any(
            value is not None for value in derived
        ):
            raise ValueError("pending alignment cannot contain mapped fields")
        if self.status is AlignmentStatus.MAPPED and any(value is None for value in derived):
            raise ValueError("mapped alignment requires all mapped fields")
        if self.session_time_ns is not None and self.session_time_ns < 0:
            raise ValueError("stored session time cannot be negative")
        if self.uncertainty_ns is not None and self.uncertainty_ns < 0:
            raise ValueError("stored timestamp uncertainty cannot be negative")
        if (
            self.clock_mapping_segment_id is not None
            and not self.clock_mapping_segment_id.strip()
        ):
            raise ValueError("clock mapping segment id cannot be empty")


@dataclass(frozen=True, slots=True)
class RawFrameRef:
    """一个 MP4 帧及其原始相机元数据的不可变引用。"""

    video_frame_row_id: int
    session_id: str
    clip_id: str
    frame_index: int
    media_path: Path
    mp4_timestamp: Mp4Timestamp

    metadata_match_status: MetadataMatchStatus
    video_frame_metadata_id: str | None
    frame_metadata_match_id: int | None
    timestamp_match_error_90khz: int | None
    connection_session_id: str | None
    camera_start_generation: int | None
    frame_id: int | None

    captured_at_rokid_sdk_ms: int | None
    received_at_elapsed_realtime_ns: int | None
    video_at_monotonic_ns: int | None
    rtp_timestamp_90khz: int | None

    received_at_client_perf_counter_ns: int
    metadata_received_at_client_perf_counter_ns: int | None
    width: int | None
    height: int | None
    rotation_degrees: int | None
    capture_config_id: str | None

    source_frame_timestamp: Mp4Timestamp | None
    stored_alignment: StoredAlignment

    def __post_init__(self) -> None:
        """验证帧索引、媒体路径以及 matched/unmatched 字段的一致性。"""

        if self.video_frame_row_id < 1:
            raise ValueError("video frame row id must be positive")
        if not self.session_id.strip() or not self.clip_id.strip():
            raise ValueError("session and clip ids cannot be empty")
        if self.frame_index < 0:
            raise ValueError("frame index cannot be negative")
        if not self.media_path.is_absolute():
            raise ValueError("media path must be absolute")
        if self.received_at_client_perf_counter_ns < 0:
            raise ValueError("client receipt time cannot be negative")
        if not isinstance(self.metadata_match_status, MetadataMatchStatus):
            raise TypeError("metadata_match_status must be a MetadataMatchStatus")

        metadata_fields = (
            self.video_frame_metadata_id,
            self.frame_metadata_match_id,
            self.timestamp_match_error_90khz,
            self.connection_session_id,
            self.camera_start_generation,
            self.frame_id,
            self.captured_at_rokid_sdk_ms,
            self.received_at_elapsed_realtime_ns,
            self.video_at_monotonic_ns,
            self.rtp_timestamp_90khz,
            self.metadata_received_at_client_perf_counter_ns,
            self.width,
            self.height,
            self.rotation_degrees,
            self.capture_config_id,
        )
        if self.metadata_match_status is MetadataMatchStatus.UNMATCHED:
            if any(value is not None for value in metadata_fields):
                raise ValueError("unmatched frame cannot contain camera metadata")
            return
        if any(value is None for value in metadata_fields):
            raise ValueError("matched frame requires complete camera metadata")

        assert self.video_frame_metadata_id is not None
        assert self.frame_metadata_match_id is not None
        assert self.timestamp_match_error_90khz is not None
        assert self.connection_session_id is not None
        assert self.camera_start_generation is not None
        assert self.frame_id is not None
        assert self.captured_at_rokid_sdk_ms is not None
        assert self.received_at_elapsed_realtime_ns is not None
        assert self.video_at_monotonic_ns is not None
        assert self.rtp_timestamp_90khz is not None
        assert self.metadata_received_at_client_perf_counter_ns is not None
        assert self.width is not None
        assert self.height is not None
        assert self.rotation_degrees is not None
        assert self.capture_config_id is not None
        if not self.video_frame_metadata_id.strip() or not self.connection_session_id.strip():
            raise ValueError("matched frame metadata ids cannot be empty")
        if self.frame_metadata_match_id < 1:
            raise ValueError("frame metadata match id must be positive")
        if self.metadata_match_status is MetadataMatchStatus.EXACT:
            if self.timestamp_match_error_90khz != 0:
                raise ValueError("exact metadata match requires zero timestamp error")
        elif self.timestamp_match_error_90khz <= 0:
            raise ValueError("within-tolerance metadata match requires positive timestamp error")
        if not self.capture_config_id.strip():
            raise ValueError("capture config id cannot be empty")
        if self.camera_start_generation < 1 or self.frame_id < 0:
            raise ValueError("camera generation and frame id are invalid")
        if min(
            self.captured_at_rokid_sdk_ms,
            self.received_at_elapsed_realtime_ns,
            self.video_at_monotonic_ns,
            self.metadata_received_at_client_perf_counter_ns,
        ) < 0:
            raise ValueError("camera timestamps cannot be negative")
        if not 0 <= self.rtp_timestamp_90khz <= 0xFFFFFFFF:
            raise ValueError("RTP timestamp is outside uint32 range")
        if self.width < 1 or self.height < 1:
            raise ValueError("frame dimensions must be positive")
        if self.rotation_degrees not in {0, 90, 180, 270}:
            raise ValueError("frame rotation must be 0, 90, 180, or 270 degrees")


@dataclass(frozen=True, slots=True)
class RawImuSample:
    """一条保持原始数值、单位、顺序和时钟的 IMU 样本。"""

    sample_id: int
    session_id: str
    connection_session_id: str
    sensor_type: ImuSensorType
    android_sensor_type: int
    sequence_number: int

    sensor_event_monotonic_ns: int
    received_at_elapsed_realtime_ns: int
    received_at_client_perf_counter_ns: int

    accuracy: int
    values: tuple[float, float, float]
    unit: str
    stored_alignment: StoredAlignment

    def __post_init__(self) -> None:
        """验证 IMU 类型、Android 类型、单位和三轴数值。"""

        if self.sample_id < 1:
            raise ValueError("sample id must be positive")
        if not self.session_id.strip() or not self.connection_session_id.strip():
            raise ValueError("session and connection ids cannot be empty")
        if not isinstance(self.sensor_type, ImuSensorType):
            raise TypeError("sensor_type must be an ImuSensorType")
        if self.sequence_number < 0:
            raise ValueError("sequence number cannot be negative")
        if min(
            self.sensor_event_monotonic_ns,
            self.received_at_elapsed_realtime_ns,
            self.received_at_client_perf_counter_ns,
        ) < 0:
            raise ValueError("IMU timestamps cannot be negative")
        if not -1 <= self.accuracy <= 3:
            raise ValueError("IMU accuracy must be between -1 and 3")
        if not isinstance(self.values, tuple) or len(self.values) != 3:
            raise TypeError("IMU values must be an immutable three-element tuple")
        if not all(isfinite(value) for value in self.values):
            raise ValueError("IMU values must be finite")

        expected_android_type, expected_unit = {
            ImuSensorType.ACCELEROMETER: (1, "m_s2"),
            ImuSensorType.GYROSCOPE: (4, "rad_s"),
        }[self.sensor_type]
        if self.android_sensor_type != expected_android_type or self.unit != expected_unit:
            raise ValueError("IMU sensor type, Android type, and unit do not match")


@dataclass(frozen=True, slots=True)
class CaptureClipRef:
    """一个经过 manifest 和文件校验的完整 MP4 clip。"""

    clip_id: str
    media_path: Path
    frame_count: int
    sha256: str
    width: int
    height: int
    nominal_fps: float


@dataclass(frozen=True, slots=True)
class CaptureSessionRef:
    """预处理 reader 已验证的完整采集会话入口。"""

    session_id: str
    session_directory: Path
    telemetry_database_path: Path
    clips: tuple[CaptureClipRef, ...]
