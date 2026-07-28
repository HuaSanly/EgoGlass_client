"""在明确的时钟分段内，把原始时间戳确定性映射到会话时间。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction

from .models import (
    ClockId,
    RawFrameRef,
    RawImuSample,
    TimeEstimate,
    TimeObservation,
    TimestampSemantic,
    TimeStatus,
)

_CLOCK_MAPPING_SCHEMA_VERSION = "1.0"
_RTP_CLOCK_HZ = 90_000
_NS_PER_SECOND = 1_000_000_000


class TimestampOutOfRangeError(ValueError):
    """时间戳不在指定映射分段的有效范围内。"""


@dataclass(frozen=True, slots=True)
class ClockMappingSegment:
    """一个来源时钟实例到会话时间轴的不可变仿射映射分段。

    映射公式为：
    ``target = target_anchor_ns + (source - source_anchor) * scale``。
    scale 使用整数分子/分母保存，禁止浮点舍入随运行环境漂移。
    """

    session_id: str
    source_clock_id: ClockId
    source_instance_id: str
    segment_index: int
    source_from: int
    source_to: int
    source_anchor: int
    target_anchor_ns: int
    scale_numerator_ns: int
    scale_denominator_source_units: int
    uncertainty_ns: int
    status: TimeStatus
    fit_method: str
    provenance_id: str
    uncertainty_basis: str
    evidence_id: str | None = None
    target_clock_id: ClockId = ClockId.SESSION_TIME_NS
    schema_version: str = _CLOCK_MAPPING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """验证映射身份、有效范围、比例、不确定度和输出时间范围。"""

        if self.schema_version != _CLOCK_MAPPING_SCHEMA_VERSION:
            raise ValueError("unsupported clock mapping schema version")
        _validate_required_text(self.session_id, "session_id")
        if not isinstance(self.source_clock_id, ClockId):
            raise TypeError("source_clock_id must be a ClockId")
        if self.source_clock_id is ClockId.SESSION_TIME_NS:
            raise ValueError("source clock must be a raw clock")
        _validate_required_text(self.source_instance_id, "source_instance_id")
        _validate_source_instance_scope(
            self.source_clock_id,
            self.session_id,
            self.source_instance_id,
        )
        if not isinstance(self.target_clock_id, ClockId):
            raise TypeError("target_clock_id must be a ClockId")
        if self.target_clock_id is not ClockId.SESSION_TIME_NS:
            raise ValueError("clock mapping target must be session_time_ns")
        integer_fields = (
            self.segment_index,
            self.source_from,
            self.source_to,
            self.source_anchor,
            self.target_anchor_ns,
            self.scale_numerator_ns,
            self.scale_denominator_source_units,
            self.uncertainty_ns,
        )
        if any(type(value) is not int for value in integer_fields):
            raise TypeError("clock mapping numeric fields must be integers")
        if self.segment_index < 0:
            raise ValueError("segment_index cannot be negative")
        if (
            self.source_from < 0
            and self.source_clock_id is not ClockId.MP4_PRESENTATION_TICKS
        ):
            raise ValueError("source range cannot be negative for this clock")
        if self.source_to < self.source_from:
            raise ValueError("source range is invalid")
        if not self.source_from <= self.source_anchor <= self.source_to:
            raise ValueError("source_anchor must lie inside the source range")
        if self.target_anchor_ns < 0:
            raise ValueError("target_anchor_ns cannot be negative")
        if self.scale_numerator_ns <= 0 or self.scale_denominator_source_units <= 0:
            raise ValueError("clock mapping scale must be positive")
        if self.uncertainty_ns < 0:
            raise ValueError("uncertainty_ns cannot be negative")
        if not isinstance(self.status, TimeStatus):
            raise TypeError("status must be a TimeStatus")
        if self.status not in {TimeStatus.VERIFIED, TimeStatus.ESTIMATED}:
            raise ValueError("clock mapping status must be verified or estimated")
        _validate_required_text(self.fit_method, "fit_method")
        _validate_required_text(self.provenance_id, "provenance_id")
        _validate_required_text(self.uncertainty_basis, "uncertainty_basis")
        if self.evidence_id is not None:
            _validate_required_text(self.evidence_id, "evidence_id")
        if self.status is TimeStatus.VERIFIED and self.evidence_id is None:
            raise ValueError("verified clock mapping requires evidence_id")

        if self._exact_target(self.source_from) < 0:
            raise ValueError("clock mapping produces negative session time")

    @property
    def clock_mapping_id(self) -> str:
        """根据规范化映射参数生成稳定的内容寻址 ID。"""

        payload = {
            "evidence_id": self.evidence_id,
            "fit_method": self.fit_method,
            "provenance_id": self.provenance_id,
            "scale_denominator_source_units": self.scale_denominator_source_units,
            "scale_numerator_ns": self.scale_numerator_ns,
            "schema_version": self.schema_version,
            "segment_index": self.segment_index,
            "session_id": self.session_id,
            "source_anchor": self.source_anchor,
            "source_clock_id": self.source_clock_id.value,
            "source_from": self.source_from,
            "source_instance_id": self.source_instance_id,
            "source_to": self.source_to,
            "status": self.status.value,
            "target_anchor_ns": self.target_anchor_ns,
            "target_clock_id": self.target_clock_id.value,
            "uncertainty_basis": self.uncertainty_basis,
            "uncertainty_ns": self.uncertainty_ns,
        }
        canonical_payload = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
        return f"clock-map-v1-{digest}"

    @property
    def scale(self) -> Fraction:
        """返回每个来源单位对应的精确目标纳秒比例。"""

        return Fraction(
            self.scale_numerator_ns,
            self.scale_denominator_source_units,
        )

    def contains(self, observation: TimeObservation) -> bool:
        """判断原始时间观察值是否属于当前时钟实例和有效范围。"""

        if not isinstance(observation, TimeObservation):
            raise TypeError("observation must be a TimeObservation")
        return (
            observation.session_id == self.session_id
            and observation.source_clock_id is self.source_clock_id
            and observation.source_instance_id == self.source_instance_id
            and self.source_from <= observation.source_timestamp <= self.source_to
        )

    def map(
        self,
        observation: TimeObservation,
        *,
        additional_uncertainty_ns: int = 0,
        additional_status: TimeStatus = TimeStatus.VERIFIED,
    ) -> TimeEstimate:
        """将一个范围内的原始观察值映射到会话时间并传播误差上界。"""

        if not isinstance(observation, TimeObservation):
            raise TypeError("observation must be a TimeObservation")
        if type(additional_uncertainty_ns) is not int:
            raise TypeError("additional uncertainty must be an integer")
        if additional_uncertainty_ns < 0:
            raise ValueError("additional uncertainty cannot be negative")
        if not isinstance(additional_status, TimeStatus):
            raise TypeError("additional_status must be a TimeStatus")
        if not self.contains(observation):
            raise TimestampOutOfRangeError(
                "timestamp does not belong to this clock mapping segment"
            )

        result_status = _weakest_status(self.status, additional_status)
        if result_status is TimeStatus.UNAVAILABLE:
            return _unavailable_estimate(observation)

        exact_target = self._exact_target(observation.source_timestamp)
        session_time_ns = _round_fraction_nearest(exact_target)
        quantization_uncertainty_ns = int(exact_target.denominator != 1)
        return TimeEstimate(
            session_id=observation.session_id,
            source_clock_id=observation.source_clock_id,
            source_instance_id=observation.source_instance_id,
            source_timestamp=observation.source_timestamp,
            timestamp_semantic=observation.timestamp_semantic,
            status=result_status,
            session_time_ns=session_time_ns,
            uncertainty_ns=(
                self.uncertainty_ns
                + additional_uncertainty_ns
                + quantization_uncertainty_ns
            ),
            clock_mapping_id=self.clock_mapping_id,
        )

    def _exact_target(self, source_timestamp: int) -> Fraction:
        """计算尚未取整的精确会话时间。"""

        return Fraction(self.target_anchor_ns) + (
            Fraction(source_timestamp - self.source_anchor) * self.scale
        )


@dataclass(frozen=True, slots=True)
class SegmentedClockMapper:
    """按来源时钟、实例和有效范围选择唯一映射分段。"""

    session_id: str
    segments: tuple[ClockMappingSegment, ...]

    def __post_init__(self) -> None:
        """拒绝重复 ID、重复分段编号和同一时钟实例内的重叠范围。"""

        _validate_required_text(self.session_id, "session_id")
        if not isinstance(self.segments, tuple):
            raise TypeError("segments must be an immutable tuple")
        if not all(isinstance(segment, ClockMappingSegment) for segment in self.segments):
            raise TypeError("segments must contain ClockMappingSegment values")
        if any(segment.session_id != self.session_id for segment in self.segments):
            raise ValueError("clock mapping segment session does not match mapper")

        mapping_ids = [segment.clock_mapping_id for segment in self.segments]
        if len(set(mapping_ids)) != len(mapping_ids):
            raise ValueError("clock mapping ids must be unique")

        groups: dict[tuple[ClockId, str], list[ClockMappingSegment]] = {}
        for segment in self.segments:
            key = (segment.source_clock_id, segment.source_instance_id)
            groups.setdefault(key, []).append(segment)

        for grouped_segments in groups.values():
            ordered = sorted(grouped_segments, key=lambda segment: segment.source_from)
            segment_indices = [segment.segment_index for segment in ordered]
            if segment_indices != list(range(len(ordered))):
                raise ValueError(
                    "segment indices must be contiguous in source-time order"
                )
            for previous, current in zip(ordered, ordered[1:], strict=False):
                if current.source_from <= previous.source_to:
                    raise ValueError("clock mapping segments cannot overlap")
                if current._exact_target(current.source_from) <= previous._exact_target(
                    previous.source_to
                ):
                    raise ValueError("clock mapping target time must strictly increase")

    def map(
        self,
        observation: TimeObservation,
        *,
        additional_uncertainty_ns: int = 0,
        additional_status: TimeStatus = TimeStatus.VERIFIED,
    ) -> TimeEstimate:
        """选择对应分段；没有有效映射时保留原始时间并返回 unavailable。"""

        if not isinstance(observation, TimeObservation):
            raise TypeError("observation must be a TimeObservation")
        if type(additional_uncertainty_ns) is not int:
            raise TypeError("additional uncertainty must be an integer")
        if additional_uncertainty_ns < 0:
            raise ValueError("additional uncertainty cannot be negative")
        if not isinstance(additional_status, TimeStatus):
            raise TypeError("additional_status must be a TimeStatus")

        matching = [segment for segment in self.segments if segment.contains(observation)]
        if not matching:
            return _unavailable_estimate(observation)
        if len(matching) != 1:
            raise RuntimeError("ambiguous clock mapping segments")
        return matching[0].map(
            observation,
            additional_uncertainty_ns=additional_uncertainty_ns,
            additional_status=additional_status,
        )


def glasses_elapsed_source_instance_id(
    session_id: str,
    connection_session_id: str,
) -> str:
    """构造眼镜 elapsed-realtime/IMU 时钟的会话级稳定实例 ID。"""

    return _source_instance_id("glasses-elapsed", session_id, connection_session_id)


def rokid_sdk_source_instance_id(
    session_id: str,
    connection_session_id: str,
    camera_start_generation: int,
) -> str:
    """构造包含会话、连接和相机代数的 Rokid SDK 时钟实例 ID。"""

    if type(camera_start_generation) is not int:
        raise TypeError("camera_start_generation must be an integer")
    if camera_start_generation < 1:
        raise ValueError("camera_start_generation must be positive")
    return _source_instance_id(
        "rokid-sdk",
        session_id,
        connection_session_id,
        str(camera_start_generation),
    )


def mp4_source_instance_id(
    session_id: str,
    clip_id: str,
    time_base_numerator: int,
    time_base_denominator: int,
) -> str:
    """构造包含会话、clip 和 time base 的 MP4 展示时钟实例 ID。"""

    if type(time_base_numerator) is not int or type(time_base_denominator) is not int:
        raise TypeError("MP4 time base must use integers")
    if time_base_numerator <= 0 or time_base_denominator <= 0:
        raise ValueError("MP4 time base must be positive")
    return _source_instance_id(
        "mp4-presentation",
        session_id,
        clip_id,
        str(time_base_numerator),
        str(time_base_denominator),
    )


def client_perf_source_instance_id(
    session_id: str,
    connection_session_id: str,
) -> str:
    """构造 client perf-counter 的会话和连接代理实例 ID。"""

    return _source_instance_id("client-perf", session_id, connection_session_id)


def imu_sensor_event_observation(sample: RawImuSample) -> TimeObservation:
    """从原始 IMU 样本创建保留 sensor-event 语义的时间观察值。"""

    if not isinstance(sample, RawImuSample):
        raise TypeError("sample must be a RawImuSample")
    return TimeObservation(
        session_id=sample.session_id,
        source_clock_id=ClockId.GLASSES_ELAPSED_REALTIME_NS,
        source_instance_id=glasses_elapsed_source_instance_id(
            sample.session_id,
            sample.connection_session_id,
        ),
        source_timestamp=sample.sensor_event_monotonic_ns,
        timestamp_semantic=TimestampSemantic.SENSOR_EVENT,
    )


def frame_callback_observation(frame: RawFrameRef) -> TimeObservation | None:
    """从匹配帧创建 camera-callback 观察值；无相机元数据时返回 ``None``。"""

    if not isinstance(frame, RawFrameRef):
        raise TypeError("frame must be a RawFrameRef")
    if frame.connection_session_id is None or frame.video_at_monotonic_ns is None:
        return None
    return TimeObservation(
        session_id=frame.session_id,
        source_clock_id=ClockId.GLASSES_ELAPSED_REALTIME_NS,
        source_instance_id=glasses_elapsed_source_instance_id(
            frame.session_id,
            frame.connection_session_id,
        ),
        source_timestamp=frame.video_at_monotonic_ns,
        timestamp_semantic=TimestampSemantic.CAMERA_CALLBACK,
    )


def frame_sdk_observation(frame: RawFrameRef) -> TimeObservation | None:
    """从匹配帧创建未验证曝光含义的 Rokid SDK 时间观察值。"""

    if not isinstance(frame, RawFrameRef):
        raise TypeError("frame must be a RawFrameRef")
    if (
        frame.connection_session_id is None
        or frame.camera_start_generation is None
        or frame.captured_at_rokid_sdk_ms is None
    ):
        return None
    return TimeObservation(
        session_id=frame.session_id,
        source_clock_id=ClockId.ROKID_SDK_MS,
        source_instance_id=rokid_sdk_source_instance_id(
            frame.session_id,
            frame.connection_session_id,
            frame.camera_start_generation,
        ),
        source_timestamp=frame.captured_at_rokid_sdk_ms,
        timestamp_semantic=TimestampSemantic.CAMERA_SDK_TIMESTAMP,
    )


def frame_presentation_observation(frame: RawFrameRef) -> TimeObservation:
    """从帧 MP4 PTS 和 time base 创建媒体展示时间观察值。"""

    if not isinstance(frame, RawFrameRef):
        raise TypeError("frame must be a RawFrameRef")
    timestamp = frame.mp4_timestamp
    return TimeObservation(
        session_id=frame.session_id,
        source_clock_id=ClockId.MP4_PRESENTATION_TICKS,
        source_instance_id=mp4_source_instance_id(
            frame.session_id,
            frame.clip_id,
            timestamp.time_base_numerator,
            timestamp.time_base_denominator,
        ),
        source_timestamp=timestamp.pts,
        timestamp_semantic=TimestampSemantic.MEDIA_PRESENTATION,
    )


def rtp_match_error_to_uncertainty_ns(timestamp_match_error_90khz: int) -> int:
    """把非负 90 kHz 匹配误差向上取整为纳秒误差上界。"""

    if type(timestamp_match_error_90khz) is not int:
        raise TypeError("RTP timestamp match error must be an integer")
    if timestamp_match_error_90khz < 0:
        raise ValueError("RTP timestamp match error cannot be negative")
    numerator = timestamp_match_error_90khz * _NS_PER_SECOND
    return (numerator + _RTP_CLOCK_HZ - 1) // _RTP_CLOCK_HZ


def _source_instance_id(kind: str, *components: str) -> str:
    """使用规范 JSON 构造可验证且无分隔符碰撞的确定性实例 ID。"""

    _validate_required_text(kind, "source instance kind")
    for component in components:
        _validate_required_text(component, "source instance component")
    return json.dumps(
        {"components": components, "kind": kind},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validate_source_instance_scope(
    source_clock_id: ClockId,
    session_id: str,
    source_instance_id: str,
) -> None:
    """验证实例 ID 的类型、组件数量以及其中的 session 边界。"""

    expected_kind, expected_component_count = {
        ClockId.GLASSES_ELAPSED_REALTIME_NS: ("glasses-elapsed", 2),
        ClockId.ROKID_SDK_MS: ("rokid-sdk", 3),
        ClockId.CLIENT_PERF_COUNTER_NS: ("client-perf", 2),
        ClockId.MP4_PRESENTATION_TICKS: ("mp4-presentation", 4),
    }[source_clock_id]
    try:
        payload = json.loads(source_instance_id)
    except json.JSONDecodeError as exc:
        raise ValueError("source_instance_id must use the canonical constructor") from exc
    if not isinstance(payload, dict) or set(payload) != {"components", "kind"}:
        raise ValueError("source_instance_id has an invalid canonical structure")
    components = payload["components"]
    if (
        payload["kind"] != expected_kind
        or not isinstance(components, list)
        or len(components) != expected_component_count
        or not all(isinstance(component, str) and component for component in components)
    ):
        raise ValueError("source_instance_id does not match source clock")
    if components[0] != session_id:
        raise ValueError("source_instance_id session does not match mapping session")
    canonical_id = _source_instance_id(expected_kind, *components)
    if source_instance_id != canonical_id:
        raise ValueError("source_instance_id must use canonical JSON serialization")


def _validate_required_text(value: str, field_name: str) -> None:
    """验证契约中的必填字符串。"""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} cannot be empty")


def _unavailable_estimate(observation: TimeObservation) -> TimeEstimate:
    """保留原始观察值并创建没有派生时间的 unavailable 结果。"""

    return TimeEstimate(
        session_id=observation.session_id,
        source_clock_id=observation.source_clock_id,
        source_instance_id=observation.source_instance_id,
        source_timestamp=observation.source_timestamp,
        timestamp_semantic=observation.timestamp_semantic,
        status=TimeStatus.UNAVAILABLE,
    )


def _weakest_status(first: TimeStatus, second: TimeStatus) -> TimeStatus:
    """按 unavailable、estimated、verified 的可信度顺序返回较弱状态。"""

    confidence = {
        TimeStatus.UNAVAILABLE: 0,
        TimeStatus.ESTIMATED: 1,
        TimeStatus.VERIFIED: 2,
    }
    return min((first, second), key=confidence.__getitem__)


def _round_fraction_nearest(value: Fraction) -> int:
    """使用确定性的 half-away-from-zero 规则把有理数取整为整数。"""

    sign = -1 if value < 0 else 1
    numerator = abs(value.numerator)
    quotient, remainder = divmod(numerator, value.denominator)
    if remainder * 2 >= value.denominator:
        quotient += 1
    return sign * quotient
