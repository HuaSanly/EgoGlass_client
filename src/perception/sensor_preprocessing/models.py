"""传感器预处理阶段使用的基础数据模型。

本模块只定义数据及其不变量，不读取文件、不执行时间拟合，也不运行感知算法。
"""

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction


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
