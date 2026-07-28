"""把离线录像或实时解码帧整理为空间感知可直接消费的数据。"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import zip_longest
from pathlib import Path
from typing import Literal

import av
import cv2
import numpy as np
from av import VideoFrame
from av.error import FFmpegError
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .capture_reader import CaptureSessionReader
from .clock_mapping import (
    SegmentedClockMapper,
    frame_callback_observation,
    frame_presentation_observation,
    imu_sensor_event_observation,
    rtp_match_error_to_uncertainty_ns,
)
from .models import (
    ImuSensorType,
    MetadataMatchStatus,
    RawFrameRef,
    RawImuSample,
    TimeEstimate,
    TimeObservation,
    TimestampSemantic,
    TimeStatus,
)

_Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
_Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]
_BgrImage = NDArray[np.uint8]


class SensorPreprocessingError(RuntimeError):
    """输入无法被安全转换为空间感知输入。"""


class CalibrationProvenance(BaseModel):
    """标定参数的来源，防止占位值或未知文件被当作实测标定。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)


class ImuCalibration(BaseModel):
    """VIO 后端需要的 IMU 频率和连续时间噪声参数。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    nominal_rate_hz: float = Field(gt=0)
    accelerometer_noise_density_m_s2_sqrt_hz: float = Field(gt=0)
    accelerometer_random_walk_m_s3_sqrt_hz: float = Field(gt=0)
    gyroscope_noise_density_rad_s_sqrt_hz: float = Field(gt=0)
    gyroscope_random_walk_rad_s2_sqrt_hz: float = Field(gt=0)


class SensorCalibration(BaseModel):
    """一个固定采集配置对应的相机、Camera-IMU 和 IMU 标定。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1.0"]
    profile_name: str = Field(min_length=1)
    placeholder: bool
    capture_config_id: str = Field(min_length=1)
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    rotation_degrees: Literal[0, 90, 180, 270]
    calibrated_width: int = Field(gt=0)
    calibrated_height: int = Field(gt=0)
    distortion_model: Literal["opencv_radtan"]
    camera_matrix: _Matrix3
    distortion_coefficients: tuple[float, float, float, float, float]
    rectified_camera_matrix: _Matrix3
    transform_camera_to_imu: _Matrix4
    imu: ImuCalibration
    provenance: CalibrationProvenance

    @model_validator(mode="after")
    def validate_geometry(self) -> SensorCalibration:
        """验证图像方向、针孔矩阵以及刚体变换是否自洽。"""

        expected_size = (
            (self.source_height, self.source_width)
            if self.rotation_degrees in {90, 270}
            else (self.source_width, self.source_height)
        )
        if (self.calibrated_width, self.calibrated_height) != expected_size:
            raise ValueError("calibrated dimensions do not match source rotation")

        camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        rectified_matrix = np.asarray(self.rectified_camera_matrix, dtype=np.float64)
        distortion = np.asarray(self.distortion_coefficients, dtype=np.float64)
        transform = np.asarray(self.transform_camera_to_imu, dtype=np.float64)
        if not all(
            np.isfinite(values).all()
            for values in (camera_matrix, rectified_matrix, distortion, transform)
        ):
            raise ValueError("calibration values must be finite")
        for matrix in (camera_matrix, rectified_matrix):
            if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
                raise ValueError("camera focal lengths must be positive")
            if not np.allclose(matrix[2], (0.0, 0.0, 1.0), atol=1e-12):
                raise ValueError("camera matrix must use homogeneous pinhole form")
            if not 0 <= matrix[0, 2] < self.calibrated_width:
                raise ValueError("camera principal point x is outside the image")
            if not 0 <= matrix[1, 2] < self.calibrated_height:
                raise ValueError("camera principal point y is outside the image")

        rotation = transform[:3, :3]
        if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
            raise ValueError("Camera-to-IMU transform must use a homogeneous last row")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("Camera-to-IMU rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError("Camera-to-IMU rotation determinant must be +1")
        return self

    @property
    def calibration_profile_id(self) -> str:
        """根据规范 JSON 内容生成稳定的标定配置 ID。"""

        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"sensor-calibration-v1-{digest}"

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        allow_placeholder: bool = False,
    ) -> SensorCalibration:
        """读取并验证标定 JSON；正式运行默认拒绝占位参数。"""

        calibration_path = Path(path)
        try:
            payload = calibration_path.read_text(encoding="utf-8")
            calibration = cls.model_validate_json(payload)
        except (OSError, UnicodeError, ValidationError) as exc:
            raise SensorPreprocessingError("invalid sensor calibration file") from exc
        if calibration.placeholder and not allow_placeholder:
            raise SensorPreprocessingError(
                "placeholder calibration requires explicit opt-in"
            )
        return calibration


@dataclass(frozen=True, slots=True)
class LiveFrameInput:
    """网关为一个已经解码的实时帧提供的最小元数据。"""

    session_id: str
    stream_id: str
    frame_index: int
    time_observation: TimeObservation
    rotation_degrees: int
    capture_config_id: str
    association_uncertainty_ns: int = 0
    association_status: TimeStatus = TimeStatus.VERIFIED

    def __post_init__(self) -> None:
        if not self.session_id.strip() or not self.stream_id.strip():
            raise ValueError("live frame session and stream ids cannot be empty")
        if self.frame_index < 0:
            raise ValueError("live frame index cannot be negative")
        if self.time_observation.session_id != self.session_id:
            raise ValueError("live frame observation session does not match")
        if self.rotation_degrees not in {0, 90, 180, 270}:
            raise ValueError("live frame rotation is invalid")
        if not self.capture_config_id.strip():
            raise ValueError("live frame capture config cannot be empty")
        if type(self.association_uncertainty_ns) is not int:
            raise TypeError("association uncertainty must be an integer")
        if self.association_uncertainty_ns < 0:
            raise ValueError("association uncertainty cannot be negative")
        if not isinstance(self.association_status, TimeStatus):
            raise TypeError("association_status must be a TimeStatus")


@dataclass(frozen=True, slots=True)
class LiveImuInput:
    """网关尚未落盘的一条实时 IMU 样本。"""

    sample_id: int
    sequence_number: int
    time_observation: TimeObservation
    sensor_type: ImuSensorType
    values: tuple[float, float, float]
    unit: str
    accuracy: int

    def __post_init__(self) -> None:
        if self.sample_id < 0 or self.sequence_number < 0:
            raise ValueError("live IMU identifiers cannot be negative")
        if not isinstance(self.sensor_type, ImuSensorType):
            raise TypeError("sensor_type must be an ImuSensorType")
        if not isinstance(self.values, tuple) or len(self.values) != 3:
            raise TypeError("live IMU values must be an immutable three-element tuple")
        if not all(np.isfinite(value) for value in self.values):
            raise ValueError("live IMU values must contain three finite values")
        expected_unit = {
            ImuSensorType.ACCELEROMETER: "m_s2",
            ImuSensorType.GYROSCOPE: "rad_s",
        }[self.sensor_type]
        if self.unit != expected_unit:
            raise ValueError("live IMU unit does not match sensor type")
        if not -1 <= self.accuracy <= 3:
            raise ValueError("live IMU accuracy must be between -1 and 3")


@dataclass(frozen=True, slots=True)
class PreparedImuSample:
    """已映射到会话时间、可直接送入 VIO 的 IMU 样本。"""

    sample_id: int
    sequence_number: int
    session_time_ns: int
    timestamp_uncertainty_ns: int
    timestamp_status: TimeStatus
    clock_mapping_id: str
    sensor_type: ImuSensorType
    values: tuple[float, float, float]
    unit: str
    accuracy: int


@dataclass(frozen=True, slots=True)
class PreparedFrameBundle:
    """空间感知每处理一帧时收到的统一输入。"""

    session_id: str
    sequence_id: str
    frame_index: int
    session_time_ns: int
    timestamp_uncertainty_ns: int
    timestamp_status: TimeStatus
    timestamp_semantic: TimestampSemantic
    clock_mapping_id: str
    image_bgr: _BgrImage
    imu_samples: tuple[PreparedImuSample, ...]
    calibration: SensorCalibration

    def __post_init__(self) -> None:
        if self.image_bgr.dtype != np.uint8 or self.image_bgr.ndim != 3:
            raise TypeError("prepared image must be an uint8 HxWx3 array")
        expected_shape = (
            self.calibration.calibrated_height,
            self.calibration.calibrated_width,
            3,
        )
        if self.image_bgr.shape != expected_shape:
            raise ValueError("prepared image dimensions do not match calibration")
        if self.image_bgr.flags.writeable:
            raise ValueError("prepared image must be read-only")


class SensorPreprocessingPipeline:
    """统一处理离线 MP4 和网关已经解码的实时帧。"""

    def __init__(
        self,
        calibration: SensorCalibration,
        clock_mapper: SegmentedClockMapper,
        *,
        allow_placeholder_calibration: bool = False,
    ) -> None:
        if not isinstance(calibration, SensorCalibration):
            raise TypeError("calibration must be a SensorCalibration")
        if not isinstance(clock_mapper, SegmentedClockMapper):
            raise TypeError("clock_mapper must be a SegmentedClockMapper")
        if calibration.placeholder and not allow_placeholder_calibration:
            raise SensorPreprocessingError(
                "placeholder calibration requires explicit opt-in"
            )
        self.calibration = calibration
        self.clock_mapper = clock_mapper
        self._map_x: NDArray[np.float32] | None = None
        self._map_y: NDArray[np.float32] | None = None
        self._initialize_rectification_maps()
        self.reset_live_state()

    @classmethod
    def from_calibration_file(
        cls,
        calibration_path: str | Path,
        clock_mapper: SegmentedClockMapper,
        *,
        allow_placeholder_calibration: bool = False,
    ) -> SensorPreprocessingPipeline:
        """从标定文件构造 pipeline，时钟映射由当前采集会话提供。"""

        calibration = SensorCalibration.load(
            calibration_path,
            allow_placeholder=allow_placeholder_calibration,
        )
        return cls(
            calibration,
            clock_mapper,
            allow_placeholder_calibration=allow_placeholder_calibration,
        )

    def iter_recorded_session(
        self,
        session_directory: str | Path,
        *,
        verify_media_hashes: bool = True,
    ) -> Iterator[PreparedFrameBundle]:
        """按 clip 和帧顺序解码一个已完成会话，并组装对应 IMU 时间窗。"""

        reader = CaptureSessionReader.open(
            session_directory,
            verify_media_hashes=verify_media_hashes,
        )
        if reader.session.session_id != self.clock_mapper.session_id:
            raise SensorPreprocessingError(
                "capture session does not match clock mapper session"
            )
        prepared_imu = tuple(
            sorted(
                (self._prepare_recorded_imu(sample) for sample in reader.iter_imu_samples()),
                key=lambda sample: (sample.session_time_ns, sample.sample_id),
            )
        )
        for clip in reader.session.clips:
            yield from self._iter_recorded_clip(reader, clip.clip_id, prepared_imu)

    def process_live_frame(
        self,
        decoded_frame: VideoFrame,
        frame_input: LiveFrameInput,
        imu_samples: Sequence[LiveImuInput] = (),
    ) -> PreparedFrameBundle:
        """直接处理网关已解码帧；不会执行视频解码或重新编码。"""

        if not isinstance(decoded_frame, VideoFrame):
            raise TypeError("decoded_frame must be an av.VideoFrame")
        if not isinstance(frame_input, LiveFrameInput):
            raise TypeError("frame_input must be a LiveFrameInput")
        if frame_input.session_id != self.clock_mapper.session_id:
            raise SensorPreprocessingError("live frame session does not match clock mapper")
        if self._live_session_id not in {None, frame_input.session_id}:
            raise SensorPreprocessingError("reset live state before changing sessions")
        self._validate_capture_config(
            decoded_frame.width,
            decoded_frame.height,
            frame_input.rotation_degrees,
            frame_input.capture_config_id,
        )

        frame_time = self.clock_mapper.map(
            frame_input.time_observation,
            additional_uncertainty_ns=frame_input.association_uncertainty_ns,
            additional_status=frame_input.association_status,
        )
        self._require_resolved_time(frame_time, "live frame")
        assert frame_time.session_time_ns is not None
        if (
            self._last_live_frame_time_ns is not None
            and frame_time.session_time_ns <= self._last_live_frame_time_ns
        ):
            raise SensorPreprocessingError("live frame session time must increase")

        new_imu = [self._prepare_live_imu(sample) for sample in imu_samples]
        previous_time = self._last_live_frame_time_ns
        if previous_time is not None and any(
            sample.session_time_ns <= previous_time for sample in new_imu
        ):
            raise SensorPreprocessingError("received a late live IMU sample")
        pending_imu = sorted(
            (*self._pending_live_imu, *new_imu),
            key=lambda sample: (sample.session_time_ns, sample.sample_id)
        )
        split_at = bisect_right(
            [sample.session_time_ns for sample in pending_imu],
            frame_time.session_time_ns,
        )
        frame_imu = tuple(pending_imu[:split_at])
        bundle = self._build_bundle(
            decoded_frame,
            session_id=frame_input.session_id,
            sequence_id=frame_input.stream_id,
            frame_index=frame_input.frame_index,
            frame_time=frame_time,
            rotation_degrees=frame_input.rotation_degrees,
            imu_samples=frame_imu,
        )
        self._pending_live_imu = pending_imu[split_at:]
        self._live_session_id = frame_input.session_id
        self._last_live_frame_time_ns = frame_time.session_time_ns
        return bundle

    def reset_live_state(self) -> None:
        """在实时会话、连接或回放重新开始时清空 IMU 分窗状态。"""

        self._live_session_id: str | None = None
        self._last_live_frame_time_ns: int | None = None
        self._pending_live_imu: list[PreparedImuSample] = []

    def _initialize_rectification_maps(self) -> None:
        """只计算一次 OpenCV 去畸变映射，避免逐帧重复准备。"""

        camera_matrix = np.asarray(self.calibration.camera_matrix, dtype=np.float64)
        rectified_matrix = np.asarray(
            self.calibration.rectified_camera_matrix,
            dtype=np.float64,
        )
        distortion = np.asarray(
            self.calibration.distortion_coefficients,
            dtype=np.float64,
        )
        if np.count_nonzero(distortion) == 0 and np.array_equal(
            camera_matrix,
            rectified_matrix,
        ):
            return
        self._map_x, self._map_y = cv2.initUndistortRectifyMap(
            camera_matrix,
            distortion,
            None,
            rectified_matrix,
            (self.calibration.calibrated_width, self.calibration.calibrated_height),
            cv2.CV_32FC1,
        )

    def _iter_recorded_clip(
        self,
        reader: CaptureSessionReader,
        clip_id: str,
        prepared_imu: tuple[PreparedImuSample, ...],
    ) -> Iterator[PreparedFrameBundle]:
        """校验解码帧与索引一一对应，并按相邻帧切出 IMU。"""

        frame_refs = reader.iter_frames(clip_id)
        clip = next(item for item in reader.session.clips if item.clip_id == clip_id)
        imu_times = [sample.session_time_ns for sample in prepared_imu]
        previous_frame_time_ns: int | None = None
        imu_cursor = 0
        sentinel = object()
        try:
            with av.open(str(clip.media_path), mode="r") as container:
                decoded_frames = container.decode(video=0)
                pairs = zip_longest(decoded_frames, frame_refs, fillvalue=sentinel)
                for decoded_frame, frame_ref in pairs:
                    if decoded_frame is sentinel or frame_ref is sentinel:
                        raise SensorPreprocessingError(
                            "decoded frame count does not match capture index"
                        )
                    assert isinstance(decoded_frame, VideoFrame)
                    assert isinstance(frame_ref, RawFrameRef)
                    self._validate_decoded_timestamp(decoded_frame, frame_ref)
                    self._validate_recorded_frame(frame_ref, decoded_frame)
                    frame_time = self._map_recorded_frame(frame_ref)
                    assert frame_time.session_time_ns is not None
                    if (
                        previous_frame_time_ns is not None
                        and frame_time.session_time_ns <= previous_frame_time_ns
                    ):
                        raise SensorPreprocessingError(
                            "recorded frame session time must increase"
                        )
                    if previous_frame_time_ns is None:
                        imu_cursor = bisect_right(imu_times, frame_time.session_time_ns)
                        frame_imu: tuple[PreparedImuSample, ...] = ()
                    else:
                        next_cursor = bisect_right(
                            imu_times,
                            frame_time.session_time_ns,
                            lo=imu_cursor,
                        )
                        frame_imu = prepared_imu[imu_cursor:next_cursor]
                        imu_cursor = next_cursor
                    previous_frame_time_ns = frame_time.session_time_ns
                    yield self._build_bundle(
                        decoded_frame,
                        session_id=frame_ref.session_id,
                        sequence_id=frame_ref.clip_id,
                        frame_index=frame_ref.frame_index,
                        frame_time=frame_time,
                        rotation_degrees=(
                            frame_ref.rotation_degrees
                            if frame_ref.rotation_degrees is not None
                            else self.calibration.rotation_degrees
                        ),
                        imu_samples=frame_imu,
                    )
        except FFmpegError as exc:
            raise SensorPreprocessingError("failed to decode recorded MP4") from exc

    def _map_recorded_frame(self, frame: RawFrameRef) -> TimeEstimate:
        """优先使用相机回调时间；无匹配元数据时退回精确 MP4 PTS。"""

        observation = frame_callback_observation(frame)
        if observation is None:
            estimate = self.clock_mapper.map(frame_presentation_observation(frame))
        else:
            assert frame.timestamp_match_error_90khz is not None
            association_status = (
                TimeStatus.VERIFIED
                if frame.metadata_match_status is MetadataMatchStatus.EXACT
                else TimeStatus.ESTIMATED
            )
            estimate = self.clock_mapper.map(
                observation,
                additional_uncertainty_ns=rtp_match_error_to_uncertainty_ns(
                    frame.timestamp_match_error_90khz
                ),
                additional_status=association_status,
            )
        self._require_resolved_time(estimate, f"recorded frame {frame.frame_index}")
        return estimate

    def _prepare_recorded_imu(self, sample: RawImuSample) -> PreparedImuSample:
        estimate = self.clock_mapper.map(imu_sensor_event_observation(sample))
        self._require_resolved_time(estimate, f"IMU sample {sample.sample_id}")
        return self._prepared_imu(
            sample.sample_id,
            sample.sequence_number,
            sample.sensor_type,
            sample.values,
            sample.unit,
            sample.accuracy,
            estimate,
        )

    def _prepare_live_imu(self, sample: LiveImuInput) -> PreparedImuSample:
        if sample.time_observation.session_id != self.clock_mapper.session_id:
            raise SensorPreprocessingError("live IMU session does not match clock mapper")
        estimate = self.clock_mapper.map(sample.time_observation)
        self._require_resolved_time(estimate, f"live IMU sample {sample.sample_id}")
        return self._prepared_imu(
            sample.sample_id,
            sample.sequence_number,
            sample.sensor_type,
            sample.values,
            sample.unit,
            sample.accuracy,
            estimate,
        )

    @staticmethod
    def _prepared_imu(
        sample_id: int,
        sequence_number: int,
        sensor_type: ImuSensorType,
        values: tuple[float, float, float],
        unit: str,
        accuracy: int,
        estimate: TimeEstimate,
    ) -> PreparedImuSample:
        assert estimate.session_time_ns is not None
        assert estimate.uncertainty_ns is not None
        assert estimate.clock_mapping_id is not None
        return PreparedImuSample(
            sample_id=sample_id,
            sequence_number=sequence_number,
            session_time_ns=estimate.session_time_ns,
            timestamp_uncertainty_ns=estimate.uncertainty_ns,
            timestamp_status=estimate.status,
            clock_mapping_id=estimate.clock_mapping_id,
            sensor_type=sensor_type,
            values=values,
            unit=unit,
            accuracy=accuracy,
        )

    def _build_bundle(
        self,
        decoded_frame: VideoFrame,
        *,
        session_id: str,
        sequence_id: str,
        frame_index: int,
        frame_time: TimeEstimate,
        rotation_degrees: int,
        imu_samples: tuple[PreparedImuSample, ...],
    ) -> PreparedFrameBundle:
        image = self._prepare_image(decoded_frame, rotation_degrees)
        assert frame_time.session_time_ns is not None
        assert frame_time.uncertainty_ns is not None
        assert frame_time.clock_mapping_id is not None
        return PreparedFrameBundle(
            session_id=session_id,
            sequence_id=sequence_id,
            frame_index=frame_index,
            session_time_ns=frame_time.session_time_ns,
            timestamp_uncertainty_ns=frame_time.uncertainty_ns,
            timestamp_status=frame_time.status,
            timestamp_semantic=frame_time.timestamp_semantic,
            clock_mapping_id=frame_time.clock_mapping_id,
            image_bgr=image,
            imu_samples=imu_samples,
            calibration=self.calibration,
        )

    def _prepare_image(
        self,
        decoded_frame: VideoFrame,
        rotation_degrees: int,
    ) -> _BgrImage:
        image = decoded_frame.to_ndarray(format="bgr24")
        image = {
            0: lambda value: value,
            90: lambda value: cv2.rotate(value, cv2.ROTATE_90_CLOCKWISE),
            180: lambda value: cv2.rotate(value, cv2.ROTATE_180),
            270: lambda value: cv2.rotate(value, cv2.ROTATE_90_COUNTERCLOCKWISE),
        }[rotation_degrees](image)
        if self._map_x is not None and self._map_y is not None:
            image = cv2.remap(
                image,
                self._map_x,
                self._map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
        image = np.ascontiguousarray(image, dtype=np.uint8)
        image.setflags(write=False)
        return image

    def _validate_recorded_frame(
        self,
        frame_ref: RawFrameRef,
        decoded_frame: VideoFrame,
    ) -> None:
        rotation = (
            frame_ref.rotation_degrees
            if frame_ref.rotation_degrees is not None
            else self.calibration.rotation_degrees
        )
        capture_config_id = frame_ref.capture_config_id or self.calibration.capture_config_id
        self._validate_capture_config(
            decoded_frame.width,
            decoded_frame.height,
            rotation,
            capture_config_id,
        )

    def _validate_capture_config(
        self,
        width: int,
        height: int,
        rotation_degrees: int,
        capture_config_id: str,
    ) -> None:
        if (width, height) != (
            self.calibration.source_width,
            self.calibration.source_height,
        ):
            raise SensorPreprocessingError(
                "decoded frame dimensions do not match calibration"
            )
        if rotation_degrees != self.calibration.rotation_degrees:
            raise SensorPreprocessingError("frame rotation does not match calibration")
        if capture_config_id != self.calibration.capture_config_id:
            raise SensorPreprocessingError("capture config does not match calibration")

    @staticmethod
    def _validate_decoded_timestamp(
        decoded_frame: VideoFrame,
        frame_ref: RawFrameRef,
    ) -> None:
        if decoded_frame.pts is None or decoded_frame.time_base is None:
            raise SensorPreprocessingError("decoded frame has no exact presentation time")
        decoded_time = Fraction(decoded_frame.pts) * Fraction(decoded_frame.time_base)
        if decoded_time != frame_ref.mp4_timestamp.presentation_time_seconds:
            raise SensorPreprocessingError("decoded frame PTS does not match capture index")

    @staticmethod
    def _require_resolved_time(estimate: TimeEstimate, description: str) -> None:
        if estimate.status is TimeStatus.UNAVAILABLE:
            raise SensorPreprocessingError(f"{description} has no session-time mapping")
