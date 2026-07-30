from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
import yaml
from av import VideoFrame

from ingest_gateway.adapters.mp4_recorder import RecordedVideoFrame
from ingest_gateway.capture_session import CaptureSessionDatabase
from ingest_gateway.recording_models import (
    CaptureSessionClip,
    CaptureSessionLifecycle,
    CaptureSessionManifest,
    CaptureSessionTimeOrigin,
    CaptureVideoProfile,
)
from ingest_gateway.webrtc_models import ImuSample
from ingest_gateway.webrtc_models import ImuSensorType as GatewayImuSensorType
from perception.sensor_preprocessing import (
    AlignmentStatus,
    CaptureSessionReader,
    ClockId,
    ClockMappingSegment,
    ImuSensorType,
    LiveFrameInput,
    LiveImuInput,
    SegmentedClockMapper,
    SensorCalibration,
    SensorPreprocessingConfig,
    SensorPreprocessingError,
    SensorPreprocessingPipeline,
    StoredAlignment,
    TimeObservation,
    TimestampSemantic,
    TimeStatus,
    glasses_elapsed_source_instance_id,
    mp4_source_instance_id,
)

SESSION_ID = "a" * 32
CONNECTION_ID = "b" * 32
CLIP_ID = "c" * 32


def _calibration_payload(
    *,
    width: int = 8,
    height: int = 6,
    rotation_degrees: int = 0,
) -> dict[str, object]:
    calibrated_width, calibrated_height = (
        (height, width) if rotation_degrees in {90, 270} else (width, height)
    )
    camera_matrix = [
        [8.0, 0.0, calibrated_width / 2],
        [0.0, 8.0, calibrated_height / 2],
        [0.0, 0.0, 1.0],
    ]
    return {
        "schema_version": "1.0",
        "profile_name": "tiny-test-calibration",
        "capture_config_id": "tiny",
        "source_width": width,
        "source_height": height,
        "rotation_degrees": rotation_degrees,
        "calibrated_width": calibrated_width,
        "calibrated_height": calibrated_height,
        "distortion_model": "opencv_radtan",
        "camera_matrix": camera_matrix,
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
        "rectified_camera_matrix": camera_matrix,
        "transform_camera_to_imu": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "imu": {
            "nominal_rate_hz": 100.0,
            "accelerometer_noise_density_m_s2_sqrt_hz": 0.02,
            "accelerometer_random_walk_m_s3_sqrt_hz": 0.002,
            "gyroscope_noise_density_rad_s_sqrt_hz": 0.002,
            "gyroscope_random_walk_rad_s2_sqrt_hz": 0.0002,
        },
        "provenance": {
            "source": "synthetic test calibration",
            "tool": "pytest",
            "tool_version": "1",
            "evidence_id": "fixture-1",
        },
    }


def _write_calibration(path: Path, **overrides: object) -> Path:
    payload = _calibration_payload(**overrides)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_preprocessing_config(
    path: Path,
    calibration_file: str | Path,
    *,
    verify_media_hashes: bool = True,
    decode_threads: int = 0,
    undistort: bool = True,
    max_pending_imu_samples: int = 2048,
    extra: dict[str, object] | None = None,
) -> Path:
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "calibration_file": str(calibration_file),
        "recorded": {
            "verify_media_hashes": verify_media_hashes,
            "decode_threads": decode_threads,
        },
        "image": {
            "undistort": undistort,
            "interpolation": "linear",
            "border_mode": "constant",
        },
        "live": {"max_pending_imu_samples": max_pending_imu_samples},
    }
    if extra:
        payload.update(extra)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _segment(
    *,
    source_clock_id: ClockId = ClockId.GLASSES_ELAPSED_REALTIME_NS,
    source_instance_id: str | None = None,
    source_from: int = 0,
    source_to: int = 1_000_000_000,
    source_anchor: int = 0,
    target_anchor_ns: int = 0,
    scale_numerator_ns: int = 1,
    scale_denominator_source_units: int = 1,
) -> ClockMappingSegment:
    return ClockMappingSegment(
        session_id=SESSION_ID,
        source_clock_id=source_clock_id,
        source_instance_id=(
            source_instance_id
            or glasses_elapsed_source_instance_id(SESSION_ID, CONNECTION_ID)
        ),
        segment_index=0,
        source_from=source_from,
        source_to=source_to,
        source_anchor=source_anchor,
        target_anchor_ns=target_anchor_ns,
        scale_numerator_ns=scale_numerator_ns,
        scale_denominator_source_units=scale_denominator_source_units,
        uncertainty_ns=100,
        status=TimeStatus.ESTIMATED,
        fit_method="synthetic_identity",
        provenance_id="test-fixture",
        uncertainty_basis="fixed_test_bound",
    )


def _pipeline(calibration: SensorCalibration) -> SensorPreprocessingPipeline:
    return SensorPreprocessingPipeline(
        calibration,
        SegmentedClockMapper(SESSION_ID, (_segment(),)),
    )


def _observation(timestamp_ns: int, semantic: TimestampSemantic) -> TimeObservation:
    return TimeObservation(
        session_id=SESSION_ID,
        source_clock_id=ClockId.GLASSES_ELAPSED_REALTIME_NS,
        source_instance_id=glasses_elapsed_source_instance_id(
            SESSION_ID,
            CONNECTION_ID,
        ),
        source_timestamp=timestamp_ns,
        timestamp_semantic=semantic,
    )


def _live_frame_input(frame_index: int, timestamp_ns: int) -> LiveFrameInput:
    return LiveFrameInput(
        session_id=SESSION_ID,
        stream_id="live-stream-1",
        frame_index=frame_index,
        time_observation=_observation(timestamp_ns, TimestampSemantic.CAMERA_CALLBACK),
        rotation_degrees=90,
        capture_config_id="tiny",
    )


def _live_imu(sample_id: int, timestamp_ns: int) -> LiveImuInput:
    return LiveImuInput(
        sample_id=sample_id,
        sequence_number=sample_id,
        time_observation=_observation(timestamp_ns, TimestampSemantic.SENSOR_EVENT),
        sensor_type=ImuSensorType.GYROSCOPE,
        values=(0.1, 0.2, 0.3),
        unit="rad_s",
        accuracy=3,
    )


def test_repository_config_selects_sample_calibration() -> None:
    config_directory = Path(__file__).parents[1] / "config"
    config = SensorPreprocessingConfig.load(
        config_directory / "sensor-preprocessing.yaml"
    )
    calibration = SensorCalibration.load(config.calibration_file)

    assert config.calibration_file == (
        config_directory / "sensor-calibration.sample.json"
    ).resolve()
    assert calibration.profile_name == "rokid-glass3-720p30-sample"
    assert calibration.rotation_degrees == 0
    assert calibration.calibrated_width == 1280
    assert calibration.calibrated_height == 720
    assert calibration.camera_matrix[0][2] == 640.0
    assert calibration.camera_matrix[1][2] == 360.0
    assert calibration.calibration_profile_id.startswith("sensor-calibration-v1-")


def test_preprocessing_config_rejects_unknown_fields(tmp_path: Path) -> None:
    calibration_path = _write_calibration(tmp_path / "calibration.json")
    config_path = _write_preprocessing_config(
        tmp_path / "sensor-preprocessing.yaml",
        calibration_path.name,
        extra={"unknown_setting": True},
    )

    with pytest.raises(SensorPreprocessingError, match="invalid sensor preprocessing"):
        SensorPreprocessingConfig.load(config_path)


def test_preprocessing_config_rejects_string_instead_of_boolean(tmp_path: Path) -> None:
    calibration_path = _write_calibration(tmp_path / "calibration.json")
    config_path = _write_preprocessing_config(
        tmp_path / "sensor-preprocessing.yaml",
        calibration_path.name,
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["recorded"]["verify_media_hashes"] = "false"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(SensorPreprocessingError, match="invalid sensor preprocessing"):
        SensorPreprocessingConfig.load(config_path)


def test_calibration_rejects_dimensions_inconsistent_with_rotation(tmp_path: Path) -> None:
    payload = _calibration_payload(rotation_degrees=90)
    payload["calibrated_width"] = 8
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SensorPreprocessingError, match="invalid sensor calibration"):
        SensorCalibration.load(path)


def test_live_pipeline_reuses_decoded_frame_and_windows_imu(tmp_path: Path) -> None:
    calibration = SensorCalibration.load(
        _write_calibration(
            tmp_path / "calibration.json",
            rotation_degrees=90,
        )
    )
    pipeline = _pipeline(calibration)
    image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape((6, 8, 3))
    decoded_frame = VideoFrame.from_ndarray(image, format="bgr24")

    first = pipeline.process_live_frame(
        decoded_frame,
        _live_frame_input(0, 20_000_000),
        (_live_imu(0, 5_000_000), _live_imu(1, 15_000_000), _live_imu(2, 25_000_000)),
    )
    second = pipeline.process_live_frame(
        decoded_frame,
        _live_frame_input(1, 30_000_000),
    )

    assert first.image_bgr.shape == (8, 6, 3)
    assert first.image_bgr.flags.writeable is False
    assert first.timestamp_semantic is TimestampSemantic.CAMERA_CALLBACK
    assert [sample.sample_id for sample in first.imu_samples] == [0, 1]
    assert [sample.sample_id for sample in second.imu_samples] == [2]
    assert second.session_time_ns == 30_000_000


def test_live_pipeline_normalizes_balanced_webrtc_resolution_to_calibration(
    tmp_path: Path,
) -> None:
    calibration = SensorCalibration.load(
        _write_calibration(tmp_path / "calibration.json", rotation_degrees=90)
    )
    pipeline = _pipeline(calibration)
    transport_scaled = VideoFrame.from_ndarray(
        np.full((3, 4, 3), 37, dtype=np.uint8),
        format="bgr24",
    )

    bundle = pipeline.process_live_frame(
        transport_scaled,
        _live_frame_input(0, 20_000_000),
    )

    assert bundle.image_bgr.shape == (8, 6, 3)
    assert np.all(bundle.image_bgr == 37)
    assert bundle.calibration is calibration


def test_live_pipeline_accepts_canonical_rgb_without_mutating_it(tmp_path: Path) -> None:
    calibration = SensorCalibration.load(
        _write_calibration(tmp_path / "calibration.json", rotation_degrees=90)
    )
    pipeline = _pipeline(calibration)
    image_rgb = np.empty((6, 8, 3), dtype=np.uint8)
    image_rgb[:] = (10, 20, 30)
    image_rgb.setflags(write=False)

    bundle = pipeline.process_live_rgb_frame(
        image_rgb,
        _live_frame_input(0, 20_000_000),
    )

    assert bundle.image_bgr.shape == (8, 6, 3)
    assert np.all(bundle.image_bgr == np.asarray((30, 20, 10), dtype=np.uint8))
    assert not bundle.image_bgr.flags.writeable
    assert not image_rgb.flags.writeable
    assert np.all(image_rgb == np.asarray((10, 20, 30), dtype=np.uint8))


def test_live_pipeline_rejects_transport_scaling_with_different_aspect_ratio(
    tmp_path: Path,
) -> None:
    calibration = SensorCalibration.load(
        _write_calibration(tmp_path / "calibration.json", rotation_degrees=90)
    )
    pipeline = _pipeline(calibration)
    invalid = VideoFrame.from_ndarray(
        np.zeros((4, 4, 3), dtype=np.uint8),
        format="bgr24",
    )

    with pytest.raises(SensorPreprocessingError, match="incompatible with calibration"):
        pipeline.process_live_frame(invalid, _live_frame_input(0, 20_000_000))


def test_live_pipeline_applies_nonzero_distortion_map(tmp_path: Path) -> None:
    payload = _calibration_payload(width=32, height=24)
    payload["distortion_coefficients"] = [0.5, 0.1, 0.0, 0.0, 0.0]
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(payload), encoding="utf-8")
    calibration = SensorCalibration.load(calibration_path)
    pipeline = _pipeline(calibration)
    image = np.arange(24 * 32 * 3, dtype=np.uint8).reshape((24, 32, 3))
    decoded_frame = VideoFrame.from_ndarray(image, format="bgr24")
    frame_input = LiveFrameInput(
        session_id=SESSION_ID,
        stream_id="live-stream-1",
        frame_index=0,
        time_observation=_observation(1, TimestampSemantic.CAMERA_CALLBACK),
        rotation_degrees=0,
        capture_config_id="tiny",
    )

    bundle = pipeline.process_live_frame(decoded_frame, frame_input)

    assert bundle.image_bgr.shape == image.shape
    assert not np.array_equal(bundle.image_bgr, image)


def test_yaml_config_can_disable_undistortion(tmp_path: Path) -> None:
    payload = _calibration_payload(width=32, height=24)
    payload["distortion_coefficients"] = [0.5, 0.1, 0.0, 0.0, 0.0]
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(payload), encoding="utf-8")
    config_path = _write_preprocessing_config(
        tmp_path / "sensor-preprocessing.yaml",
        calibration_path.name,
        undistort=False,
    )
    pipeline = SensorPreprocessingPipeline.from_config_file(
        config_path,
        SegmentedClockMapper(SESSION_ID, (_segment(),)),
    )
    image = np.arange(24 * 32 * 3, dtype=np.uint8).reshape((24, 32, 3))
    frame_input = LiveFrameInput(
        session_id=SESSION_ID,
        stream_id="live-stream-1",
        frame_index=0,
        time_observation=_observation(1, TimestampSemantic.CAMERA_CALLBACK),
        rotation_degrees=0,
        capture_config_id="tiny",
    )

    bundle = pipeline.process_live_frame(
        VideoFrame.from_ndarray(image, format="bgr24"),
        frame_input,
    )

    assert np.array_equal(bundle.image_bgr, image)


def test_yaml_live_imu_limit_rejects_without_committing_state(tmp_path: Path) -> None:
    calibration_path = _write_calibration(
        tmp_path / "calibration.json",
        rotation_degrees=90,
    )
    config_path = _write_preprocessing_config(
        tmp_path / "sensor-preprocessing.yaml",
        calibration_path.name,
        max_pending_imu_samples=1,
    )
    pipeline = SensorPreprocessingPipeline.from_config_file(
        config_path,
        SegmentedClockMapper(SESSION_ID, (_segment(),)),
    )
    frame = VideoFrame.from_ndarray(
        np.zeros((6, 8, 3), dtype=np.uint8),
        format="bgr24",
    )

    with pytest.raises(SensorPreprocessingError, match="buffer limit"):
        pipeline.process_live_frame(
            frame,
            _live_frame_input(0, 20_000_000),
            (_live_imu(1, 25_000_000), _live_imu(2, 30_000_000)),
        )

    bundle = pipeline.process_live_frame(frame, _live_frame_input(0, 20_000_000))

    assert bundle.imu_samples == ()


def test_live_pipeline_rejects_config_mismatch(tmp_path: Path) -> None:
    calibration = SensorCalibration.load(
        _write_calibration(tmp_path / "calibration.json", rotation_degrees=90)
    )
    pipeline = _pipeline(calibration)
    frame = VideoFrame.from_ndarray(np.zeros((6, 8, 3), dtype=np.uint8), format="bgr24")
    invalid_input = LiveFrameInput(
        session_id=SESSION_ID,
        stream_id="live-stream-1",
        frame_index=0,
        time_observation=_observation(1, TimestampSemantic.CAMERA_CALLBACK),
        rotation_degrees=0,
        capture_config_id="tiny",
    )

    with pytest.raises(SensorPreprocessingError, match="rotation"):
        pipeline.process_live_frame(frame, invalid_input)


def test_live_pipeline_rejects_late_imu_and_regressing_frames(tmp_path: Path) -> None:
    calibration = SensorCalibration.load(
        _write_calibration(tmp_path / "calibration.json", rotation_degrees=90)
    )
    frame = VideoFrame.from_ndarray(np.zeros((6, 8, 3), dtype=np.uint8), format="bgr24")
    pipeline = _pipeline(calibration)
    pipeline.process_live_frame(frame, _live_frame_input(0, 20_000_000))

    with pytest.raises(SensorPreprocessingError, match="late live IMU"):
        pipeline.process_live_frame(
            frame,
            _live_frame_input(1, 30_000_000),
            (_live_imu(1, 10_000_000),),
        )
    with pytest.raises(SensorPreprocessingError, match="must increase"):
        pipeline.process_live_frame(frame, _live_frame_input(1, 10_000_000))


def test_pipeline_rejects_unmapped_frame_time(tmp_path: Path) -> None:
    calibration = SensorCalibration.load(
        _write_calibration(tmp_path / "calibration.json", rotation_degrees=90)
    )
    pipeline = SensorPreprocessingPipeline(
        calibration,
        SegmentedClockMapper(SESSION_ID, ()),
    )
    frame = VideoFrame.from_ndarray(np.zeros((6, 8, 3), dtype=np.uint8), format="bgr24")

    with pytest.raises(SensorPreprocessingError, match="no session-time mapping"):
        pipeline.process_live_frame(frame, _live_frame_input(0, 1))


def _write_h264_mp4(path: Path) -> list[tuple[int, Fraction]]:
    with av.open(str(path), mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width = 8
        stream.height = 6
        stream.pix_fmt = "yuv420p"
        stream.codec_context.max_b_frames = 0
        for index, value in enumerate((40, 180)):
            image = np.full((6, 8, 3), value, dtype=np.uint8)
            frame = VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts = index
            frame.time_base = Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)

    with av.open(str(path), mode="r") as container:
        return [
            (frame.pts, Fraction(frame.time_base))
            for frame in container.decode(video=0)
            if frame.pts is not None and frame.time_base is not None
        ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recorded_session(root: Path) -> tuple[Path, list[tuple[int, Fraction]]]:
    session_directory = root / SESSION_ID
    media_directory = session_directory / "media"
    telemetry_directory = session_directory / "telemetry"
    media_directory.mkdir(parents=True)
    telemetry_directory.mkdir()
    media_path = media_directory / f"{CLIP_ID}.mp4"
    timings = _write_h264_mp4(media_path)
    assert len(timings) == 2

    database = CaptureSessionDatabase(
        SESSION_ID,
        telemetry_directory / "telemetry.sqlite",
    )
    for sequence_number, event_ns in enumerate((10_000_000, 20_000_000)):
        database.record_imu_sample(
            CONNECTION_ID,
            ImuSample(
                sensor_type=GatewayImuSensorType.GYROSCOPE,
                android_sensor_type=4,
                sequence_number=sequence_number,
                sensor_event_monotonic_ns=event_ns,
                received_at_elapsed_realtime_ns=event_ns + 100,
                accuracy=3,
                values=(0.1, 0.2, 0.3),
            ),
            received_at_client_perf_counter_ns=event_ns + 200,
        )
    database.record_clip_frames(
        CLIP_ID,
        CONNECTION_ID,
        1,
        tuple(
            RecordedVideoFrame(
                frame_index=index,
                source_frame_pts=None,
                source_frame_time_base_num=None,
                source_frame_time_base_den=None,
                mp4_pts=pts,
                mp4_time_base_num=time_base.numerator,
                mp4_time_base_den=time_base.denominator,
                received_at_client_perf_counter_ns=index,
            )
            for index, (pts, time_base) in enumerate(timings)
        ),
        expected_frame_count=2,
    )
    database.checkpoint_and_close()

    manifest = CaptureSessionManifest(
        session_id=SESSION_ID,
        display_name="Pipeline fixture",
        display_name_source="operator",
        lifecycle=CaptureSessionLifecycle(
            state="complete",
            started_at_unix_ns=1,
            ended_at_unix_ns=2,
            end_reason="client_shutdown",
        ),
        session_time_origin=CaptureSessionTimeOrigin(),
        clips=[
            CaptureSessionClip(
                clip_id=CLIP_ID,
                state="complete",
                relative_media_path=f"media/{CLIP_ID}.mp4",
                video_profile=CaptureVideoProfile(
                    width=8,
                    height=6,
                    nominal_fps=30.0,
                ),
                frame_count=2,
                sha256=_sha256(media_path),
            )
        ],
    )
    (session_directory / "session.json").write_text(
        manifest.model_dump_json(),
        encoding="utf-8",
    )
    return session_directory, timings


def test_recorded_pipeline_applies_yaml_and_assembles_imu_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_directory, timings = _recorded_session(tmp_path)
    calibration_path = _write_calibration(tmp_path / "calibration.json")
    config_path = _write_preprocessing_config(
        tmp_path / "sensor-preprocessing.yaml",
        calibration_path.name,
        verify_media_hashes=False,
        decode_threads=1,
    )
    manifest_path = session_directory / "session.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["clips"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    first_pts, time_base = timings[0]
    last_pts, last_time_base = timings[-1]
    assert last_time_base == time_base
    mp4_segment = _segment(
        source_clock_id=ClockId.MP4_PRESENTATION_TICKS,
        source_instance_id=mp4_source_instance_id(
            SESSION_ID,
            CLIP_ID,
            time_base.numerator,
            time_base.denominator,
        ),
        source_from=first_pts,
        source_to=last_pts,
        source_anchor=first_pts,
        scale_numerator_ns=1_000_000_000 * time_base.numerator,
        scale_denominator_source_units=time_base.denominator,
    )
    pipeline = SensorPreprocessingPipeline.from_config_file(
        config_path,
        SegmentedClockMapper(SESSION_ID, (_segment(), mp4_segment)),
    )

    real_av_open = av.open
    observed_decode_threads: list[int] = []

    class TrackingInputContainer:
        def __init__(self, container: av.container.InputContainer) -> None:
            self._container = container
            self.streams = container.streams

        def __enter__(self) -> TrackingInputContainer:
            self._container.__enter__()
            return self

        def __exit__(self, *args: object) -> object:
            return self._container.__exit__(*args)

        def decode(self, stream: av.video.stream.VideoStream):
            observed_decode_threads.append(stream.codec_context.thread_count)
            return self._container.decode(stream)

    def tracking_av_open(*args: object, **kwargs: object) -> TrackingInputContainer:
        return TrackingInputContainer(real_av_open(*args, **kwargs))

    monkeypatch.setattr(av, "open", tracking_av_open)

    bundles = list(pipeline.iter_recorded_session(session_directory))

    assert observed_decode_threads == [1]
    assert [bundle.frame_index for bundle in bundles] == [0, 1]
    assert bundles[0].imu_samples == ()
    assert [sample.sample_id for sample in bundles[1].imu_samples] == [1, 2]
    assert bundles[1].session_time_ns > bundles[0].session_time_ns
    assert all(bundle.image_bgr.shape == (6, 8, 3) for bundle in bundles)
    assert all(
        bundle.timestamp_semantic is TimestampSemantic.MEDIA_PRESENTATION
        for bundle in bundles
    )


def test_recorded_frame_reuses_persisted_alignment_without_clock_segments(
    tmp_path: Path,
) -> None:
    session_directory, _ = _recorded_session(tmp_path)
    reader = CaptureSessionReader.open(session_directory)
    raw_frame = next(reader.iter_frames(CLIP_ID))
    mapped_frame = replace(
        raw_frame,
        stored_alignment=StoredAlignment(
            status=AlignmentStatus.MAPPED,
            session_time_ns=123_000_000,
            uncertainty_ns=2_000_000,
            clock_mapping_segment_id="persisted-segment-1",
        ),
    )
    calibration = SensorCalibration.load(_write_calibration(tmp_path / "calibration.json"))
    pipeline = SensorPreprocessingPipeline(
        calibration,
        SegmentedClockMapper(SESSION_ID, ()),
    )

    estimate = pipeline._map_recorded_frame(mapped_frame)

    assert estimate.session_time_ns == 123_000_000
    assert estimate.uncertainty_ns == 2_000_000
    assert estimate.clock_mapping_id == "persisted-segment-1"
    assert estimate.status is TimeStatus.ESTIMATED
