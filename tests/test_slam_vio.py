from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from run_vio import _failure_summary
from sensor_preprocessing import (
    ImuCalibration,
    ImuSensorType,
    PreparedFrameBundle,
    PreparedImuSample,
    SensorCalibration,
    TimestampSemantic,
    TimeStatus,
)
from slam_vio import (
    BasaltEuRoCExporter,
    BasaltExecutionError,
    BasaltUnavailableError,
    BasaltVioConfig,
    BasaltVioRunner,
    WslBasaltExecutor,
    calibration_to_basalt_json,
    parse_euroc_trajectory,
    resolve_basalt_executable,
    synchronize_imu_samples,
)


def _calibration(*, source: str = "calibration laboratory") -> SensorCalibration:
    return SensorCalibration(
        schema_version="1.0",
        profile_name="test",
        capture_config_id="test-4x3",
        source_width=4,
        source_height=3,
        rotation_degrees=0,
        calibrated_width=4,
        calibrated_height=3,
        distortion_model="opencv_radtan",
        camera_matrix=((2.0, 0.0, 2.0), (0.0, 2.0, 1.5), (0.0, 0.0, 1.0)),
        distortion_coefficients=(0.1, 0.01, 0.0, 0.0, 0.0),
        rectified_camera_matrix=((2.0, 0.0, 2.0), (0.0, 2.0, 1.5), (0.0, 0.0, 1.0)),
        transform_camera_to_imu=(
            (1.0, 0.0, 0.0, 0.1),
            (0.0, 1.0, 0.0, 0.2),
            (0.0, 0.0, 1.0, 0.3),
            (0.0, 0.0, 0.0, 1.0),
        ),
        imu=ImuCalibration(
            nominal_rate_hz=100.0,
            accelerometer_noise_density_m_s2_sqrt_hz=0.02,
            accelerometer_random_walk_m_s3_sqrt_hz=0.002,
            gyroscope_noise_density_rad_s_sqrt_hz=0.002,
            gyroscope_random_walk_rad_s2_sqrt_hz=0.0002,
        ),
        provenance={
            "source": source,
            "tool": "kalibr",
            "tool_version": "1.0",
            "evidence_id": "lab-001",
        },
    )


def _imu(sensor: ImuSensorType, sample_id: int, timestamp_ns: int) -> PreparedImuSample:
    return PreparedImuSample(
        sample_id=sample_id,
        sequence_number=sample_id,
        session_time_ns=timestamp_ns,
        timestamp_uncertainty_ns=0,
        timestamp_status=TimeStatus.VERIFIED,
        clock_mapping_id="test-map",
        sensor_type=sensor,
        values=(
            (float(sample_id), 0.0, 9.8)
            if sensor is ImuSensorType.ACCELEROMETER
            else (0.1, 0.2, 0.3)
        ),
        unit="m_s2" if sensor is ImuSensorType.ACCELEROMETER else "rad_s",
        accuracy=3,
    )


def _bundles() -> tuple[PreparedFrameBundle, ...]:
    calibration = _calibration()
    samples = (
        _imu(ImuSensorType.ACCELEROMETER, 1, 1_000),
        _imu(ImuSensorType.ACCELEROMETER, 2, 2_000),
        _imu(ImuSensorType.ACCELEROMETER, 3, 3_000),
        _imu(ImuSensorType.GYROSCOPE, 4, 1_500),
        _imu(ImuSensorType.GYROSCOPE, 5, 2_500),
    )
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image.setflags(write=False)
    return (
        PreparedFrameBundle(
            session_id="session",
            sequence_id="clip",
            frame_index=0,
            session_time_ns=1_000,
            timestamp_uncertainty_ns=0,
            timestamp_status=TimeStatus.VERIFIED,
            timestamp_semantic=TimestampSemantic.SESSION_TIME,
            clock_mapping_id="test-map",
            image_bgr=image,
            imu_samples=samples,
            calibration=calibration,
        ),
        PreparedFrameBundle(
            session_id="session",
            sequence_id="clip",
            frame_index=1,
            session_time_ns=4_000,
            timestamp_uncertainty_ns=0,
            timestamp_status=TimeStatus.VERIFIED,
            timestamp_semantic=TimestampSemantic.SESSION_TIME,
            clock_mapping_id="test-map",
            image_bgr=image,
            imu_samples=(),
            calibration=calibration,
        ),
    )


def test_basalt_calibration_uses_rectified_matrix_and_t_imu_cam() -> None:
    payload = calibration_to_basalt_json(_calibration())
    camera = payload["value0"]["intrinsics"][0]
    assert camera == {
        "camera_type": "pinhole",
        "intrinsics": {"fx": 2.0, "fy": 2.0, "cx": 2.0, "cy": 1.5},
    }
    assert payload["value0"]["T_imu_cam"][0]["px"] == 0.1
    assert payload["value0"]["T_imu_cam"][0]["qw"] == 1.0
    assert payload["value0"]["vignette"] == []


def test_imu_union_is_interpolated_without_extrapolation() -> None:
    rows, skipped = synchronize_imu_samples(_bundles()[0].imu_samples)
    assert [row.timestamp_ns for row in rows] == [1_500, 2_000, 2_500]
    assert sum(row.interpolated for row in rows) == 3
    assert skipped == 2


def test_euroc_export_writes_images_csv_and_manifest(tmp_path: Path) -> None:
    dataset = BasaltEuRoCExporter().export(
        _bundles(),
        tmp_path / "dataset",
        calibration=_calibration(),
    )
    assert dataset.frame_count == 2
    assert dataset.imu_count == 3
    assert (dataset.root / "mav0/cam0/data/000000.png").is_file()
    assert (dataset.root / "calibration.json").is_file()
    with dataset.camera_data_csv.open(newline="", encoding="utf-8") as stream:
        assert list(csv.reader(stream))[1] == ["1000", "000000.png"]
    manifest = json.loads((dataset.root / "export.json").read_text(encoding="utf-8"))
    assert manifest["interpolated_imu_count"] == 3


def test_runner_invokes_fake_basalt_and_parses_trajectory(tmp_path: Path) -> None:
    fake = tmp_path / "fake_basalt.py"
    fake.write_text(
        """from pathlib import Path
Path('trajectory.csv').write_text(
    '#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],'
    'q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []\\n'
    '1000,0,0,0,1,0,0,0\\n2000,0.1,0,0,1,0,0,0\\n'
)
""",
        encoding="utf-8",
    )
    config = BasaltVioConfig(
        schema_version="1.0",
        executable=sys.executable,
        executable_args=(str(fake),),
        allow_unverified_calibration=False,
    )
    result = BasaltVioRunner(config).run(
        _bundles(),
        tmp_path / "run",
        calibration=_calibration(),
    )
    assert len(result.trajectory.poses) == 2
    assert result.command[0] == sys.executable
    assert result.dataset.root.is_dir()
    assert (tmp_path / "run" / "run.log").is_file()


def test_repository_config_selects_wsl_without_native_fallback() -> None:
    config = BasaltVioConfig.load(Path(__file__).parents[1] / "config/basalt-vio.yaml")

    assert config.backend == "wsl"
    assert config.wsl_distribution == "Nvidia_SDKM_Ubuntu_22.04_JetPack_7.2"
    assert config.wsl_executable == "/home/nvidia/egoglass/tools/basalt-build/basalt_vio"
    assert config.wsl_stage_root == "/home/nvidia/.cache/egoglass/basalt"
    assert config.wsl_keep_stage_on_failure is True


def test_wsl_config_rejects_unsafe_paths_and_gui() -> None:
    for values in (
        {"wsl_stage_root": "../unsafe"},
        {"wsl_executable": "relative/basalt_vio"},
        {"show_gui": True},
    ):
        try:
            BasaltVioConfig(schema_version="1.0", backend="wsl", **values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe WSL configuration was accepted: {values}")


def test_wsl_executor_preserves_argument_boundaries_and_materializes_results(
    tmp_path: Path,
) -> None:
    launcher = tmp_path / "wsl.exe"
    launcher.write_bytes(b"launcher")
    helper = tmp_path / "run basalt.sh"
    helper.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    output = tmp_path / "run with spaces"
    calls: list[list[str]] = []
    call_options: list[dict[str, object]] = []

    def run_command(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        call_options.append(kwargs)
        if "wslpath" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                "/mnt/f/translated path\n",
                "",
            )
        output.mkdir(parents=True, exist_ok=True)
        _write_trajectory_fixture(output / "trajectory.csv")
        (output / "basalt.stdout.log").write_text("ok\n", encoding="utf-8")
        (output / "basalt.stderr.log").write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    dataset = BasaltEuRoCExporter().export(
        _bundles(),
        tmp_path / "dataset with spaces",
        calibration=_calibration(),
    )
    config = BasaltVioConfig(schema_version="1.0", backend="wsl")
    result = WslBasaltExecutor(
        config,
        command_runner=run_command,
        wsl_executable=launcher,
        helper_script=helper,
    ).run(dataset, output)

    execution = calls[-1]
    assert "-lc" not in execution
    assert execution[:5] == [
        str(launcher),
        "--distribution",
        config.wsl_distribution,
        "--exec",
        "/bin/bash",
    ]
    assert "/mnt/f/translated path" in execution
    assert call_options
    assert all(options["encoding"] == "utf-8" for options in call_options)
    assert all(options["errors"] == "replace" for options in call_options)
    assert result.trajectory_path == output / "trajectory.csv"
    assert dict(result.metadata)["staging_state"] == "cleaned"


def test_wsl_executor_reports_nonzero_exit_without_native_fallback(tmp_path: Path) -> None:
    launcher = tmp_path / "wsl.exe"
    launcher.write_bytes(b"launcher")
    helper = tmp_path / "run-basalt.sh"
    helper.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    output = tmp_path / "run"

    def run_command(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if "wslpath" in command:
            return subprocess.CompletedProcess(command, 0, "/mnt/f/path\n", "")
        output.mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(command, 69, "", "missing Basalt")

    dataset = BasaltEuRoCExporter().export(
        _bundles(),
        tmp_path / "dataset",
        calibration=_calibration(),
    )
    executor = WslBasaltExecutor(
        BasaltVioConfig(schema_version="1.0", backend="wsl"),
        command_runner=run_command,
        wsl_executable=launcher,
        helper_script=helper,
    )
    try:
        executor.run(dataset, output)
    except BasaltExecutionError as error:
        assert error.returncode == 69
        assert "missing Basalt" in str(error)
        metadata = dict(error.backend_metadata)
        assert metadata["backend"] == "wsl"
        assert metadata["staging_state"] == "retained"
        assert metadata["wsl_stage_path"].startswith(
            "/home/nvidia/.cache/egoglass/basalt/"
        )
    else:
        raise AssertionError("failed WSL Basalt was accepted")


def test_wsl_failure_summary_preserves_backend_diagnostics(tmp_path: Path) -> None:
    config = BasaltVioConfig(schema_version="1.0", backend="wsl")
    error = BasaltExecutionError(
        "Basalt crashed",
        returncode=134,
        backend_metadata=(
            ("backend", "wsl"),
            ("wsl_distribution", config.wsl_distribution),
            ("wsl_executable", config.wsl_executable),
            ("basalt_revision", config.basalt_revision),
            ("wsl_stage_path", f"{config.wsl_stage_root}/session/run"),
            ("staging_state", "retained"),
        ),
    )

    payload = _failure_summary(
        output_path=tmp_path / "run",
        session_id="a" * 32,
        clip_id="clip-1",
        started_at_ns=123,
        config=config,
        error=error,
    )

    assert payload["state"] == "failed"
    assert payload["backend"] == "wsl"
    assert payload["returncode"] == 134
    assert payload["staging_state"] == "retained"
    assert payload["wsl_stage_path"] == f"{config.wsl_stage_root}/session/run"


def test_wsl_executor_rejects_missing_trajectory(tmp_path: Path) -> None:
    launcher = tmp_path / "wsl.exe"
    launcher.write_bytes(b"launcher")
    helper = tmp_path / "run-basalt.sh"
    helper.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    def run_command(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if "wslpath" in command:
            return subprocess.CompletedProcess(command, 0, "/mnt/f/path\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    dataset = BasaltEuRoCExporter().export(
        _bundles(),
        tmp_path / "dataset",
        calibration=_calibration(),
    )
    executor = WslBasaltExecutor(
        BasaltVioConfig(schema_version="1.0", backend="wsl"),
        command_runner=run_command,
        wsl_executable=launcher,
        helper_script=helper,
    )
    try:
        executor.run(dataset, tmp_path / "run")
    except BasaltExecutionError as error:
        assert "trajectory.csv" in str(error)
    else:
        raise AssertionError("missing WSL trajectory was accepted")


def test_wsl_backend_does_not_call_native_resolver(tmp_path: Path, monkeypatch) -> None:
    class UnavailableWsl:
        def __init__(self, _config: BasaltVioConfig) -> None:
            pass

        def run(self, _dataset: object, _output: Path) -> None:
            raise BasaltUnavailableError("WSL unavailable")

    monkeypatch.setattr("slam_vio.runner.WslBasaltExecutor", UnavailableWsl)
    monkeypatch.setattr(
        "slam_vio.runner.resolve_basalt_executable",
        lambda _value: (_ for _ in ()).throw(AssertionError("native fallback attempted")),
    )
    config = BasaltVioConfig(schema_version="1.0", backend="wsl")
    try:
        BasaltVioRunner(config).run(
            _bundles(),
            tmp_path / "run",
            calibration=_calibration(),
        )
    except BasaltUnavailableError as error:
        assert str(error) == "WSL unavailable"
    else:
        raise AssertionError("unavailable WSL backend was accepted")


def test_wsl_helper_restricts_recursive_cleanup_to_stage_root() -> None:
    script = (
        Path(__file__).parents[1] / "scripts/wsl/run-basalt.sh"
    ).read_text(encoding="utf-8")

    assert 'case "$stage_directory"' in script
    assert '"$stage_root"/*' in script
    assert 'rm -rf -- "$stage_directory"' in script
    assert "eval " not in script


def test_resolver_finds_workspace_local_basalt_without_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    executable = (
        tmp_path
        / ".tools"
        / "basalt-src"
        / "build"
        / "relwithdebinfo"
        / ("basalt_vio.exe" if sys.platform == "win32" else "basalt_vio")
    )
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"test executable")
    monkeypatch.delenv("EGOGLASS_BASALT_EXE", raising=False)
    monkeypatch.setattr("slam_vio.runner.shutil.which", lambda _value: None)

    resolved = resolve_basalt_executable("basalt_vio", workspace_root=tmp_path)

    assert resolved == executable.resolve()


def test_runner_rejects_sample_calibration(tmp_path: Path) -> None:
    config = BasaltVioConfig(schema_version="1.0", executable=sys.executable)
    try:
        BasaltVioRunner(config).run(
            _bundles(),
            tmp_path / "run",
            calibration=_calibration(source="unmeasured sample values"),
        )
    except BasaltExecutionError as error:
        assert "unverified" in str(error)
    else:
        raise AssertionError("sample calibration was accepted")


def test_trajectory_parser_rejects_bad_rows(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.csv"
    path.write_text("# header\n1,0,0,0,1\n", encoding="utf-8")
    try:
        parse_euroc_trajectory(path)
    except BasaltExecutionError:
        pass
    else:
        raise AssertionError("malformed trajectory was accepted")


def test_trajectory_pose_at_returns_nearest_timestamp(tmp_path: Path) -> None:
    trajectory = parse_euroc_trajectory(
        _write_trajectory_fixture(tmp_path / "trajectory.csv")
    )
    assert trajectory.pose_at(1_100).timestamp_ns == 1_000
    assert trajectory.pose_at(1_700).timestamp_ns == 2_000
    assert trajectory.pose_at(10_000, max_gap_ns=100) is None


def _write_trajectory_fixture(path: Path) -> Path:
    path.write_text(
        "# header\n1000,0,0,0,1,0,0,0\n2000,1,0,0,1,0,0,0\n",
        encoding="utf-8",
    )
    return path
