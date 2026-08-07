from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path

import pytest

from slam_vio import BasaltVioConfig, resolve_basalt_executable


def test_basalt_cli_is_ui_independent() -> None:
    path = Path(__file__).parents[1] / "src" / "run_vio.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name == "ui" or name.startswith("ui.") for name in imports)
    assert not any(name in {"PyQt6", "qfluentwidgets", "aiortc"} for name in imports)


def test_basalt_config_defaults_to_quality_safe_flags() -> None:
    path = Path(__file__).parents[1] / "config" / "basalt-vio.yaml"
    text = path.read_text(encoding="utf-8")
    assert "allow_unverified_calibration: false" in text
    assert "input_is_rectified: true" in text
    assert "use_imu: true" in text


def test_wsl_setup_avoids_powershell_char_replacement_failure() -> None:
    script = (
        Path(__file__).parents[1] / "scripts/setup-basalt-wsl.ps1"
    ).read_text(encoding="utf-8")

    assert '-replace "`0", \'\'' in script
    assert ".Replace([char]0, '')" not in script


def test_wsl_setup_carries_windows_proxy_into_linux_downloads() -> None:
    root = Path(__file__).parents[1]
    powershell = (root / "scripts/setup-basalt-wsl.ps1").read_text(encoding="utf-8")
    linux = (root / "scripts/wsl/setup-basalt.sh").read_text(encoding="utf-8")

    assert "git config --global --get https.proxy" in powershell
    assert "default via ([0-9.]+)" in powershell
    assert 'export http_proxy="$proxy_url"' in linux
    assert 'git -C "$source_directory" fetch --depth 1' in linux
    assert "cmake-$cmake_version-linux-x86_64.tar.gz" in linux
    assert '"$cmake_executable" -S "$source_directory"' in linux
    assert "-DCMAKE_CXX_FLAGS=-Wno-error=unused-variable" in linux
    assert "rev-parse --is-shallow-repository" in linux
    assert "fetch --unshallow origin" in linux
    assert "Patched Basalt source is not at the requested revision" in linux
    assert "Basalt source revision verification failed" in linux
    assert "apply --unidiff-zero --check" in linux
    assert "Retrying Basalt dependency configuration" in linux
    assert "EGOGLASS_BASALT_BUILD_JOBS must be a positive integer" in linux
    assert "libgles2-mesa-dev" in linux
    assert "libglu1-mesa-dev" in linux
    assert "libx11-dev" in linux


def test_wsl_patch_files_parse_before_external_setup() -> None:
    root = Path(__file__).parents[1]
    patches = sorted((root / "scripts/wsl/patches").glob("*.patch"))
    for patch in patches:
        completed = subprocess.run(
            ["git", "apply", "--numstat", str(patch)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr

    dependency_patch = patches[1].read_text(encoding="utf-8")
    assert '+    "boost-format",' in dependency_patch
    assert '+    "boost-thread",' in dependency_patch
    assert "-#include <basalt/io/dataset_io_kitti.h>" in dependency_patch
    assert "-#include <basalt/io/dataset_io_uzh.h>" in dependency_patch

    monocular_patch = patches[0].read_text(encoding="utf-8")
    assert "-      Eigen::Vector3d gyro, accel;" in monocular_patch


def test_wsl_basalt_help_when_installed() -> None:
    """Exercise the configured WSL binary on developer and integration hosts."""

    launcher = shutil.which("wsl.exe") or shutil.which("wsl")
    if launcher is None:
        pytest.skip("WSL launcher is unavailable")
    config = BasaltVioConfig.load(
        Path(__file__).parents[1] / "config" / "basalt-vio.yaml"
    )
    probe = subprocess.run(
        [
            launcher,
            "--distribution",
            config.wsl_distribution,
            "--exec",
            "test",
            "-x",
            config.wsl_executable,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=15,
    )
    if probe.returncode != 0:
        pytest.skip("configured WSL Basalt binary is not installed")
    completed = subprocess.run(
        [
            launcher,
            "--distribution",
            config.wsl_distribution,
            "--exec",
            config.wsl_executable,
            "--help",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--dataset-path" in completed.stdout + completed.stderr


def test_native_basalt_help_when_configured() -> None:
    """Smoke-check the optional native executable without requiring it in CI."""

    executable = resolve_basalt_executable("basalt_vio")
    if executable is None:
        pytest.skip("Basalt native executable is not configured")
    completed = subprocess.run(
        [str(executable), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--dataset-path" in completed.stdout


def test_vio_runs_inside_the_offline_processing_job() -> None:
    repository = Path(__file__).parents[1]
    view_source = (repository / "ui" / "views" / "video_processing.py").read_text(
        encoding="utf-8"
    )
    workbench_source = (repository / "ui" / "video_processing" / "workbench.py").read_text(
        encoding="utf-8"
    )
    service_source = (repository / "ui" / "processing" / "service.py").read_text(
        encoding="utf-8"
    )
    live_source = (repository / "ui" / "views" / "home.py").read_text(encoding="utf-8")
    assert "处理当前视频" in workbench_source
    assert "vioRequested" not in workbench_source
    assert "offline_vio_runner" in service_source
    assert "request_vio(" not in view_source
    assert "request_vio" not in live_source
