import ast
import subprocess
import tomllib
from pathlib import Path

import yaml

CLIENT_ROOT = Path(__file__).parents[1]
EXPECTED_PACKAGES = {
    "schemas",
    "sensor_preprocessing",
    "hand_tracking",
    "slam_vio",
    "phase_analysis",
    "object_tracking",
}


def test_client_uses_one_flat_python_workspace() -> None:
    assert (CLIENT_ROOT / "pyproject.toml").is_file()
    assert (CLIENT_ROOT / "environment.yml").is_file()
    assert not (CLIENT_ROOT / "uv.lock").exists()
    assert (CLIENT_ROOT / "scripts" / "setup_client.ps1").is_file()
    assert not (CLIENT_ROOT / "scripts" / "setup_hand_tracking.ps1").exists()
    assert not (CLIENT_ROOT / "services").exists()
    packages = {
        path.name
        for path in (CLIENT_ROOT / "src").iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }
    assert packages == EXPECTED_PACKAGES
    assert not (CLIENT_ROOT / "src" / "data_platform" / "__init__.py").exists()
    assert not any(path.name.startswith("egoglass_") for path in (CLIENT_ROOT / "src").iterdir())


def test_algorithms_are_top_level_source_packages() -> None:
    source_root = CLIENT_ROOT / "src"

    assert (source_root / "schemas" / "__init__.py").is_file()
    assert (source_root / "sensor_preprocessing" / "__init__.py").is_file()
    assert (source_root / "hand_tracking" / "__init__.py").is_file()
    assert (source_root / "slam_vio" / "__init__.py").is_file()
    assert (source_root / "process_video.py").is_file()


def test_packages_do_not_restore_nested_project_scaffolds() -> None:
    for package_name in EXPECTED_PACKAGES:
        package = CLIENT_ROOT / "src" / package_name
        assert not (package / "pyproject.toml").exists()
        assert not (package / "environment.yml").exists()
        assert not (package / "tests").exists()
        assert not (package / "evals").exists()


def test_algorithm_source_has_no_ui_or_legacy_perception_imports() -> None:
    source_root = CLIENT_ROOT / "src"
    for source_path in source_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""]
            else:
                continue
            assert not any(
                name == "ui"
                or name.startswith("ui.")
                or name == "perception"
                or name.startswith("perception.")
                or name.startswith("src.perception")
                for name in imported
            ), f"algorithm module imports UI or legacy perception: {source_path}"


def test_documented_commands_use_the_single_conda_environment() -> None:
    command_files = (
        CLIENT_ROOT / "README.md",
        CLIENT_ROOT / "AGENTS.md",
        *(CLIENT_ROOT / "docs").glob("*.md"),
        *(CLIENT_ROOT / "scripts").glob("*.ps1"),
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in command_files)

    assert "uv run" not in combined
    assert "uv sync" not in combined
    assert ".venv\\Scripts" not in combined
    assert "conda run -n egoglass" in combined


def test_object_tracking_setup_pins_sources_and_declares_sam2_runtime_dependencies() -> None:
    with (CLIENT_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    environment = yaml.safe_load(
        (CLIENT_ROOT / "environment.yml").read_text(encoding="utf-8")
    )
    setup = (CLIENT_ROOT / "scripts" / "setup_client.ps1").read_text(encoding="utf-8")
    project_dependencies = set(project["project"]["dependencies"])
    pip_dependencies = set(environment["dependencies"][-1]["pip"])

    assert {
        "iopath>=0.1.10",
        "pillow>=9.4",
        "tqdm>=4.66.1",
        "transformers>=4.45,<6",
    } <= project_dependencies
    assert {
        "iopath>=0.1.10",
        "pillow>=9.4.0",
        "tqdm>=4.66.1",
        "transformers>=4.45,<6",
    } <= pip_dependencies
    assert "facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4" in setup
    assert '"sam-2 @ git+https://github.com/facebookresearch/sam2.git@' in setup
    assert "facebookresearch/co-tracker.git@82e02e8029753ad4ef13cf06be7f4fc5facdda4d" in setup
    assert '$env:SAM2_BUILD_CUDA = "0"' in setup


def test_wsl_setup_normalizes_distribution_output_without_char_overload() -> None:
    script = (CLIENT_ROOT / "scripts" / "setup-basalt-wsl.ps1").read_text(encoding="utf-8")

    assert "-replace \"`0\", ''" in script
    assert ".Replace([char]0, '')" not in script


def test_wsl_setup_maps_loopback_proxy_and_pins_basalt_revision() -> None:
    powershell = (CLIENT_ROOT / "scripts" / "setup-basalt-wsl.ps1").read_text(encoding="utf-8")
    linux = (CLIENT_ROOT / "scripts" / "wsl" / "setup-basalt.sh").read_text(encoding="utf-8")

    assert "$proxyUri.IsLoopback" in powershell
    assert "ip route show default" in powershell
    assert "$proxyArgument" in powershell
    assert 'export HTTPS_PROXY="$proxy_url"' in linux
    assert 'fetch --depth 1 origin "$basalt_revision"' in linux
    assert "clone --recurse-submodules" not in linux
    assert "cmake_version=3.31.10" in linux
    assert 'dpkg --compare-versions "$system_cmake_version" ge 3.24' in linux
    assert '"$cmake_executable" --build' in linux
    assert "-DCMAKE_CXX_FLAGS=-Wno-error=unused-variable" in linux
    assert '"$source_directory/vcpkg-configuration.json"' in linux
    assert "fetch --unshallow origin" in linux
    assert "apply --unidiff-zero --check" in linux
    assert 'cat-file -e "$vcpkg_baseline^{commit}"' in linux
    assert 'current_revision=$(git -C "$source_directory" rev-parse HEAD)' in linux
    assert '"$current_revision" != "$basalt_revision"' in linux
    assert "configure_attempt=1" in linux
    assert "if [[ $configure_attempt -ge 3 ]]" in linux
    assert "sleep 3" in linux
    assert "build_jobs=${EGOGLASS_BASALT_BUILD_JOBS:-4}" in linux
    assert 'export VCPKG_MAX_CONCURRENCY="$build_jobs"' in linux
    assert 'build "$build_directory" --parallel "$build_jobs"' in linux
    assert "libgles2-mesa-dev" in linux
    assert "libglu1-mesa-dev" in linux
    assert "libx11-dev" in linux


def test_wsl_basalt_patches_are_well_formed_git_patches() -> None:
    patches = sorted((CLIENT_ROOT / "scripts" / "wsl" / "patches").glob("*.patch"))
    assert [path.name for path in patches] == [
        "0001-monocular-euroc.patch",
        "0002-euroc-only-dependencies.patch",
    ]
    for patch in patches:
        completed = subprocess.run(
            ["git", "apply", "--numstat", str(patch)],
            cwd=CLIENT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, f"{patch.name}: {completed.stderr}"

    dependency_patch = patches[1].read_text(encoding="utf-8")
    assert '+    "boost-format",' in dependency_patch
    assert '+    "boost-thread",' in dependency_patch
    assert "-#include <basalt/io/dataset_io_kitti.h>" in dependency_patch
    assert "-#include <basalt/io/dataset_io_uzh.h>" in dependency_patch

    monocular_patch = patches[0].read_text(encoding="utf-8")
    assert "-      Eigen::Vector3d gyro, accel;" in monocular_patch
