import ast
from pathlib import Path

CLIENT_ROOT = Path(__file__).parents[1]
EXPECTED_PACKAGES = {"schemas"}
REMOVED_RUNTIME_PATHS = (
    "src/hand_tracking",
    "src/sensor_preprocessing",
    "src/slam_vio",
    "src/process_video.py",
    "src/run_vio.py",
    "ui/annotation",
    "ui/configuration",
    "ui/presentation",
    "ui/processing",
    "ui/replay",
    "ui/video_processing",
)
RETIRED_IMPORT_ROOTS = {
    "OpenGL",
    "ahrs",
    "cv2",
    "hamer",
    "hand_tracking",
    "mediapipe",
    "scipy",
    "sensor_preprocessing",
    "slam_vio",
    "torch",
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


def test_only_recording_boundary_schemas_remain_under_source() -> None:
    source_root = CLIENT_ROOT / "src"

    assert (source_root / "schemas" / "__init__.py").is_file()
    assert {path.name for path in (source_root / "schemas").glob("*.py")} == {
        "__init__.py",
        "recording.py",
    }
    assert all(not (CLIENT_ROOT / path).exists() for path in REMOVED_RUNTIME_PATHS)


def test_only_recording_configuration_and_operator_scripts_remain() -> None:
    assert {path.name for path in (CLIENT_ROOT / "config").iterdir()} == {
        "README.md",
        "client-runtime.yaml",
    }
    assert {
        path.name
        for path in (CLIENT_ROOT / "scripts").iterdir()
        if path.is_file()
    } == {
        "benchmark_native_texture.py",
        "inspect-recording.py",
        "setup_client.ps1",
        "start-imu-calibration.ps1",
        "start-client.ps1",
    }


def test_packages_do_not_restore_nested_project_scaffolds() -> None:
    for package_name in EXPECTED_PACKAGES:
        package = CLIENT_ROOT / "src" / package_name
        assert not (package / "pyproject.toml").exists()
        assert not (package / "environment.yml").exists()
        assert not (package / "tests").exists()
        assert not (package / "evals").exists()


def test_schema_source_has_no_ui_or_retired_algorithm_imports() -> None:
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
                or name in {"hand_tracking", "sensor_preprocessing", "slam_vio"}
            for name in imported
            ), f"recording schema imports UI or a retired algorithm: {source_path}"


def test_recording_client_source_does_not_import_retired_algorithms() -> None:
    for source_path in (CLIENT_ROOT / "ui").rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [(node.module or "").split(".", 1)[0]]
            else:
                continue
            assert RETIRED_IMPORT_ROOTS.isdisjoint(imported), (
                f"recording client imports a retired algorithm: {source_path}"
            )


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
