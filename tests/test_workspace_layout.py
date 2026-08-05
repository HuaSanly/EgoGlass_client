import ast
from pathlib import Path

CLIENT_ROOT = Path(__file__).parents[1]
EXPECTED_PACKAGES = {
    "schemas",
    "sensor_preprocessing",
    "hand_tracking",
    "slam_vio",
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
