from pathlib import Path

CLIENT_ROOT = Path(__file__).parents[1]
EXPECTED_PACKAGES = {
    "ingest_gateway",
    "operator_console",
    "perception",
}


def test_client_uses_one_flat_python_workspace() -> None:
    assert (CLIENT_ROOT / "pyproject.toml").is_file()
    assert (CLIENT_ROOT / "uv.lock").is_file()
    assert not (CLIENT_ROOT / "services").exists()
    packages = {
        path.name
        for path in (CLIENT_ROOT / "src").iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }
    assert packages == EXPECTED_PACKAGES
    assert not (CLIENT_ROOT / "src" / "data_platform" / "__init__.py").exists()
    assert not any(path.name.startswith("egoglass_") for path in (CLIENT_ROOT / "src").iterdir())


def test_perception_owns_sensor_preprocessing_package() -> None:
    perception = CLIENT_ROOT / "src" / "perception"

    assert (perception / "__init__.py").is_file()
    assert (perception / "sensor_preprocessing" / "__init__.py").is_file()
    assert (perception / "sensor_preprocessing" / "models.py").is_file()


def test_packages_do_not_restore_nested_project_scaffolds() -> None:
    for package_name in EXPECTED_PACKAGES:
        package = CLIENT_ROOT / "src" / package_name
        assert not (package / "pyproject.toml").exists()
        assert not (package / "uv.lock").exists()
        assert not (package / "tests").exists()
        assert not (package / "evals").exists()
