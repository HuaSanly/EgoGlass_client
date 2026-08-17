from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

CLIENT_ROOT = Path(__file__).parents[1]
EXPECTED_PACKAGES = {
    "ui",
    "src/schemas",
}


def test_workspace_manifest_builds_every_client_package_once() -> None:
    with (CLIENT_ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    packages = set(project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"])
    scripts = project["project"]["scripts"]

    assert packages == EXPECTED_PACKAGES
    assert set(scripts) == {
        "egoglass-client",
        "egoglass-ingest-gateway",
        "egoglass-record-imu",
    }
    assert all("egoglass_" not in target for target in scripts.values())


def test_installable_project_has_no_algorithm_dependencies() -> None:
    with (CLIENT_ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    dependencies = "\n".join(project["project"]["dependencies"]).casefold()
    for retired_dependency in (
        "ahrs",
        "mediapipe",
        "opencv",
        "pyopenGL".casefold(),
        "scipy",
        "torch",
    ):
        assert retired_dependency.casefold() not in dependencies


def test_recording_only_install_has_no_offline_commands_or_configuration() -> None:
    retired_entries = (
        "config/basalt-vio.yaml",
        "config/live-hand-tracking.yaml",
        "config/offline-hand-tracking.yaml",
        "config/sensor-preprocessing.yaml",
        "config/video-processing.yaml",
        "scripts/download_hand_tracking_models.py",
        "src/process_video.py",
        "src/run_vio.py",
    )

    assert all(not (CLIENT_ROOT / entry).exists() for entry in retired_entries)


def test_recording_client_import_does_not_load_algorithm_stack() -> None:
    script = """
import json
import sys
import ui.app

banned = {
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
print("EGOGLASS_IMPORTS=" + json.dumps(sorted(
    name for name in sys.modules if name.split(".", 1)[0] in banned
)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=CLIENT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    report_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("EGOGLASS_IMPORTS=")
    )
    assert json.loads(report_line.removeprefix("EGOGLASS_IMPORTS=")) == []
