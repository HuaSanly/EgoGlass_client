from __future__ import annotations

import tomllib
from pathlib import Path

CLIENT_ROOT = Path(__file__).parents[1]
EXPECTED_PACKAGES = {
    "ui",
    "src/schemas",
    "src/sensor_preprocessing",
    "src/hand_tracking",
    "src/slam_vio",
    "src/phase_analysis",
    "src/object_tracking",
}


def test_workspace_manifest_builds_every_client_package_once() -> None:
    with (CLIENT_ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    packages = set(project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"])
    force_include = project["tool"]["hatch"]["build"]["targets"]["wheel"][
        "force-include"
    ]
    scripts = project["project"]["scripts"]

    assert packages == EXPECTED_PACKAGES
    assert set(scripts) == {
        "egoglass-client",
        "egoglass-ingest-gateway",
        "egoglass-process-video",
        "egoglass-run-vio",
    }
    assert all("egoglass_" not in target for target in scripts.values())
    assert force_include == {
        "src/process_video.py": "process_video.py",
        "src/run_vio.py": "run_vio.py",
    }
