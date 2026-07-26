from __future__ import annotations

import tomllib
from pathlib import Path

CLIENT_ROOT = Path(__file__).parents[1]
EXPECTED_PACKAGES = {
    "src/dataset_builder",
    "src/ingest_gateway",
    "src/interaction_processing",
    "src/operator_console",
    "src/spatial_perception",
}


def test_workspace_manifest_builds_every_client_package_once() -> None:
    with (CLIENT_ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    packages = set(project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"])
    scripts = project["project"]["scripts"]

    assert packages == EXPECTED_PACKAGES
    assert set(scripts) == {
        "egoglass-console",
        "egoglass-desktop",
        "egoglass-ingest-gateway",
    }
    assert all("egoglass_" not in target for target in scripts.values())
