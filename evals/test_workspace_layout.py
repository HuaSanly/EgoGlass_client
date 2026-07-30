from __future__ import annotations

import tomllib
from pathlib import Path

CLIENT_ROOT = Path(__file__).parents[1]
EXPECTED_PACKAGES = {
    "ui",
    "src/annotation",
    "src/ingest_gateway",
    "src/perception",
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
    }
    assert all("egoglass_" not in target for target in scripts.values())
