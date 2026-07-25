from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "spatial_perception"
FORBIDDEN_PACKAGES = {
    "dataset_builder",
    "interaction_processing",
}


def test_spatial_perception_does_not_import_downstream_services() -> None:
    violations: list[str] = []
    for source_path in SOURCE_ROOT.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            imported_names: list[str] = []
            if isinstance(node, ast.Import):
                imported_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names = [node.module]
            for imported_name in imported_names:
                package = imported_name.split(".", maxsplit=1)[0]
                if package in FORBIDDEN_PACKAGES:
                    violations.append(f"{source_path.name}:{node.lineno}:{imported_name}")

    assert violations == []
