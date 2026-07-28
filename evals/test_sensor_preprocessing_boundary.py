from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = (
    Path(__file__).parents[1] / "src" / "perception" / "sensor_preprocessing"
)
FORBIDDEN_IMPORT_PREFIXES = (
    "ingest_gateway",
    "operator_console",
    "perception.spatial_perception",
)


def test_sensor_preprocessing_does_not_depend_on_runtime_consumers() -> None:
    violations: list[str] = []
    for source_path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            imported_names: list[str] = []
            if isinstance(node, ast.Import):
                imported_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names = [node.module]
            for imported_name in imported_names:
                if imported_name.startswith(FORBIDDEN_IMPORT_PREFIXES):
                    violations.append(f"{source_path.name}:{node.lineno}:{imported_name}")

    assert violations == []
