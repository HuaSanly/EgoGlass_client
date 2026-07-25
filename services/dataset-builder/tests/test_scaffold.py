from __future__ import annotations

from importlib import import_module

import pytest

MODULES = (
    "coordinate_conversion",
    "dataset_split",
    "export",
    "pipeline",
    "provenance",
    "sample_builder",
    "schema_validation",
)


@pytest.mark.parametrize("module_name", MODULES)
def test_planned_dataset_builder_module_is_importable(module_name: str) -> None:
    module = import_module(f"egoglass_dataset_builder.{module_name}")
    assert module.__doc__
