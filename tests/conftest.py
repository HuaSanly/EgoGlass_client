from __future__ import annotations

import os
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qt_application() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def recordings_root(tmp_path: Path) -> Path:
    root = tmp_path / "recordings"
    root.mkdir()
    return root
