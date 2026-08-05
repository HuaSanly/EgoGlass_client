from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_headless_cli_help_starts_without_qt() -> None:
    repository = Path(__file__).parents[1]
    result = subprocess.run(
        [sys.executable, str(repository / "src" / "process_video.py"), "--help"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "without Qt" in result.stdout
    assert "PyQt6" not in result.stderr


def test_headless_run_artifacts_are_separate_from_ui_job_storage() -> None:
    source = (Path(__file__).parents[1] / "src" / "process_video.py").read_text(
        encoding="utf-8"
    )

    assert "results.jsonl" in source
    assert "jobs.sqlite3" not in source
    assert "ui.processing" not in source
