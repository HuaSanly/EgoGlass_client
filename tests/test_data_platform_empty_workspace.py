from pathlib import Path

from data_platform.store import AnnotationStore


def test_missing_recordings_root_stays_absent_and_returns_empty_workspace(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing-recordings"

    workspace = AnnotationStore(missing_root).workspace()

    assert workspace.sessions == []
    assert workspace.skipped_session_count == 0
    assert missing_root.exists() is False
