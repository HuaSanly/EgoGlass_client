import json
from pathlib import Path

from ui.processing import cleanup_legacy_hand_tracking


def _session(root: Path, name: str) -> Path:
    session = root / name
    (session / "media").mkdir(parents=True)
    (session / "telemetry").mkdir()
    (session / "perception" / "hand-tracking" / "old-run").mkdir(parents=True)
    (session / "media" / "clip.mp4").write_bytes(b"raw-media")
    (session / "telemetry" / "telemetry.sqlite").write_bytes(b"telemetry")
    (session / "session.json").write_text("{}", encoding="utf-8")
    (session / "quality.json").write_text("{}", encoding="utf-8")
    (session / "perception" / "hand-tracking" / "old-run" / "result.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    return session


def test_legacy_cleanup_removes_only_exact_result_directories(tmp_path: Path) -> None:
    first = _session(tmp_path, "session-a")
    second = _session(tmp_path, "session-b")
    audit = tmp_path / ".processing" / "cleanup.json"

    report = cleanup_legacy_hand_tracking(tmp_path, audit, apply=True)

    assert report.applied
    assert report.media_unchanged
    assert report.target_directories == (
        "session-a/perception/hand-tracking",
        "session-b/perception/hand-tracking",
    )
    assert not (first / "perception" / "hand-tracking").exists()
    assert not (second / "perception" / "hand-tracking").exists()
    for session in (first, second):
        assert (session / "media" / "clip.mp4").read_bytes() == b"raw-media"
        assert (session / "telemetry" / "telemetry.sqlite").read_bytes() == b"telemetry"
        assert (session / "session.json").is_file()
        assert (session / "quality.json").is_file()
    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert payload["state"] == "complete"
    assert payload["media_unchanged"] is True


def test_legacy_cleanup_dry_run_writes_audit_without_deleting(tmp_path: Path) -> None:
    session = _session(tmp_path, "session-a")
    audit = tmp_path / "audit.json"

    report = cleanup_legacy_hand_tracking(tmp_path, audit)

    assert not report.applied
    assert (session / "perception" / "hand-tracking").is_dir()
    assert audit.is_file()
