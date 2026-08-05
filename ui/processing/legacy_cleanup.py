from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class HashedFile:
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class LegacyCleanupReport:
    recordings_root: str
    audit_path: str
    applied: bool
    target_directories: tuple[str, ...]
    target_files: tuple[HashedFile, ...]
    media_before: tuple[HashedFile, ...]
    media_after: tuple[HashedFile, ...]
    media_unchanged: bool


def cleanup_legacy_hand_tracking(
    recordings_root: str | Path,
    audit_path: str | Path,
    *,
    apply: bool = False,
) -> LegacyCleanupReport:
    """Audit and optionally remove only legacy per-session hand-tracking results."""

    root = Path(recordings_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("recordings root is unavailable")
    audit = Path(audit_path).expanduser().resolve()
    if audit.is_relative_to(root) and any(
        audit.is_relative_to(target) for target in _legacy_targets(root)
    ):
        raise ValueError("cleanup audit cannot be written inside a deletion target")

    targets = _legacy_targets(root)
    target_files = _hashed_files(root, targets)
    media_before = _media_hashes(root)
    preflight = LegacyCleanupReport(
        recordings_root=str(root),
        audit_path=str(audit),
        applied=False,
        target_directories=tuple(_relative(root, target) for target in targets),
        target_files=target_files,
        media_before=media_before,
        media_after=media_before,
        media_unchanged=True,
    )
    _write_audit(audit, preflight, state="preflight")
    if not apply:
        return preflight

    for target in targets:
        _validate_target(root, target)
        shutil.rmtree(target)
    remaining = _legacy_targets(root)
    if remaining:
        raise RuntimeError("legacy hand-tracking targets remain after cleanup")
    media_after = _media_hashes(root)
    media_unchanged = media_before == media_after
    report = LegacyCleanupReport(
        recordings_root=str(root),
        audit_path=str(audit),
        applied=True,
        target_directories=preflight.target_directories,
        target_files=target_files,
        media_before=media_before,
        media_after=media_after,
        media_unchanged=media_unchanged,
    )
    _write_audit(audit, report, state="complete")
    if not media_unchanged:
        raise RuntimeError("raw media changed during legacy-result cleanup")
    return report


def _legacy_targets(root: Path) -> tuple[Path, ...]:
    targets: list[Path] = []
    for session in root.iterdir():
        if not session.is_dir() or session.name.startswith("."):
            continue
        target = session / "perception" / "hand-tracking"
        if target.is_dir():
            _validate_target(root, target)
            targets.append(target)
    return tuple(sorted(targets))


def _validate_target(root: Path, target: Path) -> None:
    resolved = target.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("legacy cleanup target escapes recordings root")
    relative = resolved.relative_to(root)
    if len(relative.parts) != 3 or relative.parts[1:] != (
        "perception",
        "hand-tracking",
    ):
        raise ValueError("legacy cleanup target does not match the exact session path")


def _media_hashes(root: Path) -> tuple[HashedFile, ...]:
    media_roots = tuple(
        session / "media"
        for session in root.iterdir()
        if session.is_dir() and not session.name.startswith(".")
    )
    return _hashed_files(root, media_roots)


def _hashed_files(root: Path, directories: tuple[Path, ...]) -> tuple[HashedFile, ...]:
    files = sorted(
        path
        for directory in directories
        if directory.is_dir()
        for path in directory.rglob("*")
        if path.is_file()
    )
    return tuple(
        HashedFile(
            relative_path=_relative(root, path),
            size_bytes=path.stat().st_size,
            sha256=_sha256(path),
        )
        for path in files
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _write_audit(path: Path, report: LegacyCleanupReport, *, state: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "contract_id": "legacy-hand-tracking-cleanup-v1",
        "state": state,
        "written_at_unix_ns": time.time_ns(),
        **asdict(report),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with temporary.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
