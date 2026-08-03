from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

CLIENT_ROOT = Path(__file__).resolve().parents[1]
if str(CLIENT_ROOT) not in sys.path:
    sys.path.insert(0, str(CLIENT_ROOT))

from perception.video_processing import cleanup_legacy_hand_tracking  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit and remove legacy per-session hand-tracking results"
    )
    parser.add_argument(
        "--recordings-root",
        type=Path,
        default=Path("local-data/recordings"),
    )
    parser.add_argument("--audit-path", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    audit_path = args.audit_path or (
        args.recordings_root
        / ".processing"
        / f"legacy-hand-tracking-cleanup-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    )
    report = cleanup_legacy_hand_tracking(
        args.recordings_root,
        audit_path,
        apply=args.apply,
    )
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
