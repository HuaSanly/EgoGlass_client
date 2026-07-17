from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from egoglass_ingest_gateway.recording_inspection import inspect_recording


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate one completed EgoGlass H.264 MP4 recording"
    )
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        result = inspect_recording(args.path)
    except (OSError, ValueError) as error:
        print(f"recording validation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
