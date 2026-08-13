from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ui.gateway.capture_recording import CaptureRecordingReader, CaptureRecordingReadError
from ui.gateway.recording_inspection import inspect_recording


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate one completed EgoGlass recording directory"
    )
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        reader = CaptureRecordingReader.open(args.path)
        video = inspect_recording(reader.video_path)
    except (CaptureRecordingReadError, OSError, ValueError) as error:
        print(f"recording validation failed: {error}", file=sys.stderr)
        return 1
    result = {
        "recording": reader.summary().model_dump(mode="json"),
        "calibration": reader.calibration.model_dump(mode="json"),
        "video": video.as_dict(),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
