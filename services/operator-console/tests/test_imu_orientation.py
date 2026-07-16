import json
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC_DIR = (
    Path(__file__).parents[1]
    / "src"
    / "egoglass_operator_console"
    / "static"
)


def test_display_mapping_reverses_pitch_without_changing_roll_or_yaw() -> None:
    node = shutil.which("node")
    assert node is not None, "Node.js is required for the operator-console JavaScript gate"
    scene_uri = (STATIC_DIR / "imu-scene.js").resolve().as_uri()
    three_uri = (
        STATIC_DIR / "vendor" / "three.module-0.185.1.min.js"
    ).resolve().as_uri()
    script = f"""
import * as THREE from {json.dumps(three_uri)};
import {{ mapRelativeOrientationForDisplay }} from {json.dumps(scene_uri)};

const source = new THREE.Euler(0.22, 0.31, -0.17, "XYZ");
const relative = new THREE.Quaternion().setFromEuler(source);
const mapped = mapRelativeOrientationForDisplay(relative);
const rendered = new THREE.Euler().setFromQuaternion(mapped.quaternion, "XYZ");
console.log(JSON.stringify({{
  roll: mapped.euler.x,
  pitch: mapped.euler.y,
  yaw: mapped.euler.z,
  renderedRoll: rendered.x,
  renderedPitch: rendered.y,
  renderedYaw: rendered.z,
}}));
"""

    completed = subprocess.run(
        [node, "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    mapped = json.loads(completed.stdout)

    assert mapped["roll"] == pytest.approx(0.22)
    assert mapped["pitch"] == pytest.approx(-0.31)
    assert mapped["yaw"] == pytest.approx(-0.17)
    assert mapped["renderedRoll"] == pytest.approx(0.22)
    assert mapped["renderedPitch"] == pytest.approx(-0.31)
    assert mapped["renderedYaw"] == pytest.approx(-0.17)
