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


def test_display_mapping_renders_nods_in_the_physical_direction() -> None:
    node = shutil.which("node")
    assert node is not None, "Node.js is required for the operator-console JavaScript gate"
    scene_uri = (STATIC_DIR / "imu-scene.js").resolve().as_uri()
    three_uri = (
        STATIC_DIR / "vendor" / "three.module-0.185.1.min.js"
    ).resolve().as_uri()
    script = f"""
import * as THREE from {json.dumps(three_uri)};
import {{ mapRelativeOrientationForDisplay }} from {json.dumps(scene_uri)};

const sourceDown = new THREE.Euler(-0.31, 0.22, -0.17, "XYZ");
const sourceUp = new THREE.Euler(0.31, 0.22, -0.17, "XYZ");
const mappedDown = mapRelativeOrientationForDisplay(
  new THREE.Quaternion().setFromEuler(sourceDown),
);
const mappedUp = mapRelativeOrientationForDisplay(
  new THREE.Quaternion().setFromEuler(sourceUp),
);
const renderedDown = new THREE.Euler().setFromQuaternion(mappedDown.quaternion, "XYZ");
const frontDown = new THREE.Vector3(0, 0, 1).applyQuaternion(mappedDown.quaternion);
const frontUp = new THREE.Vector3(0, 0, 1).applyQuaternion(mappedUp.quaternion);
console.log(JSON.stringify({{
  roll: mappedDown.angles.roll,
  pitch: mappedDown.angles.pitch,
  yaw: mappedDown.angles.yaw,
  renderedX: renderedDown.x,
  renderedY: renderedDown.y,
  renderedZ: renderedDown.z,
  frontDownY: frontDown.y,
  frontUpY: frontUp.y,
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

    assert mapped["roll"] == pytest.approx(-0.17)
    assert mapped["pitch"] == pytest.approx(0.31)
    assert mapped["yaw"] == pytest.approx(0.22)
    assert mapped["renderedX"] == pytest.approx(0.31)
    assert mapped["renderedY"] == pytest.approx(0.22)
    assert mapped["renderedZ"] == pytest.approx(-0.17)
    assert mapped["frontDownY"] < 0
    assert mapped["frontUpY"] > 0
