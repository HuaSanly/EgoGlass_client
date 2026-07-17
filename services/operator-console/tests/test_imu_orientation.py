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


def test_startup_reference_waits_for_stationary_filter_convergence() -> None:
    node = shutil.which("node")
    assert node is not None, "Node.js is required for the operator-console JavaScript gate"
    scene_uri = (STATIC_DIR / "imu-scene.js").resolve().as_uri()
    script = f"""
import {{ ImuFusionTracker, isStableReferenceSample }} from {json.dumps(scene_uri)};

let filterInstance;
class FakeAhrs {{
  constructor(options) {{
    this.options = options;
    this.quaternion = {{ w: 1, x: 0, y: 0, z: 0 }};
    this.initCalls = [];
    this.updateCalls = 0;
    filterInstance = this;
  }}

  init(ax, ay, az, mx, my, mz) {{
    this.initCalls.push([ax, ay, az, mx, my, mz]);
  }}

  update() {{
    this.updateCalls += 1;
    const angle = Math.min(this.updateCalls, 5) * Math.PI / 90;
    this.quaternion = {{
      w: Math.cos(angle / 2),
      x: Math.sin(angle / 2),
      y: 0,
      z: 0,
    }};
  }}

  getQuaternion() {{
    return this.quaternion;
  }}
}}

const tracker = new ImuFusionTracker(FakeAhrs);
const update = (sequence, gyroscopeValues = [0, 0, 0]) => tracker.update(
  {{ values: [0, 0, 9.80665] }},
  {{
    sequence_number: sequence,
    sensor_event_monotonic_ns: sequence * 10_000_000,
    values: gyroscopeValues,
  }},
);

for (let sequence = 1; sequence <= 3; sequence += 1) update(sequence);
const moving = update(4, [0.5, 0, 0]);
for (let sequence = 5; sequence <= 9; sequence += 1) update(sequence);
const beforeInitialisation = filterInstance.initCalls.length;
const initialised = update(10);
let lastPending = initialised;
for (let sequence = 11; sequence <= 26; sequence += 1) {{
  lastPending = update(sequence);
}}
const ready = update(27);
const relative = ready.relative;

console.log(JSON.stringify({{
  movingReady: moving.ready,
  beforeInitialisation,
  initialisedReady: initialised.ready,
  pendingReady: lastPending.ready,
  ready: ready.ready,
  relativeAngle: 2 * Math.acos(Math.min(1, Math.abs(relative.w))),
  initCalls: filterInstance.initCalls,
  beta: filterInstance.options.beta,
  doInitialisation: filterInstance.options.doInitialisation,
  stableSample: isStableReferenceSample([0, 0, 9.80665], [0, 0, 0]),
  movingSample: isStableReferenceSample([0, 0, 9.80665], [0.5, 0, 0]),
}}));
"""

    completed = subprocess.run(
        [node, "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    result = json.loads(completed.stdout)

    assert result["movingReady"] is False
    assert result["beforeInitialisation"] == 0
    assert result["initialisedReady"] is False
    assert result["pendingReady"] is False
    assert result["ready"] is True
    assert result["relativeAngle"] == pytest.approx(0, abs=1e-9)
    assert result["initCalls"][0][:2] == [0, 0]
    assert result["initCalls"][0][2] == pytest.approx(9.80665)
    assert result["initCalls"][0][3:] == [1, 0, 0]
    assert result["beta"] == pytest.approx(0.05)
    assert result["doInitialisation"] is False
    assert result["stableSample"] is True
    assert result["movingSample"] is False


def test_vendored_ahrs_stationary_startup_stays_near_reference() -> None:
    node = shutil.which("node")
    assert node is not None, "Node.js is required for the operator-console JavaScript gate"
    scene_uri = (STATIC_DIR / "imu-scene.js").resolve().as_uri()
    ahrs_path = (STATIC_DIR / "vendor" / "ahrs-1.3.3.js").resolve()
    script = f"""
import {{ readFileSync }} from "node:fs";
import {{ createContext, runInContext }} from "node:vm";
import {{
  ImuFusionTracker,
  mapRelativeOrientationForDisplay,
}} from {json.dumps(scene_uri)};

const context = {{}};
createContext(context);
runInContext(readFileSync({json.dumps(str(ahrs_path))}, "utf8"), context);
const Ahrs = context.require("ahrs");
const tracker = new ImuFusionTracker(Ahrs);
let firstReady = null;
let maxAbsPitch = 0;
let finalPitch = null;

for (let sequence = 1; sequence <= 240; sequence += 1) {{
  const result = tracker.update(
    {{
      values: [
        0.015 * Math.sin(sequence * 0.37),
        0.012 * Math.cos(sequence * 0.29),
        9.80665 + 0.01 * Math.sin(sequence * 0.19),
      ],
    }},
    {{
      sequence_number: sequence,
      sensor_event_monotonic_ns: sequence * 10_000_000,
      values: [
        0.0005 * Math.sin(sequence * 0.17),
        0.0004 * Math.cos(sequence * 0.13),
        0.0003 * Math.sin(sequence * 0.11),
      ],
    }},
  );
  if (result.ready && result.relative) {{
    if (firstReady === null) firstReady = sequence;
    const display = mapRelativeOrientationForDisplay(result.relative);
    finalPitch = display.angles.pitch * 180 / Math.PI;
    maxAbsPitch = Math.max(maxAbsPitch, Math.abs(finalPitch));
  }}
}}

console.log(JSON.stringify({{ firstReady, maxAbsPitch, finalPitch }}));
"""

    completed = subprocess.run(
        [node, "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    result = json.loads(completed.stdout)

    assert result["firstReady"] is not None
    assert result["firstReady"] <= 40
    assert result["maxAbsPitch"] < 1.0
    assert abs(result["finalPitch"]) < 0.5
