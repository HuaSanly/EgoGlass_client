import * as THREE from "./vendor/three.module-0.185.1.min.js";

const STANDARD_GRAVITY_M_S2 = 9.80665;
const REFERENCE_SAMPLE_COUNT = 8;

function createFrameBars(material, centerX) {
  const group = new THREE.Group();
  const horizontal = new THREE.BoxGeometry(1.42, 0.09, 0.1);
  const vertical = new THREE.BoxGeometry(0.09, 0.72, 0.1);
  const top = new THREE.Mesh(horizontal, material);
  const bottom = new THREE.Mesh(horizontal, material);
  const left = new THREE.Mesh(vertical, material);
  const right = new THREE.Mesh(vertical, material);
  top.position.set(centerX, 0.36, 0);
  bottom.position.set(centerX, -0.36, 0);
  left.position.set(centerX - 0.665, 0, 0);
  right.position.set(centerX + 0.665, 0, 0);
  group.add(top, bottom, left, right);
  return group;
}

function createGlassesModel() {
  const rig = new THREE.Group();
  const frameMaterial = new THREE.MeshStandardMaterial({
    color: 0x202824,
    metalness: 0.72,
    roughness: 0.28,
  });
  const edgeMaterial = new THREE.MeshStandardMaterial({
    color: 0x66d69a,
    emissive: 0x173d2b,
    metalness: 0.35,
    roughness: 0.36,
  });
  const lensMaterial = new THREE.MeshPhysicalMaterial({
    color: 0x6ca5ae,
    transparent: true,
    opacity: 0.2,
    roughness: 0.08,
    metalness: 0,
    transmission: 0.42,
    depthWrite: false,
  });

  rig.add(createFrameBars(frameMaterial, -0.82), createFrameBars(frameMaterial, 0.82));

  const lensGeometry = new THREE.BoxGeometry(1.27, 0.62, 0.035);
  const leftLens = new THREE.Mesh(lensGeometry, lensMaterial);
  const rightLens = new THREE.Mesh(lensGeometry, lensMaterial);
  leftLens.position.x = -0.82;
  rightLens.position.x = 0.82;
  rig.add(leftLens, rightLens);

  const bridge = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.1, 0.11), edgeMaterial);
  bridge.position.y = 0.08;
  rig.add(bridge);

  const armGeometry = new THREE.BoxGeometry(0.11, 0.14, 2.25);
  const leftArm = new THREE.Mesh(armGeometry, frameMaterial);
  const rightArm = new THREE.Mesh(armGeometry, frameMaterial);
  leftArm.position.set(-1.52, 0.22, -1.05);
  rightArm.position.set(1.52, 0.22, -1.05);
  leftArm.rotation.x = -0.04;
  rightArm.rotation.x = -0.04;
  rig.add(leftArm, rightArm);

  const sideModule = new THREE.Mesh(
    new THREE.BoxGeometry(0.25, 0.28, 0.92),
    edgeMaterial,
  );
  sideModule.position.set(1.52, 0.2, -0.52);
  rig.add(sideModule);

  const noseGeometry = new THREE.BoxGeometry(0.09, 0.28, 0.08);
  const leftNose = new THREE.Mesh(noseGeometry, frameMaterial);
  const rightNose = new THREE.Mesh(noseGeometry, frameMaterial);
  leftNose.position.set(-0.18, -0.34, -0.08);
  rightNose.position.set(0.18, -0.34, -0.08);
  leftNose.rotation.z = -0.28;
  rightNose.rotation.z = 0.28;
  rig.add(leftNose, rightNose);

  rig.traverse((object) => {
    if (object instanceof THREE.Mesh) {
      object.castShadow = true;
      object.receiveShadow = true;
    }
  });
  return rig;
}

function readAhrsConstructor() {
  const browserRequire = window.require;
  if (typeof browserRequire !== "function") {
    throw new Error("AHRS runtime is unavailable");
  }
  const constructor = browserRequire("ahrs");
  if (typeof constructor !== "function") {
    throw new Error("AHRS constructor is unavailable");
  }
  return constructor;
}

export function mapRelativeOrientationForDisplay(relative) {
  const sensorEuler = new THREE.Euler().setFromQuaternion(relative, "XYZ");
  const displayEuler = sensorEuler.clone();

  // Glass3 pitch is opposite to the Three.js display convention used by the model.
  displayEuler.y = -sensorEuler.y;
  return {
    quaternion: new THREE.Quaternion().setFromEuler(displayEuler).normalize(),
    euler: displayEuler,
  };
}

export class ImuSceneController {
  constructor(canvas, onOrientation) {
    this.canvas = canvas;
    this.onOrientation = onOrientation;
    this.ahrsConstructor = readAhrsConstructor();
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b0f0d);
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
    this.camera.position.set(4.2, 2.7, 4.1);
    this.camera.lookAt(0, 0, -0.45);
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;

    const hemisphere = new THREE.HemisphereLight(0xd5ede1, 0x1a2420, 1.45);
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.4);
    keyLight.position.set(3.5, 5.5, 4.5);
    keyLight.castShadow = true;
    this.scene.add(hemisphere, keyLight);

    const grid = new THREE.GridHelper(8, 16, 0x34443c, 0x1f2924);
    grid.position.y = -1.25;
    this.scene.add(grid);

    this.glasses = createGlassesModel();
    this.glasses.position.y = -0.05;
    this.scene.add(this.glasses);

    this.deviceAxes = new THREE.AxesHelper(1.2);
    this.glasses.add(this.deviceAxes);
    this.accelerationArrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 1, 0),
      new THREE.Vector3(0, 0, 0.08),
      1,
      0xe0aa52,
      0.2,
      0.1,
    );
    this.accelerationArrow.visible = false;
    this.glasses.add(this.accelerationArrow);

    this.targetQuaternion = new THREE.Quaternion();
    this.rawQuaternion = new THREE.Quaternion();
    this.referenceInverse = null;
    this.referenceSamples = 0;
    this.lastGyroscopeSequence = null;
    this.lastGyroscopeTimestampNs = null;
    this.sessionId = null;
    this.active = false;
    this.createFilter();

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas.parentElement);
    this.resize();
    this.animate = this.animate.bind(this);
    this.animationFrame = window.requestAnimationFrame(this.animate);
  }

  createFilter() {
    this.filter = new this.ahrsConstructor({
      sampleInterval: 10,
      algorithm: "Madgwick",
      beta: 0.18,
      doInitialisation: false,
    });
    this.referenceInverse = null;
    this.referenceSamples = 0;
    this.lastGyroscopeSequence = null;
    this.lastGyroscopeTimestampNs = null;
    this.targetQuaternion.identity();
  }

  beginSession(sessionId) {
    if (this.sessionId === sessionId) return;
    this.sessionId = sessionId;
    this.createFilter();
  }

  setActive(active) {
    this.active = active;
    this.accelerationArrow.visible = active;
  }

  resetReference() {
    if (this.rawQuaternion.lengthSq() > 0.5) {
      this.referenceInverse = this.rawQuaternion.clone().invert();
      this.referenceSamples = REFERENCE_SAMPLE_COUNT;
      this.targetQuaternion.identity();
      this.onOrientation({ ready: true, roll: 0, pitch: 0, yaw: 0 });
      return;
    }
    this.createFilter();
  }

  update(accelerometer, gyroscope) {
    if (gyroscope.sequence_number === this.lastGyroscopeSequence) return false;
    const timestampNs = gyroscope.sensor_event_monotonic_ns;
    let deltaSeconds = 0.01;
    if (this.lastGyroscopeTimestampNs !== null) {
      deltaSeconds = Math.min(
        0.12,
        Math.max(0.005, (timestampNs - this.lastGyroscopeTimestampNs) / 1_000_000_000),
      );
    }
    this.lastGyroscopeTimestampNs = timestampNs;
    this.lastGyroscopeSequence = gyroscope.sequence_number;

    const [gx, gy, gz] = gyroscope.values;
    const [ax, ay, az] = accelerometer.values.map((value) => value / STANDARD_GRAVITY_M_S2);
    this.filter.update(
      gx,
      gy,
      gz,
      ax,
      ay,
      az,
      undefined,
      undefined,
      undefined,
      deltaSeconds,
    );
    const quaternion = this.filter.getQuaternion();
    if (![quaternion.x, quaternion.y, quaternion.z, quaternion.w].every(Number.isFinite)) {
      this.createFilter();
      return false;
    }
    this.rawQuaternion.set(quaternion.x, quaternion.y, quaternion.z, quaternion.w).normalize();

    const acceleration = new THREE.Vector3(...accelerometer.values);
    const accelerationMagnitude = acceleration.length();
    if (accelerationMagnitude > 0.01) {
      this.accelerationArrow.setDirection(acceleration.normalize());
      this.accelerationArrow.setLength(
        Math.min(1.55, Math.max(0.55, accelerationMagnitude / STANDARD_GRAVITY_M_S2)),
        0.2,
        0.1,
      );
    }

    if (this.referenceInverse === null) {
      this.referenceSamples += 1;
      if (this.referenceSamples >= REFERENCE_SAMPLE_COUNT) {
        this.referenceInverse = this.rawQuaternion.clone().invert();
      } else {
        this.onOrientation({ ready: false, roll: 0, pitch: 0, yaw: 0 });
        return true;
      }
    }

    const relative = this.referenceInverse.clone().multiply(this.rawQuaternion).normalize();
    const displayOrientation = mapRelativeOrientationForDisplay(relative);
    this.targetQuaternion.copy(displayOrientation.quaternion);
    this.onOrientation({
      ready: true,
      roll: THREE.MathUtils.radToDeg(displayOrientation.euler.x),
      pitch: THREE.MathUtils.radToDeg(displayOrientation.euler.y),
      yaw: THREE.MathUtils.radToDeg(displayOrientation.euler.z),
    });
    return true;
  }

  resize() {
    const parent = this.canvas.parentElement;
    const width = Math.max(1, parent.clientWidth);
    const height = Math.max(1, parent.clientHeight);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  animate() {
    this.glasses.quaternion.slerp(this.targetQuaternion, this.active ? 0.2 : 0.08);
    this.renderer.render(this.scene, this.camera);
    this.animationFrame = window.requestAnimationFrame(this.animate);
  }

  dispose() {
    window.cancelAnimationFrame(this.animationFrame);
    this.resizeObserver.disconnect();
    this.renderer.dispose();
  }
}
