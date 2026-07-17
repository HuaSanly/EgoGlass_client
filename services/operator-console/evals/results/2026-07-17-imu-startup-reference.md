# IMU startup reference device eval - 2026-07-17

## Scope

- Device: Rokid Glass3 Enterprise
- Sensors: TDK-InvenSense ICM4x6xx accelerometer and gyroscope
- Device sampling: approximately 101 Hz
- Client consumption: latest sample at approximately 20 Hz
- Start condition: Android application force-stopped, then launched normally
- Operator action: no manual IMU reset

## Pass criteria

- The automatic relative reference becomes ready within 2 seconds.
- Stationary pitch remains within +/-1 degree for at least 10 seconds after ready.
- The client does not render filter convergence as head motion before ready.

## Result

PASS.

- The reference became visible by the 1.13-second observation.
- Over the following 10.3 seconds, pitch stayed between -0.3 and 0.2 degrees.
- Roll stayed between -0.3 and 0.3 degrees.
- Relative yaw stayed between -0.1 and 0.1 degrees.
- The waiting overlay remained visible until the automatic reference was ready.

The glasses were stationary and temporarily kept awake through ADB for this
eval. The application and client were stopped afterward.
