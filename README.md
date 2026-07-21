# EgoGlass Client

The client repository owns the Windows ingest, online inference orchestration,
operator tooling, and data platform. The current runnable slice receives direct
Glass3 WebRTC video and renders its live preview in the operator console. It
does not generate placeholder trajectories, metrics, calibration, or session
data.

## Services

- `services/ingest-gateway/`: terminates the direct Glass3 WebRTC video and
  frame-metadata channels, relays the live track to one loopback viewer,
  and records operator-selected H.264 MP4 clips.
- `services/operator-console/`: authenticated local UI for the real Glass3
  WebRTC track, measured displayed FPS, stream and recording controls, relative
  IMU orientation, and the local recording library.
- `services/data-platform/`: loopback-only task-attempt annotation service with
  frame-aligned proposals, drafts, quality checks, and immutable revisions.

Future services will be added behind versioned contracts rather than imported
from the operator console.

## Windows desktop

Start the complete client from the repository root:

```powershell
.\scripts\start-client.ps1
```

The command starts the LAN ingest gateway and local data platform, enables
Glass3 auto-discovery, and opens the native Windows operator console. Leave the
PowerShell command running
while using EgoGlass. Closing the Windows application or pressing `Ctrl+C` in
the launcher stops all three process trees and releases ports `8770`, `8771`,
and `8780`. A Windows Job Object also releases them if the launcher is
terminated without running its normal cleanup. After the client reports ready,
open EgoGlass directly from the Glass3 application list; no ADB launch
parameters are required.

Completed recordings are grouped by Glass3 WebRTC session under
`local-data/recordings/`. The entire `local-data/` tree is ignored by Git and
must not be committed.

The native annotation page supports manual task-attempt boundaries, whole-clip
proposals, non-overlapping fixed windows, episode labels, internal action
phases, autosave, undo/redo, and immutable publishing. It writes only beneath
each session's ignored `annotations/` directory and never edits source MP4 or
telemetry.

For individual operator-console development only:

```powershell
cd services/operator-console
uv sync --group dev
uv run egoglass-desktop
```

The desktop command opens the operator console in a native Windows WebView2
window. It does not open the system browser. The bundled FastAPI server uses a
dynamic loopback port and shuts down with the window.

Build and verify the local executable with:

```powershell
cd services/operator-console
.\scripts\build-desktop.ps1
```

The ignored build output is `dist/EgoGlass/EgoGlass.exe`.

## Verification

```powershell
cd services/operator-console
uv run pytest
uv run pytest -q evals
uv run ruff check src tests evals
```

Run the same three commands from `services/data-platform` for annotation
persistence and publishing.

The direct video path has passed a real-device first-frame check. World-aligned
feedback is a separate future service and must carry a verified calibration
profile before the glasses application is allowed to render it.
