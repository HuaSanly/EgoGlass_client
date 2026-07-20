# Operator Console Evals

The periodic eval suite measures the operator console contract separately from
gate tests. The shipped runtime consumes only the real Glass3 preview.

Current pass criteria:

- the Glass3 WebRTC viewer endpoint is the only viewer source;
- waiting, connected, and disconnected UI states follow real video frames;
- displayed FPS is measured from rendered WebView2 video frames;
- recording starts only from the gateway countdown timestamp, can be cancelled,
  and never uses a client-side placeholder state;
- the storage page presents time-named session folders, renders videos only
  after a folder is opened, persists custom names through the gateway, and
  exposes loading, empty, and error states;
- historical sessions retain and display each clip's recorded dimensions after
  the active capture profile changes;
- clip deletion requires confirmation and waits for a successful loopback
  gateway response before removing the card;
- the Three.js canvas uses the real loopback IMU endpoint and the vendored
  Madgwick filter, with no placeholder data generator or WebSocket;
- the event table stays below the live video, records bounded real runtime
  transitions, and clears without changing device or recording state;
- desktop authentication and repeated start/stop lifecycle remain recoverable;
- terminating the Windows launcher closes every descendant process and its TCP
  listener through the shared Job Object.

Run with:

```powershell
python -m uv run pytest -q evals
```

Real-device WebRTC and long-session evals remain separate named hardware gates.
