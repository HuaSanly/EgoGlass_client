# Operator Console Evals

The periodic eval suite measures the operator console contract separately from
gate tests. The shipped runtime consumes only the real Glass3 preview.

Current pass criteria:

- the Glass3 WebRTC viewer endpoint is the only viewer source;
- waiting, connected, and disconnected UI states follow real video frames;
- displayed FPS is measured from rendered WebView2 video frames;
- recording starts only from the gateway countdown timestamp, can be cancelled,
  and never uses a client-side placeholder state;
- the storage page groups validated, playable loopback media by real WebRTC
  session and exposes loading, empty, and error states;
- clip deletion requires confirmation and waits for a successful loopback
  gateway response before removing the card;
- the Three.js canvas uses the real loopback IMU endpoint and the vendored
  Madgwick filter, with no placeholder data generator or WebSocket;
- the event table and its mutable history are absent from the shipped UI;
- desktop authentication and repeated start/stop lifecycle remain recoverable.

Run with:

```powershell
python -m uv run pytest -q evals
```

Real-device WebRTC and long-session evals remain separate named hardware gates.
