# Ingest Gateway Evals

The periodic eval verifies RTSP recovery, bounded WebRTC metadata matching,
unbuffered viewer subscription, experimental IMU reception and replacement
isolation, and a real aiortc H.264 negotiation with a
browser-style receive-only offer. Credentials must never appear in
operator-visible status.

Run from this service directory:

~~~powershell
uv run pytest -q evals
~~~
