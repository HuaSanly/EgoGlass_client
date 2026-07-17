# Ingest Gateway Evals

The periodic eval verifies RTSP recovery, bounded WebRTC metadata matching,
unbuffered viewer subscription, experimental IMU reception and replacement
isolation, and a real aiortc H.264 negotiation with a
browser-style receive-only offer. Credentials must never appear in
operator-visible status.

`test_recording_eval.py` drives the production PyAV recorder with synthetic
1920x1080 frames, waits for source-end finalization, and then reopens and
decodes the published MP4. It fails if the library exposes a partial file or if
the result is not H.264, 1920x1080, nominal 30 FPS, and fully decodable.
The same eval renames the session without moving its media, then deletes the
completed clip and requires both the media file and empty session directory to
disappear.

Run from this service directory:

~~~powershell
uv run pytest -q evals
~~~
