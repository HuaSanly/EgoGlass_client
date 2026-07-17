# Ingest Gateway Evals

The periodic eval verifies RTSP recovery, bounded WebRTC metadata matching,
unbuffered viewer subscription, experimental IMU reception and replacement
isolation, and a real aiortc H.264 negotiation with a
browser-style receive-only offer. Credentials must never appear in
operator-visible status.

`test_recording_eval.py` drives the production PyAV recorder with synthetic
1280x720 frames, waits for source-end finalization, and then reopens and
decodes the published MP4. It fails if the library exposes a partial file or if
the result is not H.264, 1280x720, nominal 30 FPS, and fully decodable.
The same eval renames the session without moving its media, then deletes the
completed clip and requires both the media file and empty session directory to
disappear.
The compatibility eval loads a pre-profile-change 1920x1080 manifest and
requires the session and media path to remain available while new recording
output stays fixed at 1280x720.

Run from this service directory:

~~~powershell
uv run pytest -q evals
~~~
