# Ingest Gateway Evals

The periodic eval verifies that an RTSP source can recover after a transient
disconnect and that credentials never appear in operator-visible status.

Run from this service directory:

~~~powershell
uv run pytest -q evals
~~~
