# Data Platform

The data platform is a loopback-only service for non-destructive annotation of
completed EgoGlass capture sessions. It reads capture-session-v1 manifests,
serves their local MP4 files, resolves selected frame boundaries against the
persisted MP4 PTS index, autosaves conflict-checked drafts, and publishes
immutable content-addressed annotation revisions.

Raw captures normally have pending perception alignment. Annotation continues
to use exact frame indices and MP4 PTS; it preserves `session_time_ns=null` and
`timing_status=unmapped` instead of inventing a cross-modal timestamp.

The initial proposal providers are whole-clip and fixed-window segmentation.
Manual editing happens in the operator console. Event-marker, motion-change,
hand-object interaction, and VLM providers are declared future capabilities;
the service does not return placeholder proposals for them.

Annotation state stays beside its source session:

```text
<session_id>/annotations/episode-annotation-v1/
  draft.json
  latest.json
  revisions/<annotation_revision_id>.json
```

Source MP4, telemetry SQLite, frame metadata, and capture manifests are never
modified by annotation.

## Run

```powershell
uv sync --group dev
uv run egoglass-data-platform --recordings-root ..\..\local-data\recordings
```

The complete workspace launcher starts this service automatically on
`127.0.0.1:8780`.

## API

- `GET /api/v1/annotations/workspace`
- `GET /api/v1/annotations/sessions/{session_id}`
- `PUT /api/v1/annotations/sessions/{session_id}/draft`
- `POST /api/v1/annotations/sessions/{session_id}/proposals`
- `POST /api/v1/annotations/sessions/{session_id}/publish`
- `GET /api/v1/annotations/media/{session_id}/{clip_id}`

## Verification

```powershell
uv run pytest
uv run pytest -q evals
uv run ruff check src tests evals
```
