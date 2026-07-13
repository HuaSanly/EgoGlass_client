# EgoGlass Client Repository Guide

This file is authoritative for the standalone `EgoGlass_client` repository. It
also extends the superproject `AGENTS.md` when checked out as a submodule.

## Repository workflow

- Use `main` as the protected integration branch and Conventional Commits with
  the owning service as scope.
- Every service ships its own deterministic tests, periodic evals, README, and
  configuration in the same commit as a feature or fix.
- Use Semantic Versioning for releases and push the submodule commit before the
  superproject pointer is updated.
- Never commit credentials, raw recordings, datasets, generated outputs, or IDE
  state.

## Ownership

The Python client subproject owns:

- Device/phone media and signaling ingress.
- WebRTC termination or adaptation, stream visualization, and session health.
- Online inference orchestration and feedback delivery to the glasses.
- Raw-data persistence, timestamp alignment, quality correction, annotation,
  replay, dataset registration, and dataset version management.

It does not own Android capture internals or offline model training.

## Service boundaries

Keep the ingest gateway, online inference runtime, operator interface, and data
platform independently testable. They exchange versioned contracts and stable
storage references. They must not import each other's private modules or share
mutable process state.

The device stream format and the cloud transport are separate concerns. An
adapter may convert SDK-provided NV21 frames into a WebRTC media track, but the
ingest contract must preserve the source format and timestamps.

## Python rules

- Use a repository-selected Python version and a committed lock file once the
  first package is created. Do not mix package managers inside this subtree.
- Use type hints at service boundaries and validate all external messages.
- Use UTC for wall-clock times and explicit monotonic clocks for durations.
  Never compare timestamps from different clocks without a recorded mapping.
- Store large media and tabular data outside Git. Repositories store manifests,
  schemas, small fixtures, and immutable content references only.
- External side effects belong behind interfaces so gate tests stay local,
  deterministic, and under two seconds.
- Do not call hosted LLM APIs unless the project owner explicitly authorizes a
  change to the root policy.

## Verification

- Each service has its own unit/integration tests, eval suite, README, config,
  and runnable entry point.
- Ingest tests cover reorder, duplicate, disconnect, resume, malformed metadata,
  backpressure, and clock discontinuity.
- Data tests cover idempotent registration, alignment bounds, corrections,
  lineage, and schema compatibility.
- Online inference evals report latency percentiles, dropped-input rate,
  prediction quality, and feedback delivery success.
- Test fixtures must be tiny, synthetic, redistributable, and documented.

No service is allowed to mutate a registered dataset version. Corrections create
a new version with lineage back to the source.
