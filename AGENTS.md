# EgoGlass Client Repository Guide

This file is authoritative for the standalone `EgoGlass_client` repository. It
also extends the superproject `AGENTS.md` when checked out as a submodule.

## Repository workflow

- Keep `main` stable, runnable, and usable. Every code, dependency, build,
  contract, CI, release, or runtime-affecting configuration change uses a
  short-lived branch created from an up-to-date `main`.
- Small documentation, wording, comment, and repository-governance changes that
  cannot affect build or runtime behavior may be committed directly to `main`
  after a local diff and formatting check. If the impact is uncertain, use a
  branch.
- Use simple branch names such as `feat/stream-ingest`, `fix/clock-alignment`, or
  `chore/dependency-update`. Keep the suffix to one to three clear kebab-case
  words.
- Merge only after the required tests and evals pass. Delete merged, abandoned,
  superseded, and otherwise unused branches locally and remotely as soon as
  they no longer serve active work.
- Use Conventional Commits with the owning service as scope.
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

## Client workspace boundaries

The client uses one native Windows Conda environment named `egoglass`, one root
`environment.yml`, one root `pyproject.toml`, one `tests/` directory, and one
`evals/` directory. Do not add package-local environments, lock files, test
trees, eval trees, or project manifests.

Top-level code packages live directly under `src/` and use concise
responsibility names: `ingest_gateway`, `operator_console`, and `perception`.
The `perception` package is the independently reusable research and runtime
core; its processing stages live beneath that package. The operator console
owns annotation UI, validation, and annotation persistence. Do not add an
`egoglass_` prefix to internal Python package names. Keep these packages
independently testable, and do not import another package's private modules or
share mutable process state.

The device stream format and the cloud transport are separate concerns. An
adapter may convert SDK-provided NV21 frames into a WebRTC media track, but the
ingest contract must preserve the source format and timestamps.

## Python rules

- Use Python 3.11 from the repository Conda environment. Do not use `uv`,
  virtualenvs, package-local environments, or nested Python projects in this
  subtree. Third-party HaMeR compatibility installs remain recorded in
  `scripts/setup_client.ps1`.
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

- Package tests and evals live in the shared root directories and use the
  owning package name as a filename prefix.
- Run `conda run -n egoglass python -m pytest`,
  `conda run -n egoglass python -m pytest -q evals`, and
  `conda run -n egoglass ruff check src tests evals` from the client root.
- Ingest tests cover reorder, duplicate, disconnect, resume, malformed metadata,
  backpressure, and clock discontinuity.
- Data tests cover idempotent registration, alignment bounds, corrections,
  lineage, and schema compatibility.
- Online inference evals report latency percentiles, dropped-input rate,
  prediction quality, and feedback delivery success.
- Test fixtures must be tiny, synthetic, redistributable, and documented.

No service is allowed to mutate a registered dataset version. Corrections create
a new version with lineage back to the source.
