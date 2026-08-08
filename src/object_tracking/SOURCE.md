# Object Tracking Source

The pipeline structure is adapted from HumanEgo commit
`18fb1082abb87b79f88e53f2abb5bfb9f61de19b`:

- `preprocess/DINOSAM.py`
- `preprocess/KptsSelector.py`
- `preprocess/CoTrackerOffline.py`
- `preprocess/CamTriangulator.py`
- `preprocess/DatasetGen.py`

EgoGlass reimplements the adapters against its own schemas, Basalt poses, and
immutable capture sessions. It does not copy Project Aria I/O, HumanEgo UI,
or third-party model source. The algorithm adaptation is subject to the HumanEgo
PolyForm Noncommercial license and required notice in
`src/hand_tracking/HUMANEGO-LICENSE.txt`.

Third-party adapters load upstream packages rather than vendoring their source.
The code and model revisions are frozen in `pyproject.toml` and
`config/object-tracking.yaml`. Grounding DINO is Apache-2.0, SAM2 is
Apache-2.0, and CoTracker is CC-BY-NC-4.0; the latter is compatible with this
noncommercial HumanEgo-derived pipeline but must be reviewed before any
commercial dataset tooling release.
