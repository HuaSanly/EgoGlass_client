# Interaction Processing

`interaction-processing` will consume video plus versioned spatial-perception
artifacts and produce interaction phases, object observations, tracked
keypoints, object poses, interaction trajectories, inpainted images, and
diagnostic visualizations. It does not read spatial-perception internals and it
does not define a training dataset format.

This package currently defines the interaction-processing boundary only. No
HumanEgo or other third-party source has been copied into the package.

## Directory purpose

| Path | Purpose |
| --- | --- |
| `src/perception/interaction_processing/` | Planned interaction-processing package. |
| `tests/` | Shared fast tests, identified by `test_interaction_processing_` filenames. |
| `evals/` | Shared periodic evaluations, identified by `test_interaction_processing_` filenames. |
| `config/` | Shared versioned client configuration. |
| `pyproject.toml` | Workspace package, tool, and dependency declaration. |

## Module purpose

| Module | Future responsibility |
| --- | --- |
| `pipeline.py` | Orchestrate the complete interaction-processing run. |
| `phase_segmentation.py` | Classify navigation, transition, and manipulation intervals. |
| `window_selection.py` | Select the interaction interval used downstream. |
| `object_segmentation.py` | Segment prompted objects, hands, and arms. |
| `keypoint_selection.py` | Select stable object points for tracking. |
| `keypoint_tracking.py` | Track selected points across the interaction interval. |
| `object_pose_estimation.py` | Estimate object pose using tracks and versioned camera poses. |
| `arm_inpainting.py` | Produce explicit arm-removed image derivatives. |
| `interaction_trajectory.py` | Consolidate hand, object, and grasp-state trajectories. |
| `visualization.py` | Generate review videos, plots, and point-cloud diagnostics. |
| `quality.py` | Report stage coverage, confidence, and rejection reasons. |
| `export.py` | Write an immutable interaction-processing artifact. |

## Planned boundary

Input will be a capture session plus a compatible `spatial-perception` artifact.
Output will be a future versioned `interaction-processing-v0` artifact. Reused
HumanEgo code will be placed by responsibility and recorded with exact source
commit and permission provenance when migration begins.

## Verification

```powershell
uv sync --group dev
uv run pytest
uv run pytest -q evals
uv run ruff check src tests evals
```
