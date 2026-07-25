# Interaction Processing

`interaction-processing` will consume video plus versioned spatial-perception
artifacts and produce interaction phases, object observations, tracked
keypoints, object poses, interaction trajectories, inpainted images, and
diagnostic visualizations. It does not read spatial-perception internals and it
does not define a training dataset format.

This commit creates the service boundary only. No HumanEgo or other third-party
source has been copied into the package.

## Directory purpose

| Path | Purpose |
| --- | --- |
| `src/egoglass_interaction_processing/` | Python package for interaction processing. |
| `tests/` | Fast deterministic tests for transforms and state machines. |
| `evals/` | Periodic segmentation, tracking, pose, and boundary evaluations. |
| `config/` | Versioned model, prompt, threshold, and phase configuration. |
| `pyproject.toml` | Package, tool, and dependency declaration. |

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
