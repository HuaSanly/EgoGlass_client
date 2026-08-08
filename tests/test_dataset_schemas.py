import hashlib

import pytest

from schemas import (
    ArtifactReference,
    DatasetEpisode,
    DatasetManifest,
    DatasetSplit,
    ObjectKeypointTrack,
    ObjectPose,
    ObjectTriangulation,
    QualityGate,
    QualityIssue,
    QualitySeverity,
    VirtualVideoSpan,
)


def _artifact(path: str) -> ArtifactReference:
    return ArtifactReference(
        relative_path=path,
        sha256=hashlib.sha256(path.encode()).hexdigest(),
        media_type="application/jsonl",
    )


def test_virtual_span_and_manifest_are_strict_and_json_safe() -> None:
    span = VirtualVideoSpan(
        session_id="session",
        clip_id="clip",
        start_frame_index=10,
        end_frame_index_exclusive=20,
        start_session_time_ns=100,
        end_session_time_ns=200,
        media=_artifact("media/clip.mp4"),
    )
    episode = DatasetEpisode(
        episode_id="episode",
        span=span,
        processing_run_id="run",
        vio_run_id="vio",
        annotation_revision_id="annotation",
    )
    manifest = DatasetManifest(
        dataset_id="dataset",
        created_at_unix_ns=1,
        random_seed=17,
        episodes_artifact=_artifact("episodes.jsonl"),
        samples_artifact=_artifact("samples.jsonl"),
        quality_report_artifact=_artifact("quality-report.json"),
        provenance_artifact=_artifact("provenance.json"),
        splits_artifact=_artifact("splits.json"),
        splits=(
            DatasetSplit(name="train", session_ids=("session",), episode_ids=(episode.episode_id,)),
        ),
        source_session_ids=("session",),
    )
    assert manifest.model_dump(mode="json")["splits"][0]["name"] == "train"
    with pytest.raises(ValueError):
        ArtifactReference(relative_path="unknown", sha256="bad", media_type="text/plain")


def test_hard_quality_issue_cannot_be_restored() -> None:
    with pytest.raises(ValueError):
        QualityIssue(
            issue_id="vio",
            gate=QualityGate.HARD,
            severity=QualitySeverity.ERROR,
            message="partial VIO",
            clip_id="clip",
            start_frame_index=0,
            end_frame_index_exclusive=1,
            restorable=True,
        )


def test_object_track_rejects_non_finite_points_and_out_of_range_visibility() -> None:
    values = dict(
        object_id="cup",
        clip_id="clip",
        frame_indices=(0, 1),
        session_times_ns=(0, 1),
        points_xy_px=(((10.0, 20.0),), ((11.0, 21.0),)),
        visibility=((1.0,), (1.0,)),
    )
    with pytest.raises(ValueError, match="finite"):
        ObjectKeypointTrack(
            **{
                **values,
                "points_xy_px": (((float("nan"), 20.0),), ((11.0, 21.0),)),
            }
        )
    with pytest.raises(ValueError, match="visibility"):
        ObjectKeypointTrack(**{**values, "visibility": ((1.1,), (1.0,))})


def test_object_geometry_rejects_invalid_homogeneous_transforms() -> None:
    identity = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    non_homogeneous = (*identity[:12], 0.0, 0.0, 0.0, 2.0)
    singular = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="homogeneous"):
        ObjectPose(
            object_id="cup",
            clip_id="clip",
            frame_index=0,
            session_time_ns=0,
            transform_object_to_world=non_homogeneous,
            source="triangulation",
        )
    with pytest.raises(ValueError, match="invertible"):
        ObjectTriangulation(
            object_id="cup",
            points_world_m=((0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.0, 0.1, 1.0)),
            transform_object_to_world=identity,
            transform_object_to_camera=singular,
            mean_reprojection_error_px=0.1,
            contributing_frame_count=2,
            valid_point_count=3,
            orientation_method="pca",
        )
