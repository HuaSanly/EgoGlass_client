"""Composable object-processing stage used by the UI offline runner."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from hand_tracking.models import HandTrackingResult, TrackedHand
from schemas.object_tracking import (
    ObjectKeypointTrack,
    ObjectMaskObservation,
    ObjectPose,
    ObjectTrackingResult,
)
from schemas.phase import ObjectCentricWindow
from schemas.trajectory import VioTrajectory

from .config import ObjectTrackingConfig, ObjectTrackingError, TaskProfile
from .cotracker import CoTracker3Tracker, PointTracker
from .dino_sam import DinoSamSegmenter, ObjectSegmenter
from .keypoint_selector import ContourKeypointSelector
from .latching import ObjectPoseLatcher
from .triangulator import MultiViewTriangulator


@dataclass(frozen=True, slots=True)
class ObjectFrameInput:
    clip_id: str
    frame_index: int
    session_time_ns: int
    image_bgr: NDArray[np.uint8]
    intrinsics: NDArray[np.float64]
    hands: tuple[TrackedHand, ...]


class OfflineObjectProcessing:
    """Run the full DINO-SAM -> point track -> triangulation object stage."""

    def __init__(
        self,
        config: ObjectTrackingConfig,
        *,
        segmenter: ObjectSegmenter | None = None,
        tracker: PointTracker | None = None,
        triangulator: MultiViewTriangulator | None = None,
    ) -> None:
        self.config = config
        self.segmenter = segmenter or DinoSamSegmenter(config)
        self.tracker = tracker or CoTracker3Tracker(config)
        self.selector = ContourKeypointSelector(config)
        self.triangulator = triangulator or MultiViewTriangulator(config)

    def run(
        self,
        processing_run_id: str,
        profile: TaskProfile,
        windows: tuple[ObjectCentricWindow, ...],
        frames_by_clip: dict[str, tuple[ObjectFrameInput, ...]],
        trajectory: VioTrajectory,
        transform_camera_to_imu: NDArray[np.float64],
        output_directory: str | Path,
        *,
        is_canceled: Callable[[], bool] = lambda: False,
    ) -> ObjectTrackingResult:
        output = Path(output_directory).expanduser().resolve()
        masks_directory = output / "masks"
        masks_directory.mkdir(parents=True, exist_ok=True)
        masks: list[ObjectMaskObservation] = []
        tracks: list[ObjectKeypointTrack] = []
        triangulations = []
        selected_keypoints: list[dict[str, object]] = []
        poses_by_key: dict[tuple[str, int, str], ObjectPose] = {}
        selector = self.selector
        try:
            for window in windows:
                if is_canceled():
                    raise ObjectTrackingError("object processing canceled")
                source_frames = frames_by_clip.get(window.clip_id, ())
                window_frames = tuple(
                    frame
                    for frame in source_frames
                    if window.start_frame_index
                    <= frame.frame_index
                    < window.end_frame_index_exclusive
                )
                if not window_frames:
                    raise ObjectTrackingError("object-centric window contains no decoded frames")
                reference_offset = next(
                    (
                        index
                        for index, frame in enumerate(window_frames)
                        if frame.frame_index == window.reference_frame_index
                    ),
                    None,
                )
                if reference_offset is None:
                    raise ObjectTrackingError("object-centric reference frame is missing")
                for object_id, prompt in profile.object_prompts.items():
                    if is_canceled():
                        raise ObjectTrackingError("object processing canceled")
                    object_masks: list[ObjectMaskObservation] = []
                    mask_arrays: list[NDArray[np.uint8]] = []
                    for frame in window_frames:
                        prediction = self.segmenter.segment(frame.image_bgr, prompt)
                        mask = prediction.mask
                        mask_area_ratio = float(np.count_nonzero(mask)) / float(mask.size)
                        if mask_area_ratio < self.config.mask_min_area_ratio:
                            raise ObjectTrackingError(
                                f"{object_id} mask area is below the configured minimum"
                            )
                        relative_path = Path("masks") / f"{object_id}-{frame.frame_index:06d}.png"
                        absolute_path = output / relative_path
                        encoded, png = cv2.imencode(".png", mask)
                        if not encoded:
                            raise ObjectTrackingError(
                                f"failed to encode object mask: {relative_path}"
                            )
                        try:
                            absolute_path.write_bytes(png.tobytes())
                        except OSError as error:
                            raise ObjectTrackingError(
                                f"failed to write object mask: {relative_path}"
                            ) from error
                        observation = ObjectMaskObservation(
                            object_id=object_id,
                            clip_id=frame.clip_id,
                            frame_index=frame.frame_index,
                            session_time_ns=frame.session_time_ns,
                            mask_relative_path=relative_path.as_posix(),
                            width_px=int(mask.shape[1]),
                            height_px=int(mask.shape[0]),
                            boxes=prediction.boxes,
                            mask_area_ratio=mask_area_ratio,
                        )
                        masks.append(observation)
                        object_masks.append(observation)
                        mask_arrays.append(mask)
                    reference_points = selector.select(mask_arrays[reference_offset])
                    selected_keypoints.append(
                        {
                            "object_id": object_id,
                            "clip_id": window.clip_id,
                            "frame_index": window.reference_frame_index,
                            "points_xy_px": reference_points.tolist(),
                        }
                    )
                    tracks_xy, visibility = self.tracker.track(
                        [frame.image_bgr for frame in window_frames],
                        reference_points,
                        reference_offset,
                    )
                    track = ObjectKeypointTrack(
                        object_id=object_id,
                        clip_id=window.clip_id,
                        frame_indices=tuple(frame.frame_index for frame in window_frames),
                        session_times_ns=tuple(frame.session_time_ns for frame in window_frames),
                        points_xy_px=tuple(
                            tuple(tuple(float(value) for value in point) for point in frame_points)
                            for frame_points in tracks_xy
                        ),
                        visibility=tuple(
                            tuple(float(value) for value in frame_visibility)
                            for frame_visibility in visibility
                        ),
                    )
                    tracks.append(track)
                    observations = self.triangulator.camera_observations(
                        tuple(frame.frame_index for frame in window_frames),
                        tuple(frame.session_time_ns for frame in window_frames),
                        tuple(frame.intrinsics for frame in window_frames),
                        trajectory,
                        transform_camera_to_imu,
                    )
                    triangulation = self.triangulator.triangulate(
                        object_id,
                        observations,
                        tracks_xy,
                        visibility,
                        reference_offset,
                        profile.orientation_method,
                    )
                    triangulations.append(triangulation)
                    latcher = ObjectPoseLatcher(
                        object_id,
                        np.asarray(triangulation.transform_object_to_world).reshape(4, 4),
                        maximum_latch_distance_m=self.config.maximum_grasp_latch_distance_m,
                    )
                    for frame in window_frames:
                        pose = latcher.update(
                            frame.clip_id,
                            frame.frame_index,
                            frame.session_time_ns,
                            frame.hands,
                        )
                        poses_by_key[(object_id, frame.frame_index, window.clip_id)] = pose
                    self._write_object_artifacts(
                        output, object_id, object_masks, track, triangulation
                    )
        finally:
            close = getattr(self.segmenter, "close", None)
            if callable(close):
                close()
            close = getattr(self.tracker, "close", None)
            if callable(close):
                close()
        result = ObjectTrackingResult(
            processing_run_id=processing_run_id,
            task_profile_id=profile.profile_id,
            masks=tuple(masks),
            tracks=tuple(tracks),
            triangulations=tuple(triangulations),
            poses=tuple(poses_by_key.values()),
        )
        (output / "object-result.json").write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._write_stage_artifacts(output, result, selected_keypoints)
        return result

    @staticmethod
    def _write_stage_artifacts(
        output: Path,
        result: ObjectTrackingResult,
        selected_keypoints: list[dict[str, object]],
    ) -> None:
        with (output / "selected-keypoints.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            for item in selected_keypoints:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        (output / "tracks.json").write_text(
            json.dumps(
                [item.model_dump(mode="json") for item in result.tracks],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (output / "triangulation.json").write_text(
            json.dumps(
                [item.model_dump(mode="json") for item in result.triangulations],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        qa = {
            "schema_version": "1.0",
            "mask_count": len(result.masks),
            "track_count": len(result.tracks),
            "triangulation_count": len(result.triangulations),
            "pose_count": len(result.poses),
            "mean_reprojection_error_px": (
                sum(item.mean_reprojection_error_px for item in result.triangulations)
                / len(result.triangulations)
                if result.triangulations
                else None
            ),
        }
        (output / "object-qa.json").write_text(
            json.dumps(qa, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_object_artifacts(
        output: Path,
        object_id: str,
        masks: list[ObjectMaskObservation],
        track: ObjectKeypointTrack,
        triangulation: object,
    ) -> None:
        object_directory = output / object_id
        object_directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "object_id": object_id,
            "masks": [item.model_dump(mode="json") for item in masks],
            "track": track.model_dump(mode="json"),
            "triangulation": triangulation.model_dump(mode="json"),
        }
        (object_directory / "object-qa.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def hand_results_by_frame(
    results: tuple[HandTrackingResult, ...],
) -> dict[tuple[str, int], tuple[TrackedHand, ...]]:
    return {(result.sequence_id, result.frame_index): result.hands for result in results}
