from __future__ import annotations

import sqlite3
from pathlib import Path

import av

from sensor_preprocessing import (
    CaptureSessionReader,
)
from tests.test_sensor_preprocessing_pipeline import CLIP_ID, _recorded_session
from ui.processing import ExportSummary, ProcessingResultStore, export_annotated_clip


def test_export_summary_reports_the_full_output_path(tmp_path: Path) -> None:
    output = tmp_path / "exports" / "annotated.mp4"

    assert ExportSummary(output, 42).detail == f"已导出 42 帧到 {output}"


def test_annotated_export_reuses_original_frames_and_structured_results(
    tmp_path: Path,
) -> None:
    session, _timings = _recorded_session(tmp_path)
    with sqlite3.connect(session / "telemetry" / "telemetry.sqlite") as database:
        database.execute(
            """
            UPDATE video_frame_index
            SET alignment_status = 'mapped',
                session_time_ns = 1000000000 + frame_index * 33333333,
                timestamp_uncertainty_ns = 1000,
                clock_mapping_segment_id = 'test-mapping'
            """
        )
        database.commit()
        database.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    reader = CaptureSessionReader.open(session)
    frames = tuple(reader.iter_frames(CLIP_ID))
    run = session / "derived" / "video-processing" / "run-test"
    store = ProcessingResultStore(run / "results.sqlite")
    for frame in frames:
        assert frame.stored_alignment.session_time_ns is not None
        store.put(
            {
                "session_id": reader.session.session_id,
                "sequence_id": CLIP_ID,
                "frame_index": frame.frame_index,
                "session_time_ns": frame.stored_alignment.session_time_ns,
                "source_image_width_px": 8,
                "source_image_height_px": 6,
                "hands": [
                    {
                        "handedness": "left",
                        "source_keypoints_2d_px": [],
                        "source_bbox_xyxy_px": [1, 1, 6, 5],
                    }
                ],
            }
        )

    summary = export_annotated_clip(session, run, CLIP_ID, hold_previous_frames=0)

    assert summary.frame_count == 2
    assert summary.path == run / "exports" / f"annotated-{CLIP_ID}.mp4"
    with av.open(str(summary.path)) as container:
        assert container.streams.video[0].codec_context.name == "h264"
        assert len(list(container.decode(container.streams.video[0]))) == 2
