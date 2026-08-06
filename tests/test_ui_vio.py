from __future__ import annotations

import json
from pathlib import Path

from schemas import VioPose, VioTrajectory
from ui.processing import OfflineVioService, VioRunState
from ui.widgets.spatial_sync_canvas import SpatialSyncCanvas


def test_offline_vio_service_discovers_viewable_run(tmp_path: Path) -> None:
    session_id = "1" * 32
    run_id = "20260806T000000Z-test1234"
    run_directory = tmp_path / session_id / "derived" / "vio" / "basalt" / run_id
    run_directory.mkdir(parents=True)
    (run_directory / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "session_id": session_id,
                "clip_id": None,
                "state": "completed",
                "started_at_unix_ns": 1,
                "completed_at_unix_ns": 2,
            }
        ),
        encoding="utf-8",
    )
    (run_directory / "trajectory.csv").write_text(
        "# header\n1000,0,0,0,1,0,0,0\n2000,1,0,0,1,0,0,0\n",
        encoding="utf-8",
    )

    runs = OfflineVioService(tmp_path).list_runs(session_id)

    assert len(runs) == 1
    assert runs[0].state is VioRunState.COMPLETED
    assert runs[0].is_viewable
    assert runs[0].pose_count == 2
    assert runs[0].pose_at(1_600).timestamp_ns == 2_000


def test_spatial_canvas_renders_offline_vio_pose_and_trajectory(qt_application) -> None:
    canvas = SpatialSyncCanvas()
    trajectory = VioTrajectory(
        (
            VioPose(1_000, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            VioPose(2_000, (0.2, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        )
    )
    canvas.set_vio_trajectory(trajectory, trajectory.poses[1])

    status = canvas.status()

    assert status.has_vio_pose
    assert status.trajectory_pose_count == 2
    canvas.close()
