from pathlib import Path

import numpy as np

from ingest_gateway.live_frames import LiveFrame, LiveFramePacer


def test_native_runtime_uses_direct_frames_and_one_process() -> None:
    repository = Path(__file__).parents[1]
    runtime = (repository / "ui" / "runtime.py").read_text(encoding="utf-8")
    video = (repository / "ui" / "widgets" / "video_surface.py").read_text(
        encoding="utf-8"
    )

    assert "threading.Thread" in runtime
    assert "uvicorn.Server" in runtime
    assert "create_app(" in runtime
    assert "run_coroutine_threadsafe" in runtime
    assert "requests" not in runtime
    assert "httpx" not in runtime
    assert "subprocess" not in runtime
    assert "multiprocessing" not in runtime
    assert "add_raw_texture" in video
    assert "source.subscribe(buffered=False)" in (
        repository / "src" / "ingest_gateway" / "adapters" / "aiortc_peer.py"
    ).read_text(encoding="utf-8")
    assert "rtp_packet_loss_percent" in (
        repository / "ui" / "views" / "diagnostics.py"
    ).read_text(encoding="utf-8")
    assert "mvFormat_Float_rgb" in video
    assert "jpeg" not in video.lower()
    assert "mjpg" not in video.lower()
    assert "library_refresh_at_ns" not in runtime
    assert "library_task = asyncio.create_task(self._initial_library_refresh())" in runtime
    assert "self.runtime.request_library_refresh()" in (
        repository / "ui" / "views" / "library.py"
    ).read_text(encoding="utf-8")


def test_native_video_path_has_rgb_fanout_double_buffering_and_per_frame_overlay() -> None:
    repository = Path(__file__).parents[1]
    live_frames = (repository / "src" / "ingest_gateway" / "live_frames.py").read_text(
        encoding="utf-8"
    )
    video = (repository / "ui" / "widgets" / "video_surface.py").read_text(
        encoding="utf-8"
    )
    live_view = (repository / "ui" / "views" / "live.py").read_text(encoding="utf-8")

    assert "submit_rgb_frame" in live_frames
    assert "_texture_buffers" in video
    assert "_front_texture_index" in video
    assert "_perception_result_key" in live_view
    assert "frame_index" in live_view
    assert "recent_upload_fps" in video
    assert "LiveFramePacer" in live_frames
    assert "maximum_queue_frames: int = 4" in live_frames
    assert "video_pts_ns" in live_frames
    assert "next_for_display" in (
        repository / "ui" / "runtime.py"
    ).read_text(encoding="utf-8")
    assert "_update_active_view" in (
        repository / "ui" / "app.py"
    ).read_text(encoding="utf-8")


def test_pts_pacer_smooths_bursty_lan_arrivals_without_unbounded_latency() -> None:
    frame_interval_ns = 33_333_333
    render_interval_ns = 16_666_667
    arrival_gaps_ns = (12_000_000, 54_666_666, 29_000_000, 37_666_666)
    arrival_ns = 0
    arrivals: list[tuple[int, LiveFrame]] = []
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    image.setflags(write=False)
    for index in range(180):
        if index:
            arrival_ns += arrival_gaps_ns[(index - 1) % len(arrival_gaps_ns)]
        arrivals.append(
            (
                arrival_ns,
                LiveFrame(
                    session_id="session",
                    connection_session_id="connection",
                    frame_index=index,
                    received_at_client_monotonic_ns=arrival_ns,
                    converted_at_client_monotonic_ns=arrival_ns,
                    image_rgb=image,
                    video_pts_ns=index * frame_interval_ns,
                ),
            )
        )

    pacer = LiveFramePacer()
    presented: list[tuple[int, LiveFrame]] = []
    arrival_index = 0
    previous_index = -1
    now_ns = 0
    deadline_ns = arrivals[-1][0] + 500_000_000
    maximum_queue_depth = 0
    while now_ns <= deadline_ns:
        while arrival_index < len(arrivals) and arrivals[arrival_index][0] <= now_ns:
            pacer.enqueue(arrivals[arrival_index][1])
            arrival_index += 1
        frame = pacer.next_frame(now_ns)
        status = pacer.status()
        maximum_queue_depth = max(maximum_queue_depth, status.queue_depth)
        if frame is not None and frame.frame_index != previous_index:
            presented.append((now_ns, frame))
            previous_index = frame.frame_index
        now_ns += render_interval_ns

    presentation_gaps_ms = np.diff([time_ns for time_ns, _ in presented]) / 1_000_000
    presentation_latency_ms = np.asarray(
        [
            (time_ns - frame.video_pts_ns) / 1_000_000
            for time_ns, frame in presented
            if frame.video_pts_ns is not None
        ]
    )
    assert len(presented) >= 170
    assert maximum_queue_depth <= 4
    assert np.percentile(presentation_gaps_ms, 95) <= 50.1
    assert np.max(presentation_latency_ms) <= 120.0


def test_pts_pacer_does_not_systematically_drop_batched_rtp_frames() -> None:
    frame_interval_ns = 33_333_333
    render_interval_ns = 10_000_000
    arrival_gaps_ns = (100_000_000, 0, 0, 0, 66_666_665)
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    image.setflags(write=False)
    arrivals: list[tuple[int, LiveFrame]] = []
    arrival_ns = 0
    for index in range(180):
        if index:
            arrival_ns += arrival_gaps_ns[(index - 1) % len(arrival_gaps_ns)]
        arrivals.append(
            (
                arrival_ns,
                LiveFrame(
                    session_id="session",
                    connection_session_id="connection",
                    frame_index=index,
                    received_at_client_monotonic_ns=arrival_ns,
                    converted_at_client_monotonic_ns=arrival_ns,
                    image_rgb=image,
                    video_pts_ns=index * frame_interval_ns,
                ),
            ),
        )

    pacer = LiveFramePacer()
    presented_indices: list[int] = []
    arrival_index = 0
    previous_index = -1
    now_ns = 0
    deadline_ns = arrivals[-1][0] + 500_000_000
    while now_ns <= deadline_ns:
        while arrival_index < len(arrivals) and arrivals[arrival_index][0] <= now_ns:
            pacer.enqueue(arrivals[arrival_index][1])
            arrival_index += 1
        frame = pacer.next_frame(now_ns)
        if frame is not None and frame.frame_index != previous_index:
            presented_indices.append(frame.frame_index)
            previous_index = frame.frame_index
        now_ns += render_interval_ns

    status = pacer.status(now_ns)
    assert len(presented_indices) >= 178
    assert status.frames_dropped <= 2
    assert status.starvations <= 1
