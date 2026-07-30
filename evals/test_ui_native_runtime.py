from pathlib import Path


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
    assert "_update_active_view" in (
        repository / "ui" / "app.py"
    ).read_text(encoding="utf-8")
