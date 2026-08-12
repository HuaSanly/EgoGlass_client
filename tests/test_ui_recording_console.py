from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
from PyQt6.QtCore import QPoint, QRect
from PyQt6.QtWidgets import QApplication

from schemas.recording import (
    FrameMetadataMatchStatus,
    ImuSensorType,
    RecordingFrameRow,
    RecordingImuRow,
    RecordingLibrary,
    RecordingOutput,
)
from ui.gateway.capture_recording import CaptureRecordingReader, CaptureRecordingWriter
from ui.gateway.webrtc_models import StreamControlAction
from ui.views.home import HomeView
from ui.views.recording_library import (
    RecordingCard,
    RecordingLibraryView,
    _decode_thumbnail,
)
from ui.widgets.imu_monitor import ImuChartSample, ImuMonitorStats, ImuMonitorWidget
from ui.widgets.recording_playback import RecordingReplaySource
from ui.widgets.video_canvas import VideoCanvas


class _Runtime:
    def __init__(self, recording_directory: Path | None = None) -> None:
        self._directory = recording_directory
        self._snapshot = SimpleNamespace(
            revision=0,
            webrtc=None,
            stream_control=None,
            recording=None,
            imu=None,
            imu_telemetry=None,
            display=None,
            library=RecordingLibrary(recordings=[]),
        )
        self.stream_actions: list[str] = []
        self.recording_actions: list[str] = []

    def snapshot(self) -> object:
        return self._snapshot

    def latest_frame(self) -> None:
        return None

    def command_results(self) -> tuple[object, ...]:
        return ()

    def request_stream(self, action: str) -> None:
        self.stream_actions.append(action)

    def request_recording(self, action: str) -> None:
        self.recording_actions.append(action)

    def request_library_refresh(self) -> None:
        return None

    def recording_directory(self, _recording_id: str) -> Path | None:
        return self._directory


def test_recording_console_has_no_algorithm_or_session_controls() -> None:
    root = Path(__file__).parents[1]
    home = (root / "ui" / "views" / "home.py").read_text(encoding="utf-8")
    canvas = (root / "ui" / "widgets" / "video_canvas.py").read_text(
        encoding="utf-8"
    )
    combined = home + canvas

    for forbidden in (
        "hand_tracking",
        "HaMeR",
        "ViTPose",
        "SpatialSyncCanvas",
        "request_live_inference",
        "request_session",
        "set_overlay",
    ):
        assert forbidden not in combined
    assert "开始视频" in home
    assert "开始录制" in home
    assert "ImuMonitorWidget" in home
    assert "QLabel" not in home


def test_raw_imu_monitor_is_bounded_and_deduplicated(
    qt_application: QApplication,
) -> None:
    monitor = ImuMonitorWidget(maximum_samples=3)
    samples = tuple(
        ImuChartSample(
            sensor_type="accelerometer",
            sequence_number=index,
            recording_time_ns=index * 10_000_000,
            received_at_client_monotonic_ns=index * 10_000_000,
            values=(float(index), float(index + 1), float(index + 2)),
        )
        for index in range(5)
    )

    assert monitor.append_samples(samples) == 5
    assert monitor.append_samples(samples[-1:]) == 0
    assert monitor.sample_count("accelerometer") == 3
    monitor.set_stats(
        ImuMonitorStats(
            sample_rate_hz=100.0,
            latest_latency_ms=2.5,
            sequence_gaps=1,
            duplicate_samples=2,
            out_of_order_samples=3,
            queue_overflows=4,
            recording_window_ns=1_234_000_000,
            csv_state="writing",
        )
    )
    qt_application.processEvents()

    assert monitor.rate_badge.text() == "100.0 Hz"
    assert "丢序 1" in monitor.sequence_label.text()
    assert monitor.accelerometer_plot.listDataItems()[0].xData.tolist() == [0.0, 0.01, 0.02]


def test_home_recording_controls_dispatch_only_stream_and_recording(
    qt_application: QApplication,
) -> None:
    runtime = _Runtime()
    view = HomeView(runtime)  # type: ignore[arg-type]
    try:
        view.stream_button.click()
        view.recording_button.click()
        assert runtime.stream_actions == [StreamControlAction.START]
        assert runtime.recording_actions == ["start"]
        assert view.findChild(VideoCanvas) is view.canvas
        assert view.findChild(ImuMonitorWidget) is view.imu_monitor
        assert view.findChildren(RecordingCard) == []
    finally:
        view.close_resources()
        view.close()
        qt_application.processEvents()


def test_recording_console_layout_stays_separate_at_supported_sizes(
    qt_application: QApplication,
) -> None:
    view = HomeView(_Runtime())  # type: ignore[arg-type]
    try:
        for width, height in ((1280, 800), (1440, 900), (1920, 1080)):
            view.resize(width, height)
            view.show()
            qt_application.processEvents()
            canvas_rect = QRect(view.canvas.mapTo(view, QPoint(0, 0)), view.canvas.size())
            sidebar_rect = QRect(
                view.sidebar.mapTo(view, QPoint(0, 0)),
                view.sidebar.size(),
            )
            frame_strip_rect = QRect(
                view.frame_link_strip.mapTo(view, QPoint(0, 0)),
                view.frame_link_strip.size(),
            )
            assert not canvas_rect.intersects(sidebar_rect)
            assert not frame_strip_rect.intersects(sidebar_rect)
            assert frame_strip_rect.left() == canvas_rect.left()
            assert frame_strip_rect.right() == canvas_rect.right()
            assert not view.grab().isNull()
    finally:
        view.close_resources()
        view.close()
        qt_application.processEvents()


def test_recording_replay_uses_frame_time_for_imu_cursor(tmp_path: Path) -> None:
    directory = _recording_fixture(tmp_path)
    source = RecordingReplaySource(directory)
    try:
        first = source.seek_frame(0)
        second = source.next_frame()
        assert second is not None
        assert first.recording_time_ns == 100_000_000
        assert second.recording_time_ns == 200_000_000
        first_cursor = source.imu_cursor(first.recording_time_ns)
        second_cursor = source.imu_cursor(second.recording_time_ns)
        assert first_cursor.accelerometer == (0.0, 1.0, 2.0)
        assert first_cursor.gyroscope is None
        assert second_cursor.gyroscope == (3.0, 4.0, 5.0)
    finally:
        source.close()


def test_recording_library_flows_cards_and_playback_without_overlap(
    qt_application: QApplication,
    tmp_path: Path,
) -> None:
    directory = _recording_fixture(tmp_path)
    summary = CaptureRecordingReader.open(directory).summary()
    runtime = _Runtime(directory)
    runtime._snapshot.library = RecordingLibrary(recordings=[summary])
    view = RecordingLibraryView(runtime)  # type: ignore[arg-type]
    try:
        view.set_library(runtime._snapshot.library)
        assert tuple(view.cards) == (summary.recording_id,)
        card = view.cards[summary.recording_id]
        for width, height in ((1280, 800), (1440, 900), (1920, 1080)):
            view.resize(width, height)
            view.show()
            qt_application.processEvents()
            card_rect = QRect(card.mapTo(view, QPoint(0, 0)), card.size())
            assert view.rect().contains(card_rect)
        view.open_recording(summary.recording_id)
        qt_application.processEvents()
        assert view.stack.currentWidget() is view.playback
        assert view.playback.canvas.status().latest_frame_index == 0
    finally:
        view.close_resources()
        view.close()
        qt_application.processEvents()


def test_recording_thumbnail_decodes_without_pillow(tmp_path: Path) -> None:
    directory = _recording_fixture(tmp_path)
    result = _decode_thumbnail("a" * 32, directory / "video.mp4")

    assert result.recording_id == "a" * 32
    assert result.image is not None
    assert (result.image.width(), result.image.height()) == (310, 186)


def _recording_fixture(root: Path) -> Path:
    writer = CaptureRecordingWriter.create(
        root,
        recording_id="a" * 32,
        video_profile=RecordingOutput(width=32, height=24, fps=10.0),
        countdown_started_at_unix_ns=1_000_000_000,
        countdown_started_at_client_monotonic_ns=2_000_000_000,
    )
    with av.open(str(writer.video_path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 32
        stream.height = 24
        stream.pix_fmt = "yuv420p"
        for index in range(2):
            pixels = np.full((24, 32, 3), index * 64, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    with av.open(str(writer.video_path), mode="r") as container:
        indexed_video = tuple(
            (frame.pts, frame.time_base)
            for frame in container.decode(container.streams.video[0])
        )
    assert len(indexed_video) == 2
    for frame_index, (recording_time_ns, indexed) in enumerate(
        zip((100_000_000, 200_000_000), indexed_video, strict=True)
    ):
        pts, time_base = indexed
        assert pts is not None and time_base is not None
        writer.append_frame(
            RecordingFrameRow(
                frame_index=frame_index,
                recording_time_ns=recording_time_ns,
                mp4_pts=pts,
                mp4_time_base_num=time_base.numerator,
                mp4_time_base_den=time_base.denominator,
                connection_session_id="connection-1",
                received_at_client_monotonic_ns=2_000_000_000 + recording_time_ns,
                metadata_match_status=FrameMetadataMatchStatus.UNMATCHED,
            )
        )
    for sample_index, (time_ns, sensor, values) in enumerate(
        (
            (100_000_000, ImuSensorType.ACCELEROMETER, (0.0, 1.0, 2.0)),
            (150_000_000, ImuSensorType.GYROSCOPE, (3.0, 4.0, 5.0)),
        )
    ):
        writer.append_imu(
            RecordingImuRow(
                sample_index=sample_index,
                recording_time_ns=time_ns,
                connection_session_id="connection-1",
                sensor_type=sensor,
                sequence_number=sample_index,
                sensor_event_monotonic_ns=3_000_000_000 + time_ns,
                received_at_elapsed_realtime_ns=3_000_000_000 + time_ns,
                received_at_client_monotonic_ns=2_000_000_000 + time_ns,
                accuracy=3,
                x=values[0],
                y=values[1],
                z=values[2],
                inside_video_span=True,
            )
        )
    return writer.finalize(ended_at_unix_ns=1_200_000_000).directory
