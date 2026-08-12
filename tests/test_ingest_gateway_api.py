from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from schemas.recording import (
    RecordingLibrary,
    RecordingOutput,
    RecordingState,
    RecordingStatus,
    RecordingSummary,
)
from ui.gateway.app import create_app
from ui.gateway.webrtc_models import (
    ImuTelemetryStatus,
    StreamControlState,
    StreamControlStatus,
    WebRtcAnswer,
    WebRtcPhase,
    WebRtcStatus,
)


class _WebRtcRuntime:
    def __init__(self) -> None:
        self.control_commands = []
        self.capture_sink = None

    def set_capture_telemetry_sink(self, sink: object) -> None:
        self.capture_sink = sink

    def set_display_frame_sink(self, _sink: object) -> None:
        return None

    def set_display_imu_sink(self, _sink: object) -> None:
        return None

    async def status(self) -> WebRtcStatus:
        return WebRtcStatus(phase=WebRtcPhase.STREAMING)

    async def control_status(self) -> StreamControlStatus:
        return StreamControlStatus(state=StreamControlState.STREAMING)

    async def imu_status(self) -> ImuTelemetryStatus:
        return ImuTelemetryStatus(sensors={})

    async def send_control_command(self, command: object) -> StreamControlStatus:
        self.control_commands.append(command)
        return StreamControlStatus(state=StreamControlState.STOPPED)

    async def accept_offer(self, _offer: object, _token: str) -> WebRtcAnswer:
        return WebRtcAnswer(session_id="a" * 32, sdp="v=0\r\n" + "x" * 16)

    async def close(self) -> None:
        return None


class _RecordingRuntime:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.recording_id = "1" * 32
        directory = root / self.recording_id
        directory.mkdir(parents=True)
        self.video = directory / "video.mp4"
        self.imu = directory / "imu.csv"
        self.frames = directory / "frames.csv"
        self.video.write_bytes(b"video")
        self.imu.write_text("sample_index\n", encoding="utf-8")
        self.frames.write_text("frame_index\n", encoding="utf-8")
        self.state = RecordingState.READY
        self.deleted = False

    async def status(self) -> RecordingStatus:
        return RecordingStatus(state=self.state, output=RecordingOutput())

    async def start(self) -> RecordingStatus:
        self.state = RecordingState.COUNTDOWN
        return await self.status()

    async def stop(self) -> RecordingStatus:
        self.state = RecordingState.READY
        return await self.status()

    async def library(self) -> RecordingLibrary:
        return RecordingLibrary(
            recordings=[] if self.deleted else [self._summary()]
        )

    async def delete(self, recording_id: str) -> RecordingLibrary:
        assert recording_id == self.recording_id
        self.deleted = True
        return RecordingLibrary(recordings=[])

    async def media_path(self, recording_id: str) -> Path | None:
        return self.video if recording_id == self.recording_id and not self.deleted else None

    async def artifact_path(self, recording_id: str, artifact: str) -> Path | None:
        if recording_id != self.recording_id or self.deleted:
            return None
        return {"imu.csv": self.imu, "frames.csv": self.frames}.get(artifact)

    async def close(self) -> None:
        return None

    def _summary(self) -> RecordingSummary:
        return RecordingSummary(
            recording_id=self.recording_id,
            recorded_at_unix_ns=1,
            ended_at_unix_ns=2,
            duration_ns=1,
            width=640,
            height=480,
            fps=30,
            file_size_bytes=5,
            frame_count=1,
            imu_sample_count=0,
            hashes_verified=True,
        )


def _client(tmp_path: Path) -> tuple[TestClient, _RecordingRuntime, _WebRtcRuntime]:
    recording = _RecordingRuntime(tmp_path)
    webrtc = _WebRtcRuntime()
    app = create_app(
        webrtc_runtime=webrtc,  # type: ignore[arg-type]
        recording_runtime=recording,  # type: ignore[arg-type]
        viewer_allowed_hosts=frozenset({"testclient"}),
    )
    return TestClient(app), recording, webrtc


def test_recording_api_exposes_flat_recordings_and_csv(tmp_path: Path) -> None:
    client, recording, _webrtc = _client(tmp_path)
    with client:
        library = client.get("/api/v1/recordings/library")
        assert library.status_code == 200
        assert library.json()["recordings"][0]["recording_id"] == recording.recording_id

        video = client.get(f"/api/v1/recordings/{recording.recording_id}/video.mp4")
        assert video.status_code == 200
        assert video.content == b"video"
        assert client.get(
            f"/api/v1/recordings/{recording.recording_id}/imu.csv"
        ).status_code == 200
        assert client.get(
            f"/api/v1/recordings/{recording.recording_id}/frames.csv"
        ).status_code == 200

        deleted = client.delete(f"/api/v1/recordings/{recording.recording_id}")
        assert deleted.status_code == 200
        assert deleted.json()["recordings"] == []


def test_algorithm_and_session_routes_are_absent(tmp_path: Path) -> None:
    client, recording, _webrtc = _client(tmp_path)
    with client:
        assert client.get("/api/v1/perception/hand-tracking/status").status_code == 404
        assert client.post(
            "/api/v1/recordings/session-commands",
            json={"action": "new"},
        ).status_code in {404, 405}
        assert client.delete(
            f"/api/v1/recordings/clips/{recording.recording_id}/{'2' * 32}"
        ).status_code == 404


def test_stream_cannot_stop_during_recording(tmp_path: Path) -> None:
    client, recording, webrtc = _client(tmp_path)
    recording.state = RecordingState.RECORDING
    with client:
        response = client.post(
            "/api/v1/webrtc/control/commands",
            json={"action": "stop"},
        )
        assert response.status_code == 409
        assert webrtc.control_commands == []
