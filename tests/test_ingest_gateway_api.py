import sys
from pathlib import Path

from starlette.testclient import TestClient

import ingest_gateway.app as app_module
from ingest_gateway.app import create_app
from ingest_gateway.recording import (
    RecordingClipNotFoundError,
    RecordingSessionNotFoundError,
)
from ingest_gateway.recording_models import (
    RecordingClip,
    RecordingLibrary,
    RecordingSession,
    RecordingState,
    RecordingStatus,
)
from ingest_gateway.webrtc_models import (
    ImuChannelState,
    ImuSensorStatus,
    ImuSensorType,
    ImuTelemetryStatus,
    StreamControlCommand,
    StreamControlState,
    StreamControlStatus,
    WebRtcOffer,
    WebRtcViewerAnswer,
    WebRtcViewerOffer,
)
from ingest_gateway.webrtc_runtime import (
    StreamControlCommandError,
    StreamControlCommandTimeoutError,
    StreamControlUnavailableError,
    WebRtcSessionRuntime,
)

PAIRING_TOKEN = "api-pairing-token-123456"


class ViewerRuntime:
    async def accept_viewer_offer(self, offer: WebRtcViewerOffer) -> WebRtcViewerAnswer:
        assert "viewer-offer" in offer.sdp
        return WebRtcViewerAnswer(sdp="v=0\r\nviewer-answer-description")

    async def close(self) -> None:
        return None


class ControlRuntime:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.commands: list[StreamControlCommand] = []

    async def control_status(self) -> StreamControlStatus:
        return StreamControlStatus(state=StreamControlState.READY)

    async def recording_source(self) -> None:
        return None

    async def send_control_command(
        self,
        command: StreamControlCommand,
    ) -> StreamControlStatus:
        if self.error is not None:
            raise self.error
        self.commands.append(command)
        return StreamControlStatus(
            command_id=command.command_id,
            state=(
                StreamControlState.STARTING
                if command.action == "start"
                else StreamControlState.STOPPED
            ),
        )

    async def close(self) -> None:
        return None


class ImuRuntime:
    async def imu_status(self) -> ImuTelemetryStatus:
        return ImuTelemetryStatus(
            session_id="session-imu-test",
            device_session_id="device-session-imu-test",
            channel_state=ImuChannelState.RECEIVING,
            messages_received=3,
            capabilities_received=1,
            samples_received=2,
            sensors={
                ImuSensorType.ACCELEROMETER: ImuSensorStatus(sample_count=2),
                ImuSensorType.GYROSCOPE: ImuSensorStatus(),
            },
        )

    async def close(self) -> None:
        return None


class RecordingApiRuntime:
    def __init__(self, media_path: Path) -> None:
        self.media_file = media_path
        self.session_id = "b" * 32
        self.clip_id = "c" * 32
        self.state = RecordingState.READY
        self.deleted = False
        self.display_name: str | None = None
        self.session_commands: list[str] = []

    async def status(self) -> RecordingStatus:
        return RecordingStatus(state=self.state, session_id=self.session_id)

    async def start(self) -> RecordingStatus:
        return RecordingStatus(
            state=RecordingState.COUNTDOWN,
            session_id=self.session_id,
            clip_id=self.clip_id,
            countdown_started_at_unix_ms=1000,
            recording_starts_at_unix_ms=4000,
        )

    async def stop(self) -> RecordingStatus:
        return RecordingStatus(state=RecordingState.READY, session_id=self.session_id)

    async def session_command(self, action: str) -> RecordingStatus:
        self.session_commands.append(action)
        return RecordingStatus(state=RecordingState.READY)

    async def library(self) -> RecordingLibrary:
        if self.deleted:
            return RecordingLibrary(sessions=[])
        return RecordingLibrary(
            sessions=[
                RecordingSession(
                    session_id=self.session_id,
                    started_at_unix_ms=4000,
                    display_name=self.display_name,
                    clips=[
                        RecordingClip(
                            clip_id=self.clip_id,
                            recorded_at_unix_ms=4000,
                            ended_at_unix_ms=5000,
                            duration_ms=1000,
                            file_size_bytes=self.media_file.stat().st_size,
                            media_url=(
                                f"/api/v1/recordings/media/{self.session_id}/{self.clip_id}"
                            ),
                        )
                    ],
                )
            ]
        )

    async def media_path(self, session_id: str, clip_id: str) -> Path | None:
        if not self.deleted and (session_id, clip_id) == (
            self.session_id,
            self.clip_id,
        ):
            return self.media_file
        return None

    async def delete_clip(
        self,
        session_id: str,
        clip_id: str,
    ) -> RecordingLibrary:
        if self.deleted or (session_id, clip_id) != (self.session_id, self.clip_id):
            raise RecordingClipNotFoundError("recording clip not found")
        self.deleted = True
        return RecordingLibrary(sessions=[])

    async def rename_session(
        self,
        session_id: str,
        display_name: str,
    ) -> RecordingLibrary:
        if self.deleted or session_id != self.session_id:
            raise RecordingSessionNotFoundError("recording session not found")
        self.display_name = display_name
        return await self.library()

    async def delete_session(self, session_id: str) -> RecordingLibrary:
        if self.deleted or session_id != self.session_id:
            raise RecordingSessionNotFoundError("recording session not found")
        self.deleted = True
        return RecordingLibrary(sessions=[])

    async def close(self) -> None:
        return None


def test_health_and_removed_fallback_routes() -> None:
    with TestClient(create_app()) as client:
        health = client.get("/api/v1/health")
        status = client.get("/api/v1/status")
        probe = client.post(
            "/api/v1/rtsp/probe",
            json={"host": "obsolete.example.test", "device_id": "obsolete"},
        )

    assert health.status_code == 200
    assert health.json()["service"] == "ingest-gateway"
    assert status.status_code == 404
    assert probe.status_code == 404


def test_webrtc_offer_requires_pairing_token_and_returns_safe_answer() -> None:
    peers: list[object] = []

    class Peer:
        def __init__(self, _callbacks: object) -> None:
            self.closed = False
            peers.append(self)

        @property
        def negotiated_video_codec(self) -> str:
            return "H264"

        async def accept_offer(self, offer: WebRtcOffer) -> str:
            assert "offer" in offer.sdp
            return "v=0\r\nanswer-session-description"

        async def close(self) -> None:
            self.closed = True

    webrtc_runtime = WebRtcSessionRuntime(PAIRING_TOKEN, Peer)
    payload = {
        "schema_version": "1.0",
        "device_session_id": "device-session-0001",
        "type": "offer",
        "sdp": "v=0\r\noffer-session-description",
    }

    with TestClient(create_app(webrtc_runtime=webrtc_runtime)) as client:
        unauthorized = client.post("/api/v1/webrtc/sessions", json=payload)
        accepted = client.post(
            "/api/v1/webrtc/sessions",
            json=payload,
            headers={"Authorization": f"Bearer {PAIRING_TOKEN}"},
        )
        replacement_payload = {**payload, "device_session_id": "device-session-0002"}
        replacement = client.post(
            "/api/v1/webrtc/sessions",
            json=replacement_payload,
            headers={"Authorization": f"Bearer {PAIRING_TOKEN}"},
        )
        status = client.get("/api/v1/webrtc/status")

    assert unauthorized.status_code == 401
    assert PAIRING_TOKEN not in unauthorized.text
    assert accepted.status_code == 200
    assert replacement.status_code == 200
    assert replacement.json()["session_id"] != accepted.json()["session_id"]
    assert peers[0].closed
    assert status.json()["video_codec"] == "H264"
    assert accepted.json()["type"] == "answer"
    assert status.json()["phase"] == "negotiating"
    assert status.json()["device_session_id"] == "device-session-0002"


def test_viewer_offer_is_loopback_only_and_allows_desktop_origin() -> None:
    payload = {
        "schema_version": "1.0",
        "type": "offer",
        "sdp": "v=0\r\nviewer-offer-description",
    }
    app = create_app(
        webrtc_runtime=ViewerRuntime(),  # type: ignore[arg-type]
        viewer_allowed_hosts=frozenset({"testclient"}),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/webrtc/viewer/sessions",
            json=payload,
            headers={"Origin": "http://127.0.0.1:8765"},
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:8765"
    assert response.json()["type"] == "answer"

    with TestClient(
        create_app(webrtc_runtime=ViewerRuntime())  # type: ignore[arg-type]
    ) as client:
        forbidden = client.post("/api/v1/webrtc/viewer/sessions", json=payload)
    assert forbidden.status_code == 403


def test_viewer_offer_reports_unavailable_until_glass3_track_arrives() -> None:
    runtime = WebRtcSessionRuntime(PAIRING_TOKEN)
    payload = {
        "schema_version": "1.0",
        "type": "offer",
        "sdp": "v=0\r\nviewer-offer-description",
    }
    with TestClient(
        create_app(webrtc_runtime=runtime, viewer_allowed_hosts=frozenset({"testclient"}))
    ) as client:
        response = client.post("/api/v1/webrtc/viewer/sessions", json=payload)

    assert response.status_code == 503
    assert response.json()["detail"] == "Glass3 video is not ready"


def test_stream_control_api_is_loopback_only_and_supports_get_post_cors() -> None:
    runtime = ControlRuntime()
    app = create_app(
        webrtc_runtime=runtime,  # type: ignore[arg-type]
        viewer_allowed_hosts=frozenset({"testclient"}),
    )
    payload = {"action": "start"}
    origin = "http://127.0.0.1:8765"
    with TestClient(app) as client:
        status = client.get(
            "/api/v1/webrtc/control",
            headers={"Origin": origin},
        )
        command = client.post(
            "/api/v1/webrtc/control/commands",
            json=payload,
            headers={"Origin": origin},
        )
        preflight = client.options(
            "/api/v1/webrtc/control",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        )

    assert status.status_code == 200
    assert status.json()["state"] == "ready"
    assert status.headers["access-control-allow-origin"] == origin
    assert command.status_code == 200
    command_id = command.json()["command_id"]
    assert len(command_id) == 32
    assert all(character in "0123456789abcdef" for character in command_id)
    assert command.json()["state"] == "starting"
    assert runtime.commands[0].command_id == command_id
    assert runtime.commands[0].action == "start"
    assert preflight.status_code == 200
    assert "GET" in preflight.headers["access-control-allow-methods"]

    with TestClient(
        create_app(webrtc_runtime=ControlRuntime())  # type: ignore[arg-type]
    ) as client:
        forbidden_get = client.get("/api/v1/webrtc/control")
        forbidden_post = client.post(
            "/api/v1/webrtc/control/commands",
            json=payload,
        )
    assert forbidden_get.status_code == 403
    assert forbidden_post.status_code == 403


def test_imu_status_api_is_loopback_only_and_exposes_no_sample_history() -> None:
    origin = "http://127.0.0.1:8765"
    app = create_app(
        webrtc_runtime=ImuRuntime(),  # type: ignore[arg-type]
        viewer_allowed_hosts=frozenset({"testclient"}),
    )
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/webrtc/imu/status",
            headers={"Origin": origin},
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    payload = response.json()
    assert payload["channel_state"] == "receiving"
    assert payload["samples_received"] == 2
    assert payload["sensors"]["accelerometer"]["sample_count"] == 2
    assert "history" not in response.text

    with TestClient(
        create_app(webrtc_runtime=ImuRuntime())  # type: ignore[arg-type]
    ) as client:
        forbidden = client.get("/api/v1/webrtc/imu/status")
    assert forbidden.status_code == 403


def test_stream_control_api_maps_safe_runtime_failures() -> None:
    command = {"action": "stop"}
    cases = (
        (StreamControlUnavailableError("channel unavailable"), 503),
        (StreamControlCommandTimeoutError("ack timed out"), 504),
        (StreamControlCommandError("send failed"), 502),
    )
    for error, expected_status in cases:
        app = create_app(
            webrtc_runtime=ControlRuntime(error),  # type: ignore[arg-type]
            viewer_allowed_hosts=frozenset({"testclient"}),
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/webrtc/control/commands",
                json=command,
            )
        assert response.status_code == expected_status
        assert response.json()["detail"] == str(error)


def test_stream_stop_is_rejected_until_active_recording_stops(tmp_path: Path) -> None:
    media = tmp_path / "completed.mp4"
    media.write_bytes(b"mp4-data")
    control_runtime = ControlRuntime()
    recording_runtime = RecordingApiRuntime(media)
    recording_runtime.state = RecordingState.RECORDING
    app = create_app(
        webrtc_runtime=control_runtime,  # type: ignore[arg-type]
        recording_runtime=recording_runtime,  # type: ignore[arg-type]
        viewer_allowed_hosts=frozenset({"testclient"}),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/webrtc/control/commands",
            json={"action": "stop"},
        )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "stop the active recording before stopping the video stream"
    )
    assert control_runtime.commands == []


def test_recording_api_is_loopback_only_and_serves_only_registered_media(
    tmp_path: Path,
) -> None:
    media = tmp_path / "completed.mp4"
    media.write_bytes(b"mp4-data")
    recording_runtime = RecordingApiRuntime(media)
    app = create_app(
        recording_runtime=recording_runtime,  # type: ignore[arg-type]
        viewer_allowed_hosts=frozenset({"testclient"}),
    )
    origin = "http://127.0.0.1:8765"
    with TestClient(app) as client:
        status = client.get("/api/v1/recordings/status", headers={"Origin": origin})
        start = client.post(
            "/api/v1/recordings/commands",
            json={"action": "start"},
            headers={"Origin": origin},
        )
        library = client.get("/api/v1/recordings/library")
        media_response = client.get(
            f"/api/v1/recordings/media/{recording_runtime.session_id}/"
            f"{recording_runtime.clip_id}"
        )
        missing = client.get(
            f"/api/v1/recordings/media/{recording_runtime.session_id}/{'d' * 32}"
        )
        delete_path = (
            f"/api/v1/recordings/clips/{recording_runtime.session_id}/"
            f"{recording_runtime.clip_id}"
        )
        rename_path = (
            f"/api/v1/recordings/sessions/{recording_runtime.session_id}"
        )
        rename_preflight = client.options(
            rename_path,
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "PATCH",
            },
        )
        renamed = client.patch(
            rename_path,
            json={"display_name": "厨房采集"},
            headers={"Origin": origin},
        )
        invalid_rename = client.patch(
            rename_path,
            json={"display_name": "   "},
        )
        missing_rename = client.patch(
            f"/api/v1/recordings/sessions/{'d' * 32}",
            json={"display_name": "不存在"},
        )
        delete_preflight = client.options(
            delete_path,
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "DELETE",
            },
        )
        deleted = client.delete(delete_path, headers={"Origin": origin})
        missing_delete = client.delete(delete_path)

    assert status.status_code == 200
    assert status.headers["access-control-allow-origin"] == origin
    assert status.json()["detail"] == ""
    assert status.json()["output"] == {
        "width": 1280,
        "height": 720,
        "fps": 30,
        "container": "mp4",
        "video_codec": "h264",
    }
    assert start.json()["state"] == "countdown"
    assert start.json()["recording_starts_at_unix_ms"] == 4000
    assert library.json()["sessions"][0]["started_at_unix_ms"] == 4000
    assert library.json()["sessions"][0]["display_name"] is None
    assert library.json()["sessions"][0]["clips"][0]["recorded_at_unix_ms"] == 4000
    assert library.json()["sessions"][0]["clips"][0]["file_size_bytes"] == 8
    assert media_response.status_code == 200
    assert media_response.headers["content-type"] == "video/mp4"
    assert media_response.content == b"mp4-data"
    assert missing.status_code == 404
    assert rename_preflight.status_code == 200
    assert "PATCH" in rename_preflight.headers["access-control-allow-methods"]
    assert renamed.status_code == 200
    assert renamed.headers["access-control-allow-origin"] == origin
    assert renamed.json()["sessions"][0]["display_name"] == "厨房采集"
    assert invalid_rename.status_code == 422
    assert missing_rename.status_code == 404
    assert delete_preflight.status_code == 200
    assert "DELETE" in delete_preflight.headers["access-control-allow-methods"]
    assert deleted.status_code == 200
    assert deleted.headers["access-control-allow-origin"] == origin
    assert deleted.json() == {"schema_version": "1.0", "sessions": []}
    assert missing_delete.status_code == 404

    with TestClient(
        create_app(recording_runtime=recording_runtime)  # type: ignore[arg-type]
    ) as client:
        assert client.get("/api/v1/recordings/status").status_code == 403
        assert (
            client.patch(
                f"/api/v1/recordings/sessions/{recording_runtime.session_id}",
                json={"display_name": "未授权"},
            ).status_code
            == 403
        )
        assert (
            client.delete(
                f"/api/v1/recordings/clips/{recording_runtime.session_id}/"
                f"{recording_runtime.clip_id}"
            ).status_code
            == 403
        )


def test_session_finalize_and_delete_apis_are_strict_and_loopback_only(
    tmp_path: Path,
) -> None:
    media = tmp_path / "completed.mp4"
    media.write_bytes(b"mp4-data")
    runtime = RecordingApiRuntime(media)
    app = create_app(
        recording_runtime=runtime,  # type: ignore[arg-type]
        viewer_allowed_hosts=frozenset({"testclient"}),
    )
    with TestClient(app) as client:
        finalized = client.post(
            "/api/v1/recordings/session-commands",
            json={"action": "finalize"},
        )
        invalid = client.post(
            "/api/v1/recordings/session-commands",
            json={"action": "finalize", "force": True},
        )
        deleted = client.delete(f"/api/v1/recordings/sessions/{runtime.session_id}")
        missing = client.delete(f"/api/v1/recordings/sessions/{runtime.session_id}")

    assert finalized.status_code == 200
    assert finalized.json()["session_id"] is None
    assert runtime.session_commands == ["finalize"]
    assert invalid.status_code == 422
    assert deleted.status_code == 200
    assert deleted.json()["sessions"] == []
    assert missing.status_code == 404

    with TestClient(
        create_app(recording_runtime=RecordingApiRuntime(media))  # type: ignore[arg-type]
    ) as client:
        assert (
            client.post(
                "/api/v1/recordings/session-commands",
                json={"action": "new"},
            ).status_code
            == 403
        )


def test_cli_disables_high_frequency_preview_access_logs(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    def run(_application: object, **kwargs: object) -> None:
        captured["app"] = _application
        captured.update(kwargs)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "egoglass-ingest-gateway",
            "--pairing-token",
            PAIRING_TOKEN,
            "--hide-pairing-token",
        ],
    )
    monkeypatch.setattr(app_module.uvicorn, "run", run)

    app_module.main()

    assert captured["access_log"] is False
    assert PAIRING_TOKEN not in capsys.readouterr().out
    assert captured["app"].state.discovery_service is not None
