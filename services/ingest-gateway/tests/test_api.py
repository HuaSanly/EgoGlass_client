from starlette.testclient import TestClient

from egoglass_ingest_gateway.adapters.rtsp import RtspProbeError
from egoglass_ingest_gateway.app import create_app
from egoglass_ingest_gateway.models import ProbeResult, RtspSourceConfig, RtspTransport
from egoglass_ingest_gateway.runtime import IngestRuntime
from egoglass_ingest_gateway.webrtc_models import WebRtcOffer
from egoglass_ingest_gateway.webrtc_runtime import WebRtcSessionRuntime

PAIRING_TOKEN = "api-pairing-token-123456"


class PreviewRuntime:
    async def latest_preview_jpeg(self) -> bytes:
        return b"\xff\xd8preview\xff\xd9"

    async def close(self) -> None:
        return None


def result_fixture() -> ProbeResult:
    return ProbeResult(
        redacted_url=(
            "rtsp://media.example.test:554/rtp/"
            "34020000001550000668_34020000001550000668"
        ),
        transport=RtspTransport.TCP,
        codec="h264",
        width=1280,
        height=720,
        pixel_format="yuv420p",
        average_fps=20,
        first_frame_pts=0,
        first_frame_time_base_num=1,
        first_frame_time_base_den=90_000,
        opened_at_unix_ns=1,
        first_frame_received_at_perf_counter_ns=1,
        probe_latency_ms=50,
    )


def test_health_status_and_probe_contract() -> None:
    class Decoder:
        def probe(self, config: RtspSourceConfig) -> ProbeResult:
            assert config.transport is RtspTransport.TCP
            return result_fixture()

    with TestClient(create_app(IngestRuntime(Decoder()))) as client:
        health = client.get("/api/v1/health")
        initial = client.get("/api/v1/status")
        probe = client.post(
            "/api/v1/rtsp/probe",
            json={
                "host": "media.example.test",
                "device_id": "34020000001550000668",
            },
        )
        final = client.get("/api/v1/status")

    assert health.json()["service"] == "ingest-gateway"
    assert initial.json()["phase"] == "idle"
    assert probe.status_code == 200
    assert probe.json()["codec"] == "h264"
    assert final.json()["phase"] == "ready"


def test_probe_failure_returns_safe_gateway_error() -> None:
    class Decoder:
        def probe(self, config: RtspSourceConfig) -> ProbeResult:
            raise RtspProbeError(f"RTSP probe failed for {config.redacted_url}: TimeoutError")

    with TestClient(create_app(IngestRuntime(Decoder()))) as client:
        response = client.post(
            "/api/v1/rtsp/probe",
            json={
                "host": "media.example.test",
                "device_id": "34020000001550000668",
                "username": "operator",
                "password": "top-secret",
            },
        )

    assert response.status_code == 502
    assert "top-secret" not in response.text
    assert "operator@" not in response.text
    assert "TimeoutError" in response.json()["detail"]


def test_validation_error_does_not_echo_password() -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v1/rtsp/probe",
            json={
                "host": "media.example.test",
                "device_id": "34020000001550000668",
                "password": "top-secret",
            },
        )

    assert response.status_code == 422
    assert "top-secret" not in response.text


def test_webrtc_offer_requires_pairing_token_and_returns_safe_answer() -> None:
    class Peer:
        def __init__(self, _callbacks: object) -> None:
            pass

        @property
        def negotiated_video_codec(self) -> str:
            return "H264"

        async def accept_offer(self, offer: WebRtcOffer) -> str:
            assert "offer" in offer.sdp
            return "v=0\r\nanswer-session-description"

        async def close(self) -> None:
            return None

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
        status = client.get("/api/v1/webrtc/status")

    assert unauthorized.status_code == 401
    assert PAIRING_TOKEN not in unauthorized.text
    assert accepted.status_code == 200
    assert status.json()["video_codec"] == "H264"
    assert accepted.json()["type"] == "answer"
    assert status.json()["phase"] == "negotiating"


def test_preview_frame_is_jpeg_no_store_and_loopback_only() -> None:
    app = create_app(
        webrtc_runtime=PreviewRuntime(),  # type: ignore[arg-type]
        preview_allowed_hosts=frozenset({"testclient"}),
    )
    with TestClient(app) as client:
        response = client.get("/api/v1/webrtc/frame.jpg")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert response.content == b"\xff\xd8preview\xff\xd9"

    with TestClient(
        create_app(webrtc_runtime=PreviewRuntime())  # type: ignore[arg-type]
    ) as client:
        forbidden = client.get("/api/v1/webrtc/frame.jpg")
    assert forbidden.status_code == 403
