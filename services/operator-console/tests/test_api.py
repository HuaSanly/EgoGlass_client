from starlette.testclient import TestClient

from egoglass_operator_console.app import create_app
from egoglass_operator_console.runtime import ConsoleRuntime


def make_client() -> TestClient:
    return TestClient(create_app(ConsoleRuntime()))


def test_health_and_static_console_are_served() -> None:
    with make_client() as client:
        health = client.get("/api/v1/health")
        page = client.get("/")

    assert health.json() == {
        "status": "ok",
        "service": "operator-console",
        "version": "0.1.0",
    }
    assert page.status_code == 200
    assert 'id="scene-canvas"' in page.text
    assert "EgoGlass Operator Console" in page.text


def test_settings_round_trip_and_revision() -> None:
    with make_client() as client:
        initial = client.get("/api/v1/state").json()
        settings = initial["settings"]
        settings["capture_fps"] = 15
        settings["inference_fps"] = 5
        updated = client.put("/api/v1/settings", json=settings)

    assert updated.status_code == 200
    payload = updated.json()
    assert payload["settings_revision"] == 2
    assert payload["settings"]["capture_fps"] == 15
    assert payload["settings"]["inference_fps"] == 5


def test_invalid_rate_returns_validation_error() -> None:
    with make_client() as client:
        settings = client.get("/api/v1/state").json()["settings"]
        settings["capture_fps"] = 10
        settings["inference_fps"] = 20
        response = client.put("/api/v1/settings", json=settings)

    assert response.status_code == 422


def test_recording_and_session_conflict_are_explicit() -> None:
    with make_client() as client:
        started = client.post("/api/v1/recording/start")
        stopped_session = client.post("/api/v1/session/stop")
        conflict = client.post("/api/v1/recording/start")

    assert started.json()["recording"] is True
    assert stopped_session.json()["recording"] is False
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "recording requires a live session"


def test_websocket_emits_versioned_trajectory_telemetry() -> None:
    with make_client() as client, client.websocket_connect("/api/v1/telemetry") as websocket:
        payload = websocket.receive_json()

    assert payload["schema_version"] == "1.0"
    assert payload["source"] == "synthetic"
    assert payload["calibration"]["state"] == "simulated"
    assert [hand["side"] for hand in payload["hands"]] == ["left", "right"]
    assert len(payload["hands"][0]["waypoints"]) == 10
