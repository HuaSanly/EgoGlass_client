import pytest
from pydantic import ValidationError

from egoglass_ingest_gateway.models import RtspSourceConfig
from egoglass_ingest_gateway.webrtc_models import (
    StreamControlAction,
    StreamControlCommand,
    StreamControlState,
    StreamControlStatus,
)


def test_rokid_rtsp_url_uses_device_as_default_channel() -> None:
    config = RtspSourceConfig(
        host="ar-security-media.rokid.com",
        port=5540,
        device_id="34020000001550000668",
    )

    assert config.stream_path == "/rtp/34020000001550000668_34020000001550000668"
    assert (
        config.rtsp_url
        == "rtsp://ar-security-media.rokid.com:5540/rtp/"
        "34020000001550000668_34020000001550000668"
    )
    assert config.redacted_url == config.rtsp_url


def test_credentials_are_encoded_and_not_serialized_as_plaintext() -> None:
    config = RtspSourceConfig(
        host="media.example.test",
        device_id="34020000001550000668",
        username="operator@example.test",
        password="p@ss word",
    )

    assert config.rtsp_url.startswith(
        "rtsp://operator%40example.test:p%40ss%20word@media.example.test:554/"
    )
    assert "p@ss word" not in config.model_dump_json()
    assert "p@ss word" not in repr(config)
    assert "operator@example.test" not in config.redacted_url


def test_password_without_username_is_rejected() -> None:
    with pytest.raises(ValidationError, match="username is required"):
        RtspSourceConfig(
            host="media.example.test",
            device_id="34020000001550000668",
            password="secret",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("host", "media server"),
        ("device_id", "device/id"),
        ("channel_id", "channel id"),
    ],
)
def test_unsafe_url_components_are_rejected(field: str, value: str) -> None:
    payload = {
        "host": "media.example.test",
        "device_id": "34020000001550000668",
        field: value,
    }

    with pytest.raises(ValidationError):
        RtspSourceConfig.model_validate(payload)


def test_stream_control_contract_accepts_only_versioned_bounded_messages() -> None:
    command = StreamControlCommand(
        command_id="0123456789abcdef0123456789abcdef",
        action=StreamControlAction.START,
    )
    status = StreamControlStatus.model_validate_json(
        """{
          "schema_version":"1.0",
          "message_type":"stream_control_status",
          "command_id":null,
          "state":"ready",
          "detail":null
        }"""
    )

    assert command.action == "start"
    assert status.state is StreamControlState.READY

    command_payload = command.model_dump(mode="json")
    invalid_messages = (
        {**command_payload, "command_id": "ABCDEF" * 5 + "AB"},
        {**command_payload, "action": "restart"},
        {**command_payload, "unexpected": True},
        {**command_payload, "schema_version": "2.0"},
    )
    for message in invalid_messages:
        with pytest.raises(ValidationError):
            StreamControlCommand.model_validate(message)

    with pytest.raises(ValidationError):
        StreamControlStatus(state=StreamControlState.ERROR, detail="x" * 257)
