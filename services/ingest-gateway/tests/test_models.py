import pytest
from pydantic import ValidationError

from egoglass_ingest_gateway.models import RtspSourceConfig


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
