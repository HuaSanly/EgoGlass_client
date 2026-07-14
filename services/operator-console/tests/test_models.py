import pytest
from pydantic import ValidationError

from egoglass_operator_console.models import RuntimeSettings


def test_default_settings_match_first_client_profile() -> None:
    settings = RuntimeSettings()

    assert (settings.video_width, settings.video_height) == (1280, 720)
    assert settings.capture_fps == 20
    assert settings.inference_fps == 10
    assert settings.history_frames == 8
    assert settings.prediction_steps == 10
    assert settings.max_feedback_age_ms == 500


def test_inference_rate_cannot_exceed_capture_rate() -> None:
    with pytest.raises(ValidationError, match="inference_fps must not exceed capture_fps"):
        RuntimeSettings(capture_fps=10, inference_fps=11)


def test_unknown_settings_are_rejected() -> None:
    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate({"capture_fps": 20, "unsupported": True})
