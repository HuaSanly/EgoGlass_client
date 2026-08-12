from schemas import (
    CaptureRecordingManifest,
    CaptureRecordingQualityReport,
    FrameMetadataMatchStatus,
    RecordingFrameRow,
    RecordingImuRow,
    RecordingLibrary,
    RecordingOutput,
    RecordingState,
    RecordingStatus,
    RecordingSummary,
)


def test_public_schema_package_exports_only_recording_contracts() -> None:
    assert CaptureRecordingManifest.__module__ == "schemas.recording"
    assert CaptureRecordingQualityReport.__module__ == "schemas.recording"
    assert FrameMetadataMatchStatus.__module__ == "schemas.recording"
    assert RecordingFrameRow.__module__ == "schemas.recording"
    assert RecordingImuRow.__module__ == "schemas.recording"
    assert RecordingLibrary.__module__ == "schemas.recording"
    assert RecordingOutput.__module__ == "schemas.recording"
    assert RecordingState.__module__ == "schemas.recording"
    assert RecordingStatus.__module__ == "schemas.recording"
    assert RecordingSummary.__module__ == "schemas.recording"
