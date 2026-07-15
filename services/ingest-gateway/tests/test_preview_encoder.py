from io import BytesIO

import av

from egoglass_ingest_gateway.adapters.preview_encoder import MjpegPreviewEncoder


def test_preview_encoder_produces_decodable_jpeg_frames() -> None:
    encoder = MjpegPreviewEncoder()
    source = av.VideoFrame(320, 240, "yuv420p")

    payload = encoder.encode(source)

    assert payload.startswith(b"\xff\xd8")
    assert payload.endswith(b"\xff\xd9")
    with av.open(BytesIO(payload)) as container:
        decoded = next(container.decode(video=0))
    assert (decoded.width, decoded.height) == (320, 240)
