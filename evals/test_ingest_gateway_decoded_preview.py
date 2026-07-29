from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from av import VideoFrame

from ingest_gateway.decoded_preview import DecodedPreviewRuntime


def test_thirty_fps_input_stays_latest_first_while_preview_encoding_is_slow() -> None:
    started = threading.Event()
    release = threading.Event()
    encoded_widths: list[int] = []

    def slow_encoder(frame: VideoFrame) -> bytes:
        encoded_widths.append(frame.width)
        if len(encoded_widths) == 1:
            started.set()
            release.wait(timeout=1)
        return b"jpeg"

    async def scenario() -> None:
        runtime = DecodedPreviewRuntime(encoder=slow_encoder)
        await runtime.submit_gateway_frame(
            session_id="eval-session",
            connection_session_id="eval-connection",
            frame_index=0,
            received_at_client_monotonic_ns=0,
            decoded_frame=VideoFrame(64, 36, "bgr24"),
        )
        assert await asyncio.to_thread(started.wait, 1)
        for frame_index in range(1, 30):
            await asyncio.wait_for(
                runtime.submit_gateway_frame(
                    session_id="eval-session",
                    connection_session_id="eval-connection",
                    frame_index=frame_index,
                    received_at_client_monotonic_ns=frame_index * 33_333_333,
                    decoded_frame=VideoFrame(64 + frame_index, 36, "bgr24"),
                ),
                timeout=0.05,
            )
        release.set()
        for _ in range(100):
            status = await runtime.status()
            if status.frames_encoded == 2:
                break
            await asyncio.sleep(0.001)

        assert encoded_widths == [64, 93]
        assert status.frames_received == 30
        assert status.frames_encoded == 2
        assert status.frames_dropped == 28
        assert status.latest_frame_index == 29
        stream = runtime.stream()
        assert b"X-Frame-Index: 29" in await anext(stream)
        await stream.aclose()
        await runtime.close()

    asyncio.run(scenario())


def test_operator_preview_cannot_create_a_second_glass3_webrtc_peer() -> None:
    repository = Path(__file__).parents[1]
    script = (repository / "src/operator_console/static/app.js").read_text(encoding="utf-8")
    gateway_app = (repository / "src/ingest_gateway/app.py").read_text(encoding="utf-8")
    aiortc_adapter = (
        repository / "src/ingest_gateway/adapters/aiortc_peer.py"
    ).read_text(encoding="utf-8")

    assert "RTCPeerConnection" not in script
    assert "/api/v1/webrtc/viewer/sessions" not in script
    assert "/api/v1/webrtc/viewer/sessions" not in gateway_app
    assert "/api/v1/webrtc/decoded-preview.mjpg" in script
    assert aiortc_adapter.count("RTCPeerConnection(") == 1
