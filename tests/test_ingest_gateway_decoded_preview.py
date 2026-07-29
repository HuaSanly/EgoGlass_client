from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator

from av import VideoFrame
from starlette.testclient import TestClient

from ingest_gateway.app import create_app
from ingest_gateway.decoded_preview import (
    DecodedPreviewRuntime,
    DecodedPreviewState,
    DecodedPreviewStatus,
)


async def _wait_until_encoded(runtime: DecodedPreviewRuntime, count: int) -> None:
    for _ in range(100):
        if (await runtime.status()).frames_encoded >= count:
            return
        await asyncio.sleep(0.001)
    raise AssertionError(f"preview did not encode {count} frames")


def test_slow_encoder_keeps_the_latest_pending_decoded_frame() -> None:
    started = threading.Event()
    release = threading.Event()
    encoded_widths: list[int] = []

    def encoder(frame: VideoFrame) -> bytes:
        encoded_widths.append(frame.width)
        if len(encoded_widths) == 1:
            started.set()
            release.wait(timeout=1)
        return f"jpeg-{frame.width}".encode()

    async def scenario() -> None:
        runtime = DecodedPreviewRuntime(encoder=encoder)
        await runtime.submit_gateway_frame(
            session_id="session",
            connection_session_id="connection",
            frame_index=0,
            received_at_client_monotonic_ns=10,
            decoded_frame=VideoFrame(10, 8, "bgr24"),
        )
        while not started.is_set():
            await asyncio.sleep(0)
        for frame_index, width in ((1, 12), (2, 14)):
            await runtime.submit_gateway_frame(
                session_id="session",
                connection_session_id="connection",
                frame_index=frame_index,
                received_at_client_monotonic_ns=10 + frame_index,
                decoded_frame=VideoFrame(width, 8, "bgr24"),
            )
        release.set()
        await _wait_until_encoded(runtime, 2)
        status = await runtime.status()

        assert encoded_widths == [10, 14]
        assert status.state is DecodedPreviewState.STREAMING
        assert status.frames_received == 3
        assert status.frames_encoded == 2
        assert status.frames_dropped == 1
        assert status.latest_frame_index == 2
        assert status.width == 14
        await runtime.close()

    asyncio.run(scenario())


def test_subscribers_share_one_encoded_jpeg_and_mjpeg_headers() -> None:
    async def scenario() -> None:
        runtime = DecodedPreviewRuntime(encoder=lambda _frame: b"shared-jpeg")
        await runtime.submit_gateway_frame(
            session_id="session",
            connection_session_id="connection",
            frame_index=7,
            received_at_client_monotonic_ns=100,
            decoded_frame=VideoFrame(16, 10, "bgr24"),
        )
        await _wait_until_encoded(runtime, 1)
        first_stream = runtime.stream()
        second_stream = runtime.stream()
        first, second = await asyncio.gather(anext(first_stream), anext(second_stream))

        assert first == second
        assert first.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n")
        assert b"Content-Length: 11\r\n" in first
        assert b"X-Frame-Index: 7\r\n\r\nshared-jpeg\r\n" in first
        assert (await runtime.status()).frames_encoded == 1
        await first_stream.aclose()
        await second_stream.aclose()
        assert (await runtime.status()).clients_connected == 0
        await runtime.close()

    asyncio.run(scenario())


def test_waiting_status_and_last_frame_survive_input_pause() -> None:
    now_ns = [1_000_000]

    async def scenario() -> None:
        runtime = DecodedPreviewRuntime(
            encoder=lambda _frame: b"jpeg",
            perf_clock=lambda: now_ns[0],
        )
        waiting = await runtime.status()
        assert waiting.state is DecodedPreviewState.WAITING
        assert waiting.latest_frame_index is None

        await runtime.submit_gateway_frame(
            session_id="session",
            connection_session_id="connection",
            frame_index=3,
            received_at_client_monotonic_ns=50,
            decoded_frame=VideoFrame(8, 6, "bgr24"),
        )
        await _wait_until_encoded(runtime, 1)
        now_ns[0] = 6_000_000
        paused = await runtime.status()
        assert paused.state is DecodedPreviewState.STREAMING
        assert paused.latest_frame_index == 3
        assert paused.last_frame_age_ms == 5.0
        stream = runtime.stream()
        assert b"X-Frame-Index: 3" in await anext(stream)
        await stream.aclose()
        await runtime.close()

    asyncio.run(scenario())


class _FinitePreviewRuntime:
    def __init__(self) -> None:
        self.closed = False

    async def status(self) -> DecodedPreviewStatus:
        return DecodedPreviewStatus(
            state=DecodedPreviewState.STREAMING,
            frames_received=4,
            frames_encoded=3,
            frames_dropped=1,
            clients_connected=1,
            width=16,
            height=10,
            output_fps=29.5,
            latest_frame_index=3,
        )

    async def stream(self) -> AsyncIterator[bytes]:
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\njpeg\r\n"

    async def close(self) -> None:
        self.closed = True


def test_decoded_preview_endpoints_are_loopback_only_and_disable_buffering() -> None:
    runtime = _FinitePreviewRuntime()
    origin = "http://127.0.0.1:8765"
    app = create_app(
        decoded_preview_runtime=runtime,  # type: ignore[arg-type]
        viewer_allowed_hosts=frozenset({"testclient"}),
    )
    with TestClient(app) as client:
        status = client.get(
            "/api/v1/webrtc/decoded-preview/status",
            headers={"Origin": origin},
        )
        preview = client.get("/api/v1/webrtc/decoded-preview.mjpg")

    assert status.status_code == 200
    assert status.json()["frames_encoded"] == 3
    assert status.headers["access-control-allow-origin"] == origin
    assert preview.status_code == 200
    assert preview.headers["content-type"].startswith(
        "multipart/x-mixed-replace; boundary=frame"
    )
    assert preview.headers["cache-control"].startswith("no-store")
    assert preview.headers["x-accel-buffering"] == "no"
    assert preview.content.endswith(b"jpeg\r\n")
    assert runtime.closed

    with TestClient(create_app()) as client:
        forbidden = client.get("/api/v1/webrtc/decoded-preview/status")
    assert forbidden.status_code == 403
