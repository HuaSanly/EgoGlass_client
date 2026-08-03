from __future__ import annotations

import argparse
import json
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi import Path as ApiPath
from fastapi.responses import FileResponse, StreamingResponse

from perception.runtime import HandTrackingRuntime

from .discovery import DISCOVERY_PORT, LanDiscoveryService
from .imu_preview import ImuPreviewRuntime
from .live_frames import LiveFrameBuffer, LiveFrameStatus
from .recording import (
    RecordingClipNotFoundError,
    RecordingConflictError,
    RecordingFailureError,
    RecordingRuntime,
    RecordingSessionNotFoundError,
    RecordingUnavailableError,
)
from .recording_models import (
    RecordingCommandRequest,
    RecordingLibrary,
    RecordingSessionCommandRequest,
    RecordingSessionRenameRequest,
    RecordingState,
    RecordingStatus,
)
from .webrtc_models import (
    ImuTelemetryStatus,
    StreamControlAction,
    StreamControlCommand,
    StreamControlRequest,
    StreamControlStatus,
    WebRtcAnswer,
    WebRtcOffer,
    WebRtcStatus,
)
from .webrtc_runtime import (
    PairingTokenError,
    StreamControlCommandError,
    StreamControlCommandTimeoutError,
    StreamControlUnavailableError,
    WebRtcSessionError,
    WebRtcSessionRuntime,
)


def create_app(
    webrtc_runtime: WebRtcSessionRuntime | None = None,
    discovery_service: LanDiscoveryService | None = None,
    recording_runtime: RecordingRuntime | None = None,
    perception_runtime: HandTrackingRuntime | None = None,
    live_frame_buffer: LiveFrameBuffer | None = None,
    imu_preview_runtime: ImuPreviewRuntime | None = None,
    *,
    recordings_root: Path | None = None,
    viewer_allowed_hosts: frozenset[str] = frozenset({"127.0.0.1", "::1"}),
) -> FastAPI:
    active_webrtc_runtime = webrtc_runtime or WebRtcSessionRuntime(secrets.token_urlsafe(24))
    active_recordings_root = recordings_root or Path("local-data/recordings")
    active_recording_runtime = recording_runtime or RecordingRuntime(
        active_recordings_root,
        lambda: active_webrtc_runtime.recording_source(),
    )
    active_perception_runtime = perception_runtime or HandTrackingRuntime()
    active_live_frame_buffer = live_frame_buffer
    active_imu_preview_runtime = imu_preview_runtime
    set_capture_sink = getattr(active_webrtc_runtime, "set_capture_telemetry_sink", None)
    if set_capture_sink is not None:
        set_capture_sink(active_recording_runtime)
    set_perception_sink = getattr(active_webrtc_runtime, "set_perception_live_frame_sink", None)
    if set_perception_sink is not None:
        set_perception_sink(
            active_perception_runtime if active_live_frame_buffer is None else None
        )
    set_display_sink = getattr(active_webrtc_runtime, "set_display_frame_sink", None)
    if set_display_sink is not None and active_live_frame_buffer is not None:
        set_display_sink(active_live_frame_buffer)
        active_live_frame_buffer.set_rgb_frame_sink(active_perception_runtime)
    set_display_imu_sink = getattr(active_webrtc_runtime, "set_display_imu_sink", None)
    if set_display_imu_sink is not None and active_imu_preview_runtime is not None:
        set_display_imu_sink(active_imu_preview_runtime)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if discovery_service is not None:
            await discovery_service.start()
        try:
            yield
        finally:
            if discovery_service is not None:
                await discovery_service.close()
            await active_webrtc_runtime.close()
            if active_live_frame_buffer is not None:
                await active_live_frame_buffer.close()
            await active_perception_runtime.close()
            if active_imu_preview_runtime is not None:
                await active_imu_preview_runtime.close()
            await active_recording_runtime.close()

    app = FastAPI(
        title="EgoGlass Ingest Gateway",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.webrtc_runtime = active_webrtc_runtime
    app.state.recording_runtime = active_recording_runtime
    app.state.perception_runtime = active_perception_runtime
    app.state.live_frame_buffer = active_live_frame_buffer
    app.state.imu_preview_runtime = active_imu_preview_runtime
    app.state.discovery_service = discovery_service
    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "ingest-gateway", "version": "0.1.0"}

    @app.get("/api/v1/webrtc/status", response_model=WebRtcStatus)
    async def webrtc_status() -> WebRtcStatus:
        return await active_webrtc_runtime.status()

    @app.get("/api/v1/native-display/status", response_model=LiveFrameStatus)
    async def native_display_status(request: Request) -> LiveFrameStatus:
        _require_loopback(request, viewer_allowed_hosts, "native display")
        if active_live_frame_buffer is None:
            raise HTTPException(status_code=404, detail="native display is unavailable")
        return active_live_frame_buffer.status()

    @app.get("/api/v1/webrtc/imu/status", response_model=ImuTelemetryStatus)
    async def imu_telemetry_status(request: Request) -> ImuTelemetryStatus:
        _require_loopback(request, viewer_allowed_hosts, "IMU telemetry")
        return await active_webrtc_runtime.imu_status()

    @app.get("/api/v1/perception/hand-tracking/status")
    async def hand_tracking_status(request: Request) -> dict[str, object]:
        _require_loopback(request, viewer_allowed_hosts, "hand tracking")
        return await active_perception_runtime.status()

    @app.get("/api/v1/perception/hand-tracking/events")
    async def hand_tracking_events(request: Request) -> StreamingResponse:
        _require_loopback(request, viewer_allowed_hosts, "hand tracking")

        async def event_stream():
            async for payload in active_perception_runtime.status_events():
                if await request.is_disconnected():
                    return
                if payload is None:
                    yield ": heartbeat\n\n"
                    continue
                data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                yield f"event: status\ndata: {data}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/v1/webrtc/control", response_model=StreamControlStatus)
    async def get_stream_control_status(request: Request) -> StreamControlStatus:
        _require_loopback(request, viewer_allowed_hosts, "stream control")
        return await active_webrtc_runtime.control_status()

    @app.post(
        "/api/v1/webrtc/control/commands",
        response_model=StreamControlStatus,
    )
    async def send_stream_control_command(
        control_request: StreamControlRequest,
        request: Request,
    ) -> StreamControlStatus:
        _require_loopback(request, viewer_allowed_hosts, "stream control")
        if control_request.action == StreamControlAction.STOP:
            recording = await active_recording_runtime.status()
            if recording.state in {
                RecordingState.COUNTDOWN,
                RecordingState.RECORDING,
                RecordingState.FINALIZING,
            }:
                raise HTTPException(
                    status_code=409,
                    detail="stop the active recording before stopping the video stream",
                )
        command = StreamControlCommand(
            command_id=secrets.token_hex(16),
            action=StreamControlAction(control_request.action),
        )
        try:
            return await active_webrtc_runtime.send_control_command(command)
        except StreamControlUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except StreamControlCommandTimeoutError as error:
            raise HTTPException(status_code=504, detail=str(error)) from error
        except StreamControlCommandError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.post("/api/v1/webrtc/sessions", response_model=WebRtcAnswer)
    async def create_webrtc_session(
        offer: WebRtcOffer,
        authorization: Annotated[str | None, Header()] = None,
    ) -> WebRtcAnswer:
        token = _bearer_token(authorization)
        try:
            return await active_webrtc_runtime.accept_offer(offer, token)
        except PairingTokenError as error:
            raise HTTPException(status_code=401, detail=str(error)) from error
        except WebRtcSessionError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.get("/api/v1/recordings/status", response_model=RecordingStatus)
    async def recording_status(request: Request) -> RecordingStatus:
        _require_loopback(request, viewer_allowed_hosts, "recording")
        return await active_recording_runtime.status()

    @app.post("/api/v1/recordings/commands", response_model=RecordingStatus)
    async def recording_command(
        command: RecordingCommandRequest,
        request: Request,
    ) -> RecordingStatus:
        _require_loopback(request, viewer_allowed_hosts, "recording")
        try:
            if command.action == "start":
                return await active_recording_runtime.start()
            return await active_recording_runtime.stop()
        except RecordingUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except RecordingConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RecordingFailureError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @app.get("/api/v1/recordings/library", response_model=RecordingLibrary)
    async def recording_library(request: Request) -> RecordingLibrary:
        _require_loopback(request, viewer_allowed_hosts, "recording library")
        return await active_recording_runtime.library()

    @app.post(
        "/api/v1/recordings/session-commands",
        response_model=RecordingStatus,
    )
    async def recording_session_command(
        command: RecordingSessionCommandRequest,
        request: Request,
    ) -> RecordingStatus:
        _require_loopback(request, viewer_allowed_hosts, "recording session")
        try:
            return await active_recording_runtime.session_command(command.action)
        except RecordingConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RecordingFailureError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @app.patch(
        "/api/v1/recordings/sessions/{session_id}",
        response_model=RecordingLibrary,
    )
    async def rename_recording_session(
        request: Request,
        session_id: Annotated[str, ApiPath(pattern=r"^[0-9a-f]{32}$")],
        command: RecordingSessionRenameRequest,
    ) -> RecordingLibrary:
        _require_loopback(request, viewer_allowed_hosts, "recording rename")
        try:
            return await active_recording_runtime.rename_session(
                session_id,
                command.display_name,
            )
        except RecordingSessionNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except RecordingFailureError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @app.delete(
        "/api/v1/recordings/clips/{session_id}/{clip_id}",
        response_model=RecordingLibrary,
    )
    async def delete_recording_clip(
        request: Request,
        session_id: Annotated[str, ApiPath(pattern=r"^[0-9a-f]{32}$")],
        clip_id: Annotated[str, ApiPath(pattern=r"^[0-9a-f]{32}$")],
    ) -> RecordingLibrary:
        _require_loopback(request, viewer_allowed_hosts, "recording deletion")
        try:
            return await active_recording_runtime.delete_clip(session_id, clip_id)
        except RecordingConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RecordingClipNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except RecordingFailureError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @app.delete(
        "/api/v1/recordings/sessions/{session_id}",
        response_model=RecordingLibrary,
    )
    async def delete_recording_session(
        request: Request,
        session_id: Annotated[str, ApiPath(pattern=r"^[0-9a-f]{32}$")],
    ) -> RecordingLibrary:
        _require_loopback(request, viewer_allowed_hosts, "recording session deletion")
        try:
            return await active_recording_runtime.delete_session(session_id)
        except RecordingConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RecordingSessionNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except RecordingFailureError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @app.get("/api/v1/recordings/media/{session_id}/{clip_id}")
    async def recording_media(
        request: Request,
        session_id: Annotated[str, ApiPath(pattern=r"^[0-9a-f]{32}$")],
        clip_id: Annotated[str, ApiPath(pattern=r"^[0-9a-f]{32}$")],
    ) -> FileResponse:
        _require_loopback(request, viewer_allowed_hosts, "recording media")
        media_path = await active_recording_runtime.media_path(session_id, clip_id)
        if media_path is None:
            raise HTTPException(status_code=404, detail="recording clip not found")
        return FileResponse(media_path, media_type="video/mp4")

    return app


def _bearer_token(authorization: str | None) -> str:
    prefix = "Bearer "
    if authorization is None or not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="Bearer pairing token required")
    token = authorization[len(prefix) :]
    if not token:
        raise HTTPException(status_code=401, detail="Bearer pairing token required")
    return token


def _require_loopback(
    request: Request,
    allowed_hosts: frozenset[str],
    resource: str,
) -> None:
    client_host = request.client.host if request.client is not None else ""
    if client_host not in allowed_hosts:
        raise HTTPException(status_code=403, detail=f"{resource} is available on loopback only")


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive the EgoGlass WebRTC stream")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--discovery-port", type=int, default=DISCOVERY_PORT)
    parser.add_argument(
        "--pairing-token",
        default=os.environ.get("EGOGLASS_PAIRING_TOKEN"),
        help="Runtime WebRTC pairing secret; generated when omitted",
    )
    parser.add_argument("--disable-discovery", action="store_true")
    parser.add_argument("--hide-pairing-token", action="store_true")
    parser.add_argument(
        "--recordings-root",
        type=Path,
        default=Path(os.environ.get("EGOGLASS_RECORDINGS_ROOT", "local-data/recordings")),
        help="Directory for completed MP4 clips and session manifests",
    )
    args = parser.parse_args()
    pairing_token = args.pairing_token or secrets.token_urlsafe(24)
    if len(pairing_token) < 16:
        parser.error("--pairing-token must contain at least 16 characters")
    if not args.hide_pairing_token:
        print(f"EgoGlass WebRTC pairing token: {pairing_token}", flush=True)
    discovery_service = None
    if not args.disable_discovery:
        discovery_service = LanDiscoveryService(
            pairing_token,
            args.port,
            discovery_port=args.discovery_port,
        )
    uvicorn.run(
        create_app(
            webrtc_runtime=WebRtcSessionRuntime(pairing_token),
            discovery_service=discovery_service,
            recordings_root=args.recordings_root,
        ),
        host=args.host,
        port=args.port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
