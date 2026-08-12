from __future__ import annotations

import argparse
import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi import Path as ApiPath
from fastapi.responses import FileResponse

from schemas.recording import (
    RecordingCommandRequest,
    RecordingLibrary,
    RecordingState,
    RecordingStatus,
)

from .discovery import DISCOVERY_PORT, LanDiscoveryService
from .imu_telemetry import ImuTelemetryRuntime
from .live_frames import LiveFrameBuffer, LiveFrameStatus
from .recording import (
    RecordingConflictError,
    RecordingFailureError,
    RecordingNotFoundError,
    RecordingRuntime,
    RecordingUnavailableError,
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
    live_frame_buffer: LiveFrameBuffer | None = None,
    imu_telemetry_runtime: ImuTelemetryRuntime | None = None,
    *,
    recordings_root: Path | None = None,
    viewer_allowed_hosts: frozenset[str] = frozenset({"127.0.0.1", "::1"}),
) -> FastAPI:
    active_webrtc = webrtc_runtime or WebRtcSessionRuntime(secrets.token_urlsafe(24))
    active_recording = recording_runtime or RecordingRuntime(
        recordings_root or Path("local-data/recordings"),
        lambda: active_webrtc.recording_source(),
    )
    active_webrtc.set_capture_telemetry_sink(active_recording)
    if live_frame_buffer is not None:
        active_webrtc.set_display_frame_sink(live_frame_buffer)
    if imu_telemetry_runtime is not None:
        active_webrtc.set_display_imu_sink(imu_telemetry_runtime)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if discovery_service is not None:
            await discovery_service.start()
        try:
            yield
        finally:
            if discovery_service is not None:
                await discovery_service.close()
            await active_recording.close()
            await active_webrtc.close()
            if live_frame_buffer is not None:
                await live_frame_buffer.close()
            if imu_telemetry_runtime is not None:
                await imu_telemetry_runtime.close()

    app = FastAPI(
        title="EgoGlass Recording Gateway",
        version="0.2.0",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.webrtc_runtime = active_webrtc
    app.state.recording_runtime = active_recording
    app.state.live_frame_buffer = live_frame_buffer
    app.state.imu_telemetry_runtime = imu_telemetry_runtime
    app.state.discovery_service = discovery_service

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "recording-gateway", "version": "0.2.0"}

    @app.get("/api/v1/webrtc/status", response_model=WebRtcStatus)
    async def webrtc_status() -> WebRtcStatus:
        return await active_webrtc.status()

    @app.get("/api/v1/native-display/status", response_model=LiveFrameStatus)
    async def native_display_status(request: Request) -> LiveFrameStatus:
        _require_loopback(request, viewer_allowed_hosts, "native display")
        if live_frame_buffer is None:
            raise HTTPException(status_code=404, detail="native display is unavailable")
        return live_frame_buffer.status()

    @app.get("/api/v1/webrtc/imu/status", response_model=ImuTelemetryStatus)
    async def imu_status(request: Request) -> ImuTelemetryStatus:
        _require_loopback(request, viewer_allowed_hosts, "IMU telemetry")
        return await active_webrtc.imu_status()

    @app.get("/api/v1/native-imu/status")
    async def native_imu_status(request: Request) -> dict[str, object]:
        _require_loopback(request, viewer_allowed_hosts, "native IMU monitor")
        if imu_telemetry_runtime is None:
            raise HTTPException(status_code=404, detail="native IMU monitor is unavailable")
        return asdict(imu_telemetry_runtime.snapshot())

    @app.get("/api/v1/webrtc/control", response_model=StreamControlStatus)
    async def stream_control_status(request: Request) -> StreamControlStatus:
        _require_loopback(request, viewer_allowed_hosts, "stream control")
        return await active_webrtc.control_status()

    @app.post("/api/v1/webrtc/control/commands", response_model=StreamControlStatus)
    async def stream_control_command(
        command: StreamControlRequest,
        request: Request,
    ) -> StreamControlStatus:
        _require_loopback(request, viewer_allowed_hosts, "stream control")
        action = StreamControlAction(command.action)
        if action is StreamControlAction.STOP:
            recording = await active_recording.status()
            if recording.state in {
                RecordingState.COUNTDOWN,
                RecordingState.RECORDING,
                RecordingState.FINALIZING,
            }:
                raise HTTPException(
                    status_code=409,
                    detail="stop the active recording before stopping the video stream",
                )
        try:
            return await active_webrtc.send_control_command(
                StreamControlCommand(command_id=secrets.token_hex(16), action=action)
            )
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
        try:
            return await active_webrtc.accept_offer(offer, _bearer_token(authorization))
        except PairingTokenError as error:
            raise HTTPException(status_code=401, detail=str(error)) from error
        except WebRtcSessionError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.get("/api/v1/recordings/status", response_model=RecordingStatus)
    async def recording_status(request: Request) -> RecordingStatus:
        _require_loopback(request, viewer_allowed_hosts, "recording")
        return await active_recording.status()

    @app.post("/api/v1/recordings/commands", response_model=RecordingStatus)
    async def recording_command(
        command: RecordingCommandRequest,
        request: Request,
    ) -> RecordingStatus:
        _require_loopback(request, viewer_allowed_hosts, "recording")
        try:
            return (
                await active_recording.start()
                if command.action == "start"
                else await active_recording.stop()
            )
        except RecordingUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except RecordingConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RecordingFailureError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @app.get("/api/v1/recordings/library", response_model=RecordingLibrary)
    async def recording_library(request: Request) -> RecordingLibrary:
        _require_loopback(request, viewer_allowed_hosts, "recording library")
        return await active_recording.library()

    @app.delete("/api/v1/recordings/{recording_id}", response_model=RecordingLibrary)
    async def delete_recording(
        request: Request,
        recording_id: Annotated[str, ApiPath(pattern=r"^[0-9a-f]{32}$")],
    ) -> RecordingLibrary:
        _require_loopback(request, viewer_allowed_hosts, "recording deletion")
        try:
            return await active_recording.delete(recording_id)
        except RecordingConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RecordingNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except RecordingFailureError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @app.get("/api/v1/recordings/{recording_id}/video.mp4")
    async def recording_media(
        request: Request,
        recording_id: Annotated[str, ApiPath(pattern=r"^[0-9a-f]{32}$")],
    ) -> FileResponse:
        _require_loopback(request, viewer_allowed_hosts, "recording media")
        media_path = await active_recording.media_path(recording_id)
        if media_path is None:
            raise HTTPException(status_code=404, detail="recording not found")
        return FileResponse(media_path, media_type="video/mp4")

    @app.get("/api/v1/recordings/{recording_id}/{artifact}")
    async def recording_csv(
        request: Request,
        recording_id: Annotated[str, ApiPath(pattern=r"^[0-9a-f]{32}$")],
        artifact: Annotated[str, ApiPath(pattern=r"^(imu|frames)[.]csv$")],
    ) -> FileResponse:
        _require_loopback(request, viewer_allowed_hosts, "recording CSV")
        artifact_path = await active_recording.artifact_path(recording_id, artifact)
        if artifact_path is None:
            raise HTTPException(status_code=404, detail="recording artifact not found")
        return FileResponse(artifact_path, media_type="text/csv", filename=artifact)

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
    parser = argparse.ArgumentParser(description="Receive and record the EgoGlass stream")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--discovery-port", type=int, default=DISCOVERY_PORT)
    parser.add_argument("--pairing-token", default=os.environ.get("EGOGLASS_PAIRING_TOKEN"))
    parser.add_argument("--disable-discovery", action="store_true")
    parser.add_argument("--hide-pairing-token", action="store_true")
    parser.add_argument(
        "--recordings-root",
        type=Path,
        default=Path(os.environ.get("EGOGLASS_RECORDINGS_ROOT", "local-data/recordings")),
    )
    args = parser.parse_args()
    pairing_token = args.pairing_token or secrets.token_urlsafe(24)
    if len(pairing_token) < 16:
        parser.error("--pairing-token must contain at least 16 characters")
    if not args.hide_pairing_token:
        print(f"EgoGlass WebRTC pairing token: {pairing_token}", flush=True)
    discovery = (
        None
        if args.disable_discovery
        else LanDiscoveryService(
            pairing_token,
            args.port,
            discovery_port=args.discovery_port,
        )
    )
    uvicorn.run(
        create_app(
            webrtc_runtime=WebRtcSessionRuntime(pairing_token),
            discovery_service=discovery,
            recordings_root=args.recordings_root,
        ),
        host=args.host,
        port=args.port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
