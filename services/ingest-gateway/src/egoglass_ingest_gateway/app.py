from __future__ import annotations

import argparse
import os
import secrets
from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from .adapters.rtsp import RtspProbeError
from .models import IngestStatus, ProbeResult, RtspSourceConfig
from .runtime import IngestRuntime, ProbeBusyError
from .webrtc_models import WebRtcAnswer, WebRtcOffer, WebRtcStatus
from .webrtc_runtime import (
    PairingTokenError,
    WebRtcSessionBusyError,
    WebRtcSessionError,
    WebRtcSessionRuntime,
)


def create_app(
    runtime: IngestRuntime | None = None,
    webrtc_runtime: WebRtcSessionRuntime | None = None,
    *,
    preview_allowed_hosts: frozenset[str] = frozenset({"127.0.0.1", "::1"}),
) -> FastAPI:
    ingest_runtime = runtime or IngestRuntime()
    active_webrtc_runtime = webrtc_runtime or WebRtcSessionRuntime(secrets.token_urlsafe(24))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await active_webrtc_runtime.close()

    app = FastAPI(
        title="EgoGlass Ingest Gateway",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.ingest_runtime = ingest_runtime
    app.state.webrtc_runtime = active_webrtc_runtime

    @app.exception_handler(RequestValidationError)
    async def redact_validation_input(
        _request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        detail = [
            {key: value for key, value in item.items() if key not in {"input", "ctx"}}
            for item in error.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "ingest-gateway", "version": "0.1.0"}

    @app.get("/api/v1/status", response_model=IngestStatus)
    async def status() -> IngestStatus:
        return await ingest_runtime.status()

    @app.post("/api/v1/rtsp/probe", response_model=ProbeResult)
    async def probe(config: RtspSourceConfig) -> ProbeResult:
        try:
            return await ingest_runtime.probe(config)
        except ProbeBusyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RtspProbeError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.get("/api/v1/webrtc/status", response_model=WebRtcStatus)
    async def webrtc_status() -> WebRtcStatus:
        return await active_webrtc_runtime.status()

    @app.get("/api/v1/webrtc/frame.jpg", response_class=Response)
    async def webrtc_preview_frame(request: Request) -> Response:
        client_host = request.client.host if request.client is not None else ""
        if client_host not in preview_allowed_hosts:
            raise HTTPException(status_code=403, detail="preview is available on loopback only")
        payload = await active_webrtc_runtime.latest_preview_jpeg()
        if payload is None:
            raise HTTPException(status_code=503, detail="video preview is not ready")
        return Response(
            content=payload,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

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
        except WebRtcSessionBusyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except WebRtcSessionError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    return app


def _bearer_token(authorization: str | None) -> str:
    prefix = "Bearer "
    if authorization is None or not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="Bearer pairing token required")
    token = authorization[len(prefix) :]
    if not token:
        raise HTTPException(status_code=401, detail="Bearer pairing token required")
    return token


app = create_app(
    webrtc_runtime=WebRtcSessionRuntime(
        os.environ.get("EGOGLASS_PAIRING_TOKEN") or secrets.token_urlsafe(24)
    )
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive EgoGlass WebRTC or probe RTSP streams")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument(
        "--pairing-token",
        default=os.environ.get("EGOGLASS_PAIRING_TOKEN"),
        help="Runtime WebRTC pairing secret; generated when omitted",
    )
    args = parser.parse_args()
    pairing_token = args.pairing_token or secrets.token_urlsafe(24)
    if len(pairing_token) < 16:
        parser.error("--pairing-token must contain at least 16 characters")
    print(f"EgoGlass WebRTC pairing token: {pairing_token}", flush=True)
    uvicorn.run(
        create_app(webrtc_runtime=WebRtcSessionRuntime(pairing_token)),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
