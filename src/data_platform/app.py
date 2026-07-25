from __future__ import annotations

import argparse
import ipaddress
import os
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi import Path as ApiPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .models import (
    AnnotationDraft,
    AnnotationSession,
    AnnotationWorkspace,
    ProposalBatch,
    ProposalRequest,
    PublishedRevision,
    PublishRequest,
    SaveDraftRequest,
)
from .store import (
    AnnotationNotFoundError,
    AnnotationReadOnlyError,
    AnnotationStore,
    AnnotationValidationError,
    RevisionConflictError,
)


def _is_loopback_client(host: str) -> bool:
    if host == "testclient":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def create_app(recordings_root: Path | None = None) -> FastAPI:
    root = recordings_root or Path(
        os.environ.get("EGOGLASS_RECORDINGS_ROOT", "local-data/recordings")
    )
    store = AnnotationStore(root)
    app = FastAPI(
        title="EgoGlass Data Platform",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.annotation_store = store
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^http://127[.]0[.]0[.]1(?::[0-9]+)?$",
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def require_loopback_client(request: Request, call_next):
        client_host = "" if request.client is None else request.client.host
        if not _is_loopback_client(client_host):
            return JSONResponse(status_code=403, content={"detail": "loopback client required"})
        return await call_next(request)

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "data-platform", "version": "0.1.0"}

    @app.get("/api/v1/annotations/workspace", response_model=AnnotationWorkspace)
    async def annotation_workspace() -> AnnotationWorkspace:
        return store.workspace()

    @app.get(
        "/api/v1/annotations/sessions/{session_id}",
        response_model=AnnotationSession,
    )
    async def annotation_session(
        session_id: Annotated[str, ApiPath(pattern=r"^[0-9a-f]{32}$")],
    ) -> AnnotationSession:
        try:
            return store.session(session_id)
        except AnnotationNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.put(
        "/api/v1/annotations/sessions/{session_id}/draft",
        response_model=AnnotationDraft,
    )
    async def save_annotation_draft(
        session_id: Annotated[str, ApiPath(pattern=r"^[0-9a-f]{32}$")],
        request: SaveDraftRequest,
    ) -> AnnotationDraft:
        try:
            return store.save_draft(session_id, request)
        except AnnotationNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except AnnotationReadOnlyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RevisionConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except AnnotationValidationError as error:
            raise HTTPException(status_code=422, detail={"issues": error.issues}) from error

    @app.post(
        "/api/v1/annotations/sessions/{session_id}/proposals",
        response_model=ProposalBatch,
    )
    async def create_episode_proposals(
        session_id: Annotated[str, ApiPath(pattern=r"^[0-9a-f]{32}$")],
        request: ProposalRequest,
    ) -> ProposalBatch:
        try:
            return store.proposals(session_id, request)
        except AnnotationNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post(
        "/api/v1/annotations/sessions/{session_id}/publish",
        response_model=PublishedRevision,
    )
    async def publish_annotation_revision(
        session_id: Annotated[str, ApiPath(pattern=r"^[0-9a-f]{32}$")],
        request: PublishRequest,
    ) -> PublishedRevision:
        try:
            return store.publish(session_id, request.base_revision)
        except AnnotationNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (AnnotationReadOnlyError, RevisionConflictError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except AnnotationValidationError as error:
            raise HTTPException(status_code=422, detail={"issues": error.issues}) from error

    @app.get("/api/v1/annotations/media/{session_id}/{clip_id}")
    async def annotation_media(
        request: Request,
        session_id: Annotated[str, ApiPath(pattern=r"^[0-9a-f]{32}$")],
        clip_id: Annotated[str, ApiPath(pattern=r"^[0-9a-f]{32}$")],
    ) -> FileResponse:
        del request
        try:
            return FileResponse(store.media_path(session_id, clip_id), media_type="video/mp4")
        except AnnotationNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the EgoGlass local data platform")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument(
        "--recordings-root",
        type=Path,
        default=Path(os.environ.get("EGOGLASS_RECORDINGS_ROOT", "local-data/recordings")),
    )
    args = parser.parse_args()
    try:
        address = ipaddress.ip_address(args.host)
    except ValueError as error:
        raise SystemExit("data-platform host must be a loopback IP address") from error
    if not address.is_loopback:
        raise SystemExit("data-platform may only bind to loopback")
    uvicorn.run(create_app(args.recordings_root), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
