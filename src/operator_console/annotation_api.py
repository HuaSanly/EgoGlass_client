from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException
from fastapi import Path as ApiPath
from fastapi.responses import FileResponse

from .annotation_models import (
    AnnotationDraft,
    AnnotationSession,
    AnnotationWorkspace,
    ProposalBatch,
    ProposalRequest,
    PublishedRevision,
    PublishRequest,
    SaveDraftRequest,
)
from .annotation_store import (
    AnnotationNotFoundError,
    AnnotationReadOnlyError,
    AnnotationStore,
    AnnotationValidationError,
    RevisionConflictError,
)


def create_annotation_router(store: AnnotationStore) -> APIRouter:
    router = APIRouter(prefix="/api/v1/annotations", tags=["annotations"])

    @router.get("/workspace", response_model=AnnotationWorkspace)
    async def annotation_workspace() -> AnnotationWorkspace:
        return store.workspace()

    @router.get("/sessions/{session_id}", response_model=AnnotationSession)
    async def annotation_session(
        session_id: Annotated[str, ApiPath(pattern=r"^[0-9a-f]{32}$")],
    ) -> AnnotationSession:
        try:
            return store.session(session_id)
        except AnnotationNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.put("/sessions/{session_id}/draft", response_model=AnnotationDraft)
    async def save_annotation_draft(
        session_id: Annotated[str, ApiPath(pattern=r"^[0-9a-f]{32}$")],
        request: SaveDraftRequest,
    ) -> AnnotationDraft:
        try:
            return store.save_draft(session_id, request)
        except AnnotationNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (AnnotationReadOnlyError, RevisionConflictError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except AnnotationValidationError as error:
            raise HTTPException(status_code=422, detail={"issues": error.issues}) from error

    @router.post("/sessions/{session_id}/proposals", response_model=ProposalBatch)
    async def create_episode_proposals(
        session_id: Annotated[str, ApiPath(pattern=r"^[0-9a-f]{32}$")],
        request: ProposalRequest,
    ) -> ProposalBatch:
        try:
            return store.proposals(session_id, request)
        except AnnotationNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @router.post("/sessions/{session_id}/publish", response_model=PublishedRevision)
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

    @router.get("/media/{session_id}/{clip_id}")
    async def annotation_media(
        session_id: Annotated[str, ApiPath(pattern=r"^[0-9a-f]{32}$")],
        clip_id: Annotated[str, ApiPath(pattern=r"^[0-9a-f]{32}$")],
    ) -> FileResponse:
        try:
            return FileResponse(store.media_path(session_id, clip_id), media_type="video/mp4")
        except AnnotationNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    return router
