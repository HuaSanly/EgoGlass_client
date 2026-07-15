from __future__ import annotations

import argparse
import secrets
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"
DESKTOP_COOKIE_NAME = "egoglass_desktop_session"


def create_app(
    *,
    desktop_token: str | None = None,
) -> FastAPI:
    app = FastAPI(
        title="EgoGlass Operator Console",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.desktop_token = desktop_token

    @app.middleware("http")
    async def require_desktop_session(request: Request, call_next):
        if desktop_token is None or request.url.path == "/api/v1/health":
            return await call_next(request)

        cookie_token = request.cookies.get(DESKTOP_COOKIE_NAME, "")
        query_token = request.query_params.get("desktop_token", "")
        cookie_valid = bool(cookie_token) and secrets.compare_digest(cookie_token, desktop_token)
        query_valid = bool(query_token) and secrets.compare_digest(query_token, desktop_token)
        if not cookie_valid and not query_valid:
            return JSONResponse(status_code=401, content={"detail": "desktop session required"})

        response = await call_next(request)
        if query_valid:
            response.set_cookie(
                DESKTOP_COOKIE_NAME,
                desktop_token,
                httponly=True,
                samesite="strict",
            )
        return response

    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "operator-console", "version": "0.1.0"}

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the EgoGlass operator console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run(
        "egoglass_operator_console.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
