from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .models import ConsoleState, RuntimeSettings
from .runtime import ConsoleRuntime

STATIC_DIR = Path(__file__).parent / "static"


def create_app(runtime: ConsoleRuntime | None = None) -> FastAPI:
    console_runtime = runtime or ConsoleRuntime()
    app = FastAPI(
        title="EgoGlass Operator Console",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.console_runtime = console_runtime
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "operator-console", "version": "0.1.0"}

    @app.get("/api/v1/state", response_model=ConsoleState)
    async def get_state() -> ConsoleState:
        return await console_runtime.state()

    @app.put("/api/v1/settings", response_model=ConsoleState)
    async def update_settings(settings: RuntimeSettings) -> ConsoleState:
        return await console_runtime.update_settings(settings)

    @app.post("/api/v1/session/start", response_model=ConsoleState)
    async def start_session() -> ConsoleState:
        return await console_runtime.set_session_active(True)

    @app.post("/api/v1/session/stop", response_model=ConsoleState)
    async def stop_session() -> ConsoleState:
        return await console_runtime.set_session_active(False)

    @app.post("/api/v1/recording/start", response_model=ConsoleState)
    async def start_recording() -> ConsoleState:
        try:
            return await console_runtime.set_recording(True)
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/v1/recording/stop", response_model=ConsoleState)
    async def stop_recording() -> ConsoleState:
        return await console_runtime.set_recording(False)

    @app.websocket("/api/v1/telemetry")
    async def telemetry(websocket: WebSocket) -> None:
        await websocket.accept()
        tick = 0
        try:
            while True:
                snapshot = await console_runtime.telemetry(tick)
                await websocket.send_json(snapshot.model_dump(mode="json"))
                tick += 1
                state = await console_runtime.state()
                await asyncio.sleep(1 / state.settings.inference_fps)
        except WebSocketDisconnect:
            return

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
