from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import sys
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, HTTPException

from ui.gateway.discovery import DISCOVERY_PORT, LanDiscoveryService
from ui.gateway.webrtc_models import WebRtcAnswer, WebRtcOffer
from ui.gateway.webrtc_runtime import PairingTokenError, WebRtcSessionError, WebRtcSessionRuntime

from .adb_device import AdbDevicePreparationError, AdbGlassController
from .service import ImuCalibrationService


class _ImuCaptureServer(uvicorn.Server):
    def install_signal_handlers(self) -> None:
        """Let asyncio.run route Ctrl+C through the capture finalizer."""


def create_app(
    runtime: WebRtcSessionRuntime,
    service: ImuCalibrationService,
    discovery: LanDiscoveryService | None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if discovery is not None:
            await discovery.start()
        try:
            yield
        finally:
            if discovery is not None:
                await discovery.close()
            await runtime.close()

    app = FastAPI(
        title="EgoGlass IMU Calibration Capture",
        version="1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "imu-calibration-capture"}

    @app.post("/api/v1/webrtc/sessions", response_model=WebRtcAnswer)
    async def session(
        offer: WebRtcOffer, authorization: str | None = Header(default=None)
    ) -> WebRtcAnswer:
        prefix = "Bearer "
        if authorization is None or not authorization.startswith(prefix):
            raise HTTPException(status_code=401, detail="Bearer pairing token required")
        try:
            return await runtime.accept_offer(offer, authorization[len(prefix) :])
        except PairingTokenError as error:
            raise HTTPException(status_code=401, detail=str(error)) from error
        except WebRtcSessionError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    app.state.imu_capture_service = service
    return app


async def _run(args: argparse.Namespace) -> int:
    adb_controller = AdbGlassController(args.adb_serial) if args.adb_serial is not None else None
    if adb_controller is not None:
        network = await asyncio.to_thread(adb_controller.preflight)
        print(
            f"ADB Glass3 ready: serial={adb_controller.serial} "
            f"glasses={network.device_address} client={network.client_address}",
            flush=True,
        )
    pairing_token = args.pairing_token or secrets.token_urlsafe(24)
    runtime = WebRtcSessionRuntime(pairing_token)
    service = ImuCalibrationService(runtime, args.output_root)
    print(
        f"Waiting for Glass3 IMU: capture_id={service.writer.capture_id}",
        flush=True,
    )
    discovery = (
        None
        if args.disable_discovery
        else LanDiscoveryService(pairing_token, args.port, discovery_port=args.discovery_port)
    )
    app = create_app(runtime, service, discovery)
    server = _ImuCaptureServer(
        uvicorn.Config(app, host=args.host, port=args.port, access_log=False, log_level="warning")
    )
    server_task = asyncio.create_task(server.serve())
    server_watchdog_task = asyncio.create_task(_watch_server(server_task, service))
    reporter_task = asyncio.create_task(_report(service, args.duration_seconds))
    watchdog_task = asyncio.create_task(service.watchdog())
    try:
        while not server.started:
            if server_task.done():
                await server_task
                raise RuntimeError("IMU capture gateway stopped during startup")
            await asyncio.sleep(0.01)
        if adb_controller is not None:
            await asyncio.to_thread(adb_controller.prepare_and_launch)
            print("ADB Glass3 launched and held awake over USB", flush=True)
        await service.wait_until_started_or_done()
        if service.phase.value == "failed":
            print(f"IMU capture failed: {service.status().error}", flush=True)
            return 1
        if args.until_interrupted:
            await service.wait()
        else:
            try:
                await service.wait(args.duration_seconds)
            except TimeoutError:
                await service.finish()
        if service.phase.value == "complete":
            print(f"Published IMU capture: {service.writer.final_path}", flush=True)
            return 0
        print(f"IMU capture failed: {service.status().error}", flush=True)
        return 1
    except (KeyboardInterrupt, asyncio.CancelledError):
        if service.phase.value == "capturing":
            await service.finish()
            if service.phase.value == "complete":
                print(f"Published IMU capture: {service.writer.final_path}", flush=True)
            return 0 if service.phase.value == "complete" else 1
        await service.fail("interrupted before capture started")
        return 130
    finally:
        reporter_task.cancel()
        watchdog_task.cancel()
        server.should_exit = True
        with suppress(Exception):
            await server_task
        server_watchdog_task.cancel()
        await asyncio.gather(
            reporter_task,
            watchdog_task,
            server_watchdog_task,
            return_exceptions=True,
        )
        if adb_controller is not None:
            try:
                await asyncio.to_thread(adb_controller.restore)
            except AdbDevicePreparationError as error:
                print(f"Warning: failed to restore Glass3 power setting: {error}", file=sys.stderr)


async def _report(service: ImuCalibrationService, duration: float | None) -> None:
    while True:
        await asyncio.sleep(10)
        status = service.status()
        remaining = (
            "manual"
            if duration is None or status.phase.value != "capturing"
            else f"{max(0.0, duration - status.elapsed_seconds):.0f}s"
        )
        print(
            f"IMU {status.phase.value}: rows={status.rows} "
            f"accel={status.accelerometer_rows} gyro={status.gyroscope_rows} "
            f"rates={status.accelerometer_rate_hz:.1f}/{status.gyroscope_rate_hz:.1f}Hz "
            f"span={status.device_span_seconds:.1f}s gaps={status.sequence_gaps} "
            f"queue={status.queue_size} size={status.bytes_written}B "
            f"remaining={remaining}",
            flush=True,
        )


async def _watch_server(
    server_task: asyncio.Task[bool],
    service: ImuCalibrationService,
) -> None:
    try:
        await server_task
    except Exception as error:
        await service.fail(f"IMU capture gateway failed: {error}")
        return
    if service.phase.value not in {"finalizing", "complete", "failed"}:
        await service.fail("IMU capture gateway stopped unexpectedly")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture raw Glass3 IMU data for calibration")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--duration-seconds", type=_positive_duration)
    mode.add_argument("--until-interrupted", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--discovery-port", type=int, default=DISCOVERY_PORT)
    parser.add_argument("--pairing-token", default=os.environ.get("EGOGLASS_PAIRING_TOKEN"))
    parser.add_argument("--adb-serial")
    parser.add_argument("--disable-discovery", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("local-data/imu-calibration"))
    args = parser.parse_args(argv)
    if args.pairing_token is not None and len(args.pairing_token) < 16:
        parser.error("--pairing-token must contain at least 16 characters")
    try:
        return asyncio.run(_run(args))
    except AdbDevicePreparationError as error:
        print(f"IMU capture preflight failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


def _positive_duration(value: str) -> float:
    duration = float(value)
    if not 0 < duration <= 691_200:
        raise argparse.ArgumentTypeError("duration must be between 0 and 691200 seconds")
    return duration


if __name__ == "__main__":
    raise SystemExit(main())
