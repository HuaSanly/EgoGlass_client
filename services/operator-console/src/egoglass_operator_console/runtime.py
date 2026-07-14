from __future__ import annotations

import asyncio
import math
import time

from .models import (
    CalibrationState,
    CalibrationSummary,
    ConsoleState,
    HandSide,
    HandTrajectory,
    RuntimeMetrics,
    RuntimeSettings,
    SessionPhase,
    TelemetrySnapshot,
    Waypoint3D,
)


class ConsoleRuntime:
    """Owns operator-visible state and a deterministic synthetic source."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._settings = RuntimeSettings()
        self._settings_revision = 1
        self._session_id = "simulation-session-001"
        self._session_phase = SessionPhase.LIVE
        self._recording = False
        self._calibration = CalibrationSummary(
            profile_id="simulation-calibration-001",
            state=CalibrationState.SIMULATED,
            reprojection_error_px=0.31,
        )

    async def state(self) -> ConsoleState:
        async with self._lock:
            return self._state_unlocked()

    async def update_settings(self, settings: RuntimeSettings) -> ConsoleState:
        async with self._lock:
            self._settings = settings
            self._settings_revision += 1
            return self._state_unlocked()

    async def set_session_active(self, active: bool) -> ConsoleState:
        async with self._lock:
            self._session_phase = SessionPhase.LIVE if active else SessionPhase.IDLE
            if not active:
                self._recording = False
            return self._state_unlocked()

    async def set_recording(self, active: bool) -> ConsoleState:
        async with self._lock:
            if active and self._session_phase is not SessionPhase.LIVE:
                raise RuntimeError("recording requires a live session")
            self._recording = active
            return self._state_unlocked()

    async def telemetry(self, tick: int, *, now_unix_ns: int | None = None) -> TelemetrySnapshot:
        async with self._lock:
            settings = self._settings.model_copy(deep=True)
            phase = self._session_phase
            recording = self._recording
            calibration = self._calibration.model_copy(deep=True)

        return build_telemetry(
            tick=tick,
            settings=settings,
            session_id=self._session_id,
            session_phase=phase,
            recording=recording,
            calibration=calibration,
            now_unix_ns=now_unix_ns,
        )

    def _state_unlocked(self) -> ConsoleState:
        return ConsoleState(
            service_version="0.1.0",
            session_id=self._session_id,
            session_phase=self._session_phase,
            recording=self._recording,
            settings_revision=self._settings_revision,
            settings=self._settings,
            calibration=self._calibration,
        )


def build_telemetry(
    *,
    tick: int,
    settings: RuntimeSettings,
    session_id: str,
    session_phase: SessionPhase,
    recording: bool,
    calibration: CalibrationSummary,
    now_unix_ns: int | None = None,
) -> TelemetrySnapshot:
    """Produce repeatable telemetry for the same tick and settings."""

    phase = tick / max(settings.inference_fps, 1)
    hands = [
        _trajectory(HandSide.LEFT, phase, settings),
        _trajectory(HandSide.RIGHT, phase, settings),
    ]
    media_latency = 41.0 + 4.0 * (1.0 + math.sin(phase * 0.47))
    inference_latency = 54.0 + 6.0 * (1.0 + math.sin(phase * 0.31 + 0.7))
    feedback_latency = media_latency + inference_latency + 19.0

    return TelemetrySnapshot(
        session_id=session_id,
        session_phase=session_phase,
        recording=recording,
        frame_seq=round(tick * settings.capture_fps / settings.inference_fps),
        captured_at_sdk_ms=round(tick * 1000 / settings.inference_fps),
        received_at_perf_counter_ns=round(tick * 1_000_000_000 / settings.inference_fps),
        generated_at_unix_ns=now_unix_ns if now_unix_ns is not None else time.time_ns(),
        max_feedback_age_ms=settings.max_feedback_age_ms,
        calibration=calibration,
        metrics=RuntimeMetrics(
            capture_fps=float(settings.capture_fps),
            inference_fps=float(settings.inference_fps),
            media_latency_ms=round(media_latency, 1),
            inference_latency_ms=round(inference_latency, 1),
            feedback_latency_ms=round(feedback_latency, 1),
            dropped_frames=tick // 240,
            queue_depth=0,
            gpu_memory_gb=0.0,
        ),
        hands=hands,
    )


def _trajectory(side: HandSide, phase: float, settings: RuntimeSettings) -> HandTrajectory:
    direction = -1.0 if side is HandSide.LEFT else 1.0
    lateral_origin = 0.16 * direction
    points: list[Waypoint3D] = []

    for index in range(settings.prediction_steps):
        progress = index / max(settings.prediction_steps - 1, 1)
        hand_phase = phase * 0.72 + progress * 1.25 + (0.4 if side is HandSide.RIGHT else 0)
        x_m = lateral_origin + direction * 0.065 * progress + 0.015 * math.sin(hand_phase)
        y_m = 0.10 - 0.12 * progress + 0.018 * math.cos(hand_phase * 1.3)
        z_m = 0.72 - 0.16 * progress + 0.025 * math.sin(hand_phase * 0.8)
        points.append(
            Waypoint3D(
                t_offset_ms=index * settings.prediction_interval_ms,
                x_m=round(x_m, 5),
                y_m=round(y_m, 5),
                z_m=round(z_m, 5),
                confidence=round(0.96 - 0.025 * index, 3),
            )
        )

    return HandTrajectory(
        side=side,
        present=True,
        confidence=0.96,
        waypoints=points,
    )
