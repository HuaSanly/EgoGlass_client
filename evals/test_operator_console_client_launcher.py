from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path

import pytest

CLIENT_ROOT = Path(__file__).resolve().parents[1]
PROCESS_LIFECYCLE = CLIENT_ROOT / "scripts" / "client-process-lifecycle.ps1"


def test_one_command_launcher_keeps_pairing_secret_ephemeral() -> None:
    script = (CLIENT_ROOT / "scripts" / "start-client.ps1").read_text(encoding="utf-8")

    assert "New-RuntimePairingToken" in script
    assert "EGOGLASS_PAIRING_TOKEN" in script
    assert "--hide-pairing-token" in script
    assert "Set-Content" not in script
    assert "'--pairing-token'," not in script


def test_one_command_launcher_owns_all_process_lifecycles() -> None:
    script = (CLIENT_ROOT / "scripts" / "start-client.ps1").read_text(encoding="utf-8")
    lifecycle = PROCESS_LIFECYCLE.read_text(encoding="utf-8")

    assert "ingest_gateway.app" in script
    assert "data_platform" not in script
    assert "DataPort" not in script
    assert "8780" not in script
    assert "operator_console.desktop" in script
    assert ".venv\\Scripts\\python.exe" in script
    assert "services\\" not in script
    assert "Wait-DataPlatformHealth" not in script
    assert "EGOGLASS_DATA_PLATFORM_ORIGIN" not in script
    assert "EGOGLASS_RECORDINGS_ROOT" in script
    assert "Wait-IngestHealth" in script
    assert "Wait-Process" in script
    assert "Stop-ClientProcesses" in script
    assert "Complete-EgoGlassCaptureSession" in script
    assert "api/v1/recordings/session-commands" in lifecycle
    assert script.index("Complete-EgoGlassCaptureSession") < script.index(
        "Stop-ClientProcesses"
    )
    assert "JobObjectLimitKillOnJobClose" in lifecycle


def test_one_command_launcher_keeps_recordings_out_of_git() -> None:
    script = (CLIENT_ROOT / "scripts" / "start-client.ps1").read_text(encoding="utf-8")
    ignore = (CLIENT_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "local-data\\recordings" in script
    assert "--recordings-root" in script
    assert "local-data/" in ignore


def _reserve_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_port(port: int, *, listening: bool, timeout_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(0.1)
            is_listening = client.connect_ex(("127.0.0.1", port)) == 0
        if is_listening == listening:
            return
        time.sleep(0.05)
    state = "listen" if listening else "close"
    raise AssertionError(f"port {port} did not {state} before timeout")


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object")
def test_process_job_releases_descendant_port_when_launcher_is_terminated(
    tmp_path: Path,
) -> None:
    port = _reserve_tcp_port()
    child_script = tmp_path / "child.ps1"
    owner_script = tmp_path / "owner.ps1"
    child_pid_path = tmp_path / "child.pid"
    child_script.write_text(
        "\n".join(
            (
                "param([int] $Port)",
                "$listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, $Port)",
                "$listener.Start()",
                "try { while ($true) { Start-Sleep -Seconds 1 } } finally { $listener.Stop() }",
            )
        ),
        encoding="utf-8-sig",
    )
    lifecycle_path = str(PROCESS_LIFECYCLE).replace("'", "''")
    child_path = str(child_script).replace("'", "''")
    child_pid_file = str(child_pid_path).replace("'", "''")
    owner_script.write_text(
        "\n".join(
            (
                "$ErrorActionPreference = 'Stop'",
                f". '{lifecycle_path}'",
                "$job = New-EgoGlassProcessJob",
                "$child = Start-Process powershell.exe -ArgumentList @("
                f"'-NoProfile', '-File', '{child_path}', '-Port', '{port}'"
                ") -WindowStyle Hidden -PassThru",
                f"$child.Id | Set-Content -LiteralPath '{child_pid_file}' -Encoding ascii",
                "Add-ProcessTreeToJob -Job $job -ProcessId $child.Id",
                "Wait-Process -Id $child.Id",
            )
        ),
        encoding="utf-8-sig",
    )

    owner = subprocess.Popen(
        ["powershell.exe", "-NoProfile", "-File", str(owner_script)],
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        _wait_for_port(port, listening=True)
        owner.kill()
        owner.wait(timeout=5)
        _wait_for_port(port, listening=False)
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)
        if child_pid_path.exists():
            child_pid = child_pid_path.read_text(encoding="ascii").strip()
            subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-Command",
                    f"Stop-Process -Id {int(child_pid)} -Force -ErrorAction SilentlyContinue",
                ],
                check=False,
            )
