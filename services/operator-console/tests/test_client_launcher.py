from pathlib import Path

CLIENT_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = CLIENT_ROOT / "scripts" / "start-client.ps1"
PROCESS_LIFECYCLE = CLIENT_ROOT / "scripts" / "client-process-lifecycle.ps1"


def test_workspace_launcher_starts_discovery_ingest_and_native_desktop() -> None:
    script = LAUNCHER.read_text(encoding="utf-8")

    assert "RandomNumberGenerator" in script
    assert "$env:EGOGLASS_PAIRING_TOKEN = $pairingToken" in script
    assert "--discovery-port" in script
    assert "--hide-pairing-token" in script
    assert "egoglass_ingest_gateway.app" in script
    assert "egoglass_operator_console.desktop" in script
    assert "-WindowStyle Hidden" in script


def test_workspace_launcher_rejects_port_conflicts_and_cleans_process_trees() -> None:
    script = LAUNCHER.read_text(encoding="utf-8")
    lifecycle = PROCESS_LIFECYCLE.read_text(encoding="utf-8")

    assert "Assert-TcpPortAvailable" in script
    assert "Assert-UdpPortAvailable" in script
    assert "client-process-lifecycle.ps1" in script
    assert "New-EgoGlassProcessJob" in script
    assert script.count("Add-ProcessTreeToJob") == 2
    assert "Wait-Process -Id $desktopProcess.Id" in script
    assert "Stop-ClientProcesses" in script
    assert "finally {" in script
    assert "JobObjectLimitKillOnJobClose" in lifecycle
    assert "AssignProcessToJobObject" in lifecycle
    assert "function Stop-ProcessTree" in lifecycle
    assert "function Stop-ClientProcesses" in lifecycle
    assert "pairing-token-123456" not in script


def test_workspace_launcher_uses_ignored_local_recording_directory() -> None:
    script = LAUNCHER.read_text(encoding="utf-8")
    ignore = (CLIENT_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "$recordingsDirectory = Join-Path $repositoryRoot 'local-data\\recordings'" in script
    assert "New-Item -ItemType Directory -Force -Path $recordingsDirectory" in script
    assert "'--recordings-root'," in script
    assert "local-data/" in ignore
