from pathlib import Path

CLIENT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = CLIENT_ROOT / "scripts" / "start-client.ps1"


def test_workspace_launcher_runs_one_unified_native_process() -> None:
    script = LAUNCHER.read_text(encoding="utf-8")

    assert "conda info --base" in script
    assert "envs\\$EnvironmentName\\python.exe" in script
    assert "& $workspacePython -m ui" in script
    assert "--discovery-port" in script
    assert "--recordings-root" in script
    assert "Start-Process" not in script
    assert "ui.gateway.app" not in script
    assert "operator_console" not in script
    assert "client-process-lifecycle" not in script


def test_workspace_launcher_rejects_conflicting_runtime_ports() -> None:
    script = LAUNCHER.read_text(encoding="utf-8")

    assert "Assert-TcpPortAvailable" in script
    assert "Assert-UdpPortAvailable" in script
    assert "TCP port $Port is already in use" in script
    assert "UDP port $Port is already in use" in script


def test_workspace_launcher_uses_ignored_local_recording_directory() -> None:
    script = LAUNCHER.read_text(encoding="utf-8")
    ignore = (CLIENT_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "$recordingsDirectory = Join-Path $repositoryRoot 'local-data\\recordings'" in script
    assert "New-Item -ItemType Directory -Force -Path $recordingsDirectory" in script
    assert "local-data/" in ignore
