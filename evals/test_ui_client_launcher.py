from pathlib import Path

CLIENT_ROOT = Path(__file__).resolve().parents[1]


def test_one_command_launcher_has_one_application_process_owner() -> None:
    script = (CLIENT_ROOT / "scripts" / "start-client.ps1").read_text(encoding="utf-8")

    assert script.count("& $workspacePython -m ui") == 1
    assert "Start-Process" not in script
    assert "Wait-Process" not in script
    assert "Stop-Process" not in script
    assert "ui.gateway.app" not in script
    assert "operator_console" not in script


def test_one_command_launcher_keeps_pairing_secret_process_local() -> None:
    script = (CLIENT_ROOT / "scripts" / "start-client.ps1").read_text(encoding="utf-8")
    runtime = (CLIENT_ROOT / "ui" / "application" / "runtime_host.py").read_text(
        encoding="utf-8"
    )

    assert "EGOGLASS_PAIRING_TOKEN" not in script
    assert "Set-Content" not in script
    assert "secrets.token_urlsafe(24)" in runtime


def test_one_command_launcher_keeps_recordings_out_of_git() -> None:
    script = (CLIENT_ROOT / "scripts" / "start-client.ps1").read_text(encoding="utf-8")
    ignore = (CLIENT_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "local-data\\recordings" in script
    assert "--recordings-root" in script
    assert "local-data/" in ignore
