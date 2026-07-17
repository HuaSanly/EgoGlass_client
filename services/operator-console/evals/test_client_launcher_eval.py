from pathlib import Path


def test_one_command_launcher_keeps_pairing_secret_ephemeral() -> None:
    client_root = Path(__file__).resolve().parents[3]
    script = (client_root / "scripts" / "start-client.ps1").read_text(encoding="utf-8")

    assert "New-RuntimePairingToken" in script
    assert "EGOGLASS_PAIRING_TOKEN" in script
    assert "--hide-pairing-token" in script
    assert "Set-Content" not in script
    assert "'--pairing-token'," not in script


def test_one_command_launcher_owns_both_process_lifecycles() -> None:
    client_root = Path(__file__).resolve().parents[3]
    script = (client_root / "scripts" / "start-client.ps1").read_text(encoding="utf-8")

    assert "egoglass_ingest_gateway.app" in script
    assert "egoglass_operator_console.desktop" in script
    assert "Wait-IngestHealth" in script
    assert "Wait-Process" in script
    assert "Stop-ProcessTree" in script


def test_one_command_launcher_keeps_recordings_out_of_git() -> None:
    client_root = Path(__file__).resolve().parents[3]
    script = (client_root / "scripts" / "start-client.ps1").read_text(encoding="utf-8")
    ignore = (client_root / ".gitignore").read_text(encoding="utf-8")

    assert "local-data\\recordings" in script
    assert "--recordings-root" in script
    assert "local-data/" in ignore
