from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def test_pyinstaller_spec_resolves_all_packaged_inputs() -> None:
    service_root = Path(__file__).parent.parent
    spec_path = service_root / "packaging" / "egoglass-client.spec"
    captured: dict[str, Any] = {}

    def analysis(scripts: list[str], **kwargs: Any) -> SimpleNamespace:
        captured["scripts"] = scripts
        captured["datas"] = kwargs["datas"]
        return SimpleNamespace(pure=[], scripts=[], binaries=[], datas=[])

    def executable(*args: Any, **kwargs: Any) -> object:
        captured["version"] = kwargs["version"]
        return object()

    runpy.run_path(
        spec_path,
        init_globals={
            "SPECPATH": str(spec_path.parent),
            "Analysis": analysis,
            "PYZ": lambda *args, **kwargs: object(),
            "EXE": executable,
            "COLLECT": lambda *args, **kwargs: object(),
        },
    )

    assert Path(captured["scripts"][0]).is_file()
    assert Path(captured["datas"][0][0]).name == "THIRD_PARTY_NOTICES.txt"
    assert Path(captured["datas"][0][0]).is_file()
    assert Path(captured["version"]).is_file()
