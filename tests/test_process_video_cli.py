from __future__ import annotations

import ast
from pathlib import Path

from process_video import build_parser


def test_headless_entrypoint_has_no_ui_or_gateway_imports() -> None:
    source_path = Path(__file__).parents[1] / "src" / "process_video.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    assert not any(
        name == "ui" or name.startswith("ui.") or name.startswith("ui.gateway")
        for name in imported
    )
    assert not any(name in {"PyQt6", "qfluentwidgets", "aiortc"} for name in imported)


def test_headless_parser_requires_session_and_output() -> None:
    parser = build_parser()
    args = parser.parse_args(["--session", "session", "--output", "run"])

    assert args.session == Path("session")
    assert args.output == Path("run")
    assert args.clip_id is None
