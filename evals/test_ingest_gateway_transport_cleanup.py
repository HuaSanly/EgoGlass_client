from pathlib import Path

import ui.gateway as gateway
from ui.gateway.app import create_app


def test_retired_platform_fallback_is_absent_from_the_ingest_service() -> None:
    package = Path(gateway.__path__[0])
    source_root = package.parent
    app_source = (package / "app.py").read_text(encoding="utf-8")
    service_root = source_root.parent
    readme = (service_root / "README.md").read_text(encoding="utf-8")
    project = (service_root / "pyproject.toml").read_text(encoding="utf-8")

    assert not (package / "adapters" / "rtsp.py").exists()
    assert not (package / "models.py").exists()
    assert not (package / "runtime.py").exists()
    assert "/api/v1/rtsp/probe" not in app_source
    assert "GB28181" not in readme
    assert "RTSP" not in readme
    assert "RTSP" not in project
    assert all(route.path != "/api/v1/rtsp/probe" for route in create_app().routes)
