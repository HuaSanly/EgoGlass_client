from starlette.testclient import TestClient

from egoglass_operator_console.app import create_app
from egoglass_operator_console.runtime import ConsoleRuntime


def test_primary_navigation_has_one_real_page_destination() -> None:
    with TestClient(create_app(ConsoleRuntime())) as client:
        page = client.get("/")

    navigation = page.text.split('<nav class="nav-rail"', maxsplit=1)[1].split(
        "</nav>", maxsplit=1
    )[0]
    assert navigation.count("<a ") == 1
    assert 'href="/"' in navigation
    assert 'aria-current="page"' in navigation
    assert "主页" in navigation
    assert "<button" not in navigation
    assert "data-view" not in navigation
