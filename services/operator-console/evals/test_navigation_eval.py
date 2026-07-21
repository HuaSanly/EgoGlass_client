from starlette.testclient import TestClient

from egoglass_operator_console.app import create_app


def test_primary_navigation_has_home_storage_and_annotation_destinations() -> None:
    with TestClient(create_app()) as client:
        home_page = client.get("/")
        storage_page = client.get("/storage")
        annotation_page = client.get("/annotations")

    navigation = home_page.text.split('<nav class="nav-rail"', maxsplit=1)[1].split(
        "</nav>", maxsplit=1
    )[0]
    assert navigation.count("<a ") == 3
    assert 'href="/"' in navigation
    assert 'href="/storage"' in navigation
    assert 'href="/annotations"' in navigation
    assert 'aria-current="page"' in navigation
    assert "主页" in navigation
    assert "存储" in navigation
    assert "标注" in navigation
    assert "<button" not in navigation
    assert "data-view" not in navigation
    assert storage_page.status_code == 200
    assert annotation_page.status_code == 200
    storage_navigation = storage_page.text.split(
        '<nav class="nav-rail"', maxsplit=1
    )[1].split("</nav>", maxsplit=1)[0]
    assert '<a class="nav-item is-active" href="/storage" aria-current="page"' in storage_navigation
    annotation_navigation = annotation_page.text.split(
        '<nav class="nav-rail"', maxsplit=1
    )[1].split("</nav>", maxsplit=1)[0]
    assert (
        '<a class="nav-item is-active" href="/annotations" aria-current="page"'
        in annotation_navigation
    )
