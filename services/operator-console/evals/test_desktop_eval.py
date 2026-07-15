import http.cookiejar
import socket
import urllib.error
import urllib.request

from egoglass_operator_console.desktop import LocalConsoleServer, build_desktop_url


def test_repeated_desktop_server_lifecycle_is_private_and_recoverable() -> None:
    ports: list[int] = []

    for iteration in range(3):
        token = f"desktop-eval-token-{iteration}"
        server = LocalConsoleServer(token)
        server.start()
        ports.append(server.port)
        try:
            try:
                urllib.request.urlopen(server.origin, timeout=5)
            except urllib.error.HTTPError as error:
                assert error.code == 401
            else:
                raise AssertionError("desktop UI accepted a request without its session token")

            cookie_jar = http.cookiejar.CookieJar()
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
            with opener.open(build_desktop_url(server.origin, token), timeout=5) as response:
                assert response.status == 200
            with opener.open(f"{server.origin}/assets/app.js", timeout=5) as response:
                script = response.read()
                assert b"frame.jpg" in script
                assert b"WebSocket" not in script
        finally:
            server.stop()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(1)
            assert client.connect_ex(("127.0.0.1", server.port)) != 0

    assert all(port > 0 for port in ports)
