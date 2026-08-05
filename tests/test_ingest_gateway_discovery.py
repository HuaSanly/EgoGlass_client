from __future__ import annotations

import asyncio
import json
import socket

from ui.gateway.discovery import (
    DISCOVERY_MAX_DATAGRAM_BYTES,
    ClientDiscoveryResponse,
    DiscoveryResponder,
    LanDiscoveryService,
)

TOKEN = "discovery-pairing-token-123456"
NONCE = "0123456789abcdef0123456789abcdef"


def request_payload(**overrides: object) -> bytes:
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "message_type": "client_discovery_request",
        "nonce": NONCE,
    }
    payload.update(overrides)
    return json.dumps(payload).encode()


def test_responder_returns_source_routed_signaling_config_and_echoes_nonce() -> None:
    responder = DiscoveryResponder(
        TOKEN,
        8770,
        local_address_resolver=lambda remote: (
            "192.168.3.185" if remote == "192.168.3.45" else "127.0.0.1"
        ),
    )

    payload = responder.respond(request_payload(), "192.168.3.45")

    assert payload is not None
    response = ClientDiscoveryResponse.model_validate_json(payload)
    assert response.nonce == NONCE
    assert response.signaling_url == (
        "http://192.168.3.185:8770/api/v1/webrtc/sessions"
    )
    assert response.pairing_token == TOKEN
    assert TOKEN not in request_payload().decode()


def test_responder_silently_rejects_untrusted_or_malformed_datagrams() -> None:
    responder = DiscoveryResponder(
        TOKEN,
        8770,
        local_address_resolver=lambda _remote: "192.168.3.185",
    )

    assert responder.respond(request_payload(), "8.8.8.8") is None
    assert responder.respond(request_payload(), "169.254.1.2") is None
    assert responder.respond(request_payload(), "192.0.2.10") is None
    assert responder.respond(b"not-json", "192.168.3.45") is None
    assert responder.respond(request_payload(extra="field"), "192.168.3.45") is None
    assert responder.respond(request_payload(nonce="short"), "192.168.3.45") is None
    assert (
        responder.respond(b"x" * (DISCOVERY_MAX_DATAGRAM_BYTES + 1), "192.168.3.45")
        is None
    )

    public_route_responder = DiscoveryResponder(
        TOKEN,
        8770,
        local_address_resolver=lambda _remote: "203.0.113.5",
    )
    assert public_route_responder.respond(request_payload(), "192.168.3.45") is None


def test_udp_service_replies_and_releases_its_port() -> None:
    async def scenario() -> None:
        service = LanDiscoveryService(
            TOKEN,
            8770,
            bind_host="127.0.0.1",
            discovery_port=0,
            local_address_resolver=lambda _remote: "127.0.0.1",
        )
        await service.start()
        port = service.port

        def exchange() -> bytes:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
                client.settimeout(2)
                client.sendto(request_payload(), ("127.0.0.1", port))
                return client.recvfrom(4096)[0]

        response_payload = await asyncio.to_thread(exchange)
        assert ClientDiscoveryResponse.model_validate_json(response_payload).nonce == NONCE
        await service.close()

        replacement = LanDiscoveryService(
            TOKEN,
            8770,
            bind_host="127.0.0.1",
            discovery_port=port,
            local_address_resolver=lambda _remote: "127.0.0.1",
        )
        await replacement.start()
        await replacement.close()

    asyncio.run(scenario())
