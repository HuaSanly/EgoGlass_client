import json

from egoglass_ingest_gateway.discovery import ClientDiscoveryResponse, DiscoveryResponder


def test_discovery_response_uses_runtime_secret_and_request_nonce() -> None:
    token_a = "eval-runtime-token-a-123456"
    token_b = "eval-runtime-token-b-123456"
    nonce = "abcdef0123456789abcdef0123456789"
    request = json.dumps(
        {
            "schema_version": "1.0",
            "message_type": "client_discovery_request",
            "nonce": nonce,
        }
    ).encode()

    responses = [
        ClientDiscoveryResponse.model_validate_json(
            DiscoveryResponder(
                token,
                8770,
                local_address_resolver=lambda _remote: "192.168.3.185",
            ).respond(request, "192.168.3.45")
        )
        for token in (token_a, token_b)
    ]

    assert [response.nonce for response in responses] == [nonce, nonce]
    assert [response.pairing_token for response in responses] == [token_a, token_b]
    assert responses[0].pairing_token != responses[1].pairing_token
