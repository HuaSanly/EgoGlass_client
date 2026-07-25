from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Callable
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

DISCOVERY_PORT = 8771
DISCOVERY_MAX_DATAGRAM_BYTES = 2048
WEBRTC_SIGNALING_PATH = "/api/v1/webrtc/sessions"
TRUSTED_DISCOVERY_NETWORKS = tuple(
    ipaddress.ip_network(cidr)
    for cidr in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8")
)


class ClientDiscoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    message_type: Literal["client_discovery_request"] = "client_discovery_request"
    nonce: str = Field(min_length=32, max_length=32, pattern=r"^[a-f0-9]{32}$")


class ClientDiscoveryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    message_type: Literal["client_discovery_response"] = "client_discovery_response"
    nonce: str = Field(min_length=32, max_length=32, pattern=r"^[a-f0-9]{32}$")
    signaling_url: str = Field(min_length=16, max_length=2048)
    pairing_token: str = Field(min_length=16, max_length=256)


def is_trusted_discovery_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return isinstance(address, ipaddress.IPv4Address) and any(
        address in network for network in TRUSTED_DISCOVERY_NETWORKS
    )


def local_ipv4_for_remote(remote_host: str) -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route_socket:
        route_socket.connect((remote_host, DISCOVERY_PORT))
        return str(route_socket.getsockname()[0])


class DiscoveryResponder:
    def __init__(
        self,
        pairing_token: str,
        gateway_port: int,
        *,
        local_address_resolver: Callable[[str], str] = local_ipv4_for_remote,
    ) -> None:
        if len(pairing_token) < 16:
            raise ValueError("pairing_token must contain at least 16 characters")
        if gateway_port not in range(1, 65_536):
            raise ValueError("gateway_port must be a valid TCP port")
        self._pairing_token = pairing_token
        self._gateway_port = gateway_port
        self._local_address_resolver = local_address_resolver

    def respond(self, payload: bytes, remote_host: str) -> bytes | None:
        if len(payload) > DISCOVERY_MAX_DATAGRAM_BYTES:
            return None
        try:
            remote_address = ipaddress.ip_address(remote_host)
            if not is_trusted_discovery_address(remote_address):
                return None
            request = ClientDiscoveryRequest.model_validate_json(payload)
            local_host = self._local_address_resolver(remote_host)
            local_address = ipaddress.ip_address(local_host)
            if not is_trusted_discovery_address(local_address):
                return None
        except (OSError, ValueError, ValidationError):
            return None

        response = ClientDiscoveryResponse(
            nonce=request.nonce,
            signaling_url=(
                f"http://{local_address}:{self._gateway_port}{WEBRTC_SIGNALING_PATH}"
            ),
            pairing_token=self._pairing_token,
        )
        return response.model_dump_json().encode("utf-8")


class _DiscoveryDatagramProtocol(asyncio.DatagramProtocol):
    def __init__(
        self,
        responder: DiscoveryResponder,
        closed: asyncio.Future[None],
    ) -> None:
        self._responder = responder
        self._closed = closed
        self._transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self._transport = cast(asyncio.DatagramTransport, transport)

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        response = self._responder.respond(data, addr[0])
        if response is not None and self._transport is not None:
            self._transport.sendto(response, addr)

    def connection_lost(self, exc: Exception | None) -> None:
        if not self._closed.done():
            self._closed.set_result(None)


class LanDiscoveryService:
    def __init__(
        self,
        pairing_token: str,
        gateway_port: int,
        *,
        bind_host: str = "0.0.0.0",
        discovery_port: int = DISCOVERY_PORT,
        local_address_resolver: Callable[[str], str] = local_ipv4_for_remote,
    ) -> None:
        if discovery_port not in range(0, 65_536):
            raise ValueError("discovery_port must be a valid UDP port")
        self._responder = DiscoveryResponder(
            pairing_token,
            gateway_port,
            local_address_resolver=local_address_resolver,
        )
        self._bind_host = bind_host
        self._discovery_port = discovery_port
        self._transport: asyncio.DatagramTransport | None = None
        self._closed: asyncio.Future[None] | None = None

    @property
    def port(self) -> int:
        if self._transport is None:
            return self._discovery_port
        return int(self._transport.get_extra_info("sockname")[1])

    async def start(self) -> None:
        if self._transport is not None:
            return
        loop = asyncio.get_running_loop()
        closed = loop.create_future()
        transport, _protocol = await loop.create_datagram_endpoint(
            lambda: _DiscoveryDatagramProtocol(self._responder, closed),
            local_addr=(self._bind_host, self._discovery_port),
            allow_broadcast=True,
        )
        self._transport = cast(asyncio.DatagramTransport, transport)
        self._closed = closed

    async def close(self) -> None:
        if self._transport is not None:
            transport, self._transport = self._transport, None
            closed, self._closed = self._closed, None
            transport.close()
            if closed is not None:
                await closed
