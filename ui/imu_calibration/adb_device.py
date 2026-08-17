from __future__ import annotations

import ipaddress
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ui.gateway.discovery import local_ipv4_for_remote

GLASSES_PACKAGE = "com.egoglass.glasses"
_ADB_SERIAL_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_WLAN_ADDRESS_PATTERN = re.compile(r"\binet\s+(\d+(?:\.\d+){3}/\d+)\b")


class AdbDevicePreparationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AdbNetworkStatus:
    device_address: ipaddress.IPv4Address
    device_network: ipaddress.IPv4Network
    client_address: ipaddress.IPv4Address


CommandRunner = Callable[[Sequence[str]], str]
RouteResolver = Callable[[str], str]


class AdbGlassController:
    """Prepare one explicitly selected Glass3 for a long headless capture."""

    def __init__(
        self,
        serial: str,
        *,
        command_runner: CommandRunner | None = None,
        route_resolver: RouteResolver = local_ipv4_for_remote,
    ) -> None:
        if not _ADB_SERIAL_PATTERN.fullmatch(serial):
            raise ValueError("ADB serial contains unsupported characters")
        self.serial = serial
        self._command_runner = command_runner or _run_command
        self._route_resolver = route_resolver
        self._previous_stay_awake: str | None = None
        self._prepared = False

    def preflight(self) -> AdbNetworkStatus:
        output = self._adb("shell", "ip", "-4", "-o", "addr", "show", "dev", "wlan0")
        match = _WLAN_ADDRESS_PATTERN.search(output)
        if match is None:
            raise AdbDevicePreparationError(
                f"ADB device {self.serial} has no active wlan0 IPv4 address"
            )
        device_interface = ipaddress.ip_interface(match.group(1))
        if not isinstance(device_interface, ipaddress.IPv4Interface):
            raise AdbDevicePreparationError("Glass3 discovery requires IPv4")
        try:
            client_address = ipaddress.ip_address(self._route_resolver(str(device_interface.ip)))
        except (OSError, ValueError) as error:
            raise AdbDevicePreparationError(
                f"cannot resolve the client route to Glass3 at {device_interface.ip}"
            ) from error
        if not isinstance(client_address, ipaddress.IPv4Address):
            raise AdbDevicePreparationError("Glass3 discovery requires an IPv4 client route")
        if client_address not in device_interface.network:
            raise AdbDevicePreparationError(
                "Glass3 and client are not on the same IPv4 subnet: "
                f"glasses={device_interface.ip}/{device_interface.network.prefixlen}, "
                f"client={client_address}. Connect both to the same Wi-Fi network."
            )
        return AdbNetworkStatus(
            device_address=device_interface.ip,
            device_network=device_interface.network,
            client_address=client_address,
        )

    def prepare_and_launch(self) -> None:
        if self._prepared:
            return
        previous = self._adb(
            "shell", "settings", "get", "global", "stay_on_while_plugged_in"
        ).strip()
        if previous not in {"", "null"} and not previous.isdigit():
            raise AdbDevicePreparationError(
                f"unexpected stay_on_while_plugged_in value: {previous!r}"
            )
        self._previous_stay_awake = None if previous in {"", "null"} else previous
        self._prepared = True
        try:
            self._adb(
                "shell",
                "settings",
                "put",
                "global",
                "stay_on_while_plugged_in",
                "2",
            )
            self._adb("shell", "input", "keyevent", "224")
            self._adb("shell", "wm", "dismiss-keyguard")
            self._adb("shell", "am", "force-stop", GLASSES_PACKAGE)
            self._adb(
                "shell",
                "am",
                "start",
                "-W",
                "-a",
                "android.intent.action.MAIN",
                "-c",
                "android.intent.category.LAUNCHER",
                "-n",
                f"{GLASSES_PACKAGE}/.MainActivity",
            )
        except Exception:
            self.restore()
            raise

    def restore(self) -> None:
        if not self._prepared:
            return
        previous, self._prepared = self._previous_stay_awake, False
        self._previous_stay_awake = None
        if previous is None:
            self._adb("shell", "settings", "delete", "global", "stay_on_while_plugged_in")
        else:
            self._adb(
                "shell",
                "settings",
                "put",
                "global",
                "stay_on_while_plugged_in",
                previous,
            )

    def _adb(self, *args: str) -> str:
        try:
            return self._command_runner(("adb", "-s", self.serial, *args))
        except (OSError, subprocess.SubprocessError) as error:
            raise AdbDevicePreparationError(
                f"ADB command failed for device {self.serial}: {' '.join(args)}"
            ) from error


def _run_command(command: Sequence[str]) -> str:
    completed = subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    return completed.stdout
