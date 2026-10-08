"""Process-wide egress block for on-prem mode.

While the flag is on, a connection is allowed only to loopback, to a Unix socket or to an explicitly allowed host.
Everything else (including DNS lookups for outside names) raises `EgressBlocked`. Only host and port are ever logged.
"""
from __future__ import annotations

import asyncio.base_events
import asyncio.selector_events
import ipaddress
import logging
import socket
import sys
import threading
from collections import deque
from collections.abc import Callable, Iterable
from typing import Any

log = logging.getLogger("evora.privacy")

_LOCAL_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}
_UNSPECIFIED = {"", "0.0.0.0", "::"}  # connecting to these reaches the local machine


class EgressBlocked(OSError):
    """On-prem mode refused a connection or lookup that would leave this machine."""


def is_local(host: str | bytes | None) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    host = host.strip().strip("[]").split("%", 1)[0].lower()
    if host in _LOCAL_NAMES or host in _UNSPECIFIED:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False  # a hostname other than localhost: resolving it would leave the machine
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback


class PrivacyGuard:
    def __init__(self) -> None:
        self.active: Callable[[], bool] = lambda: False
        self.allow_hosts: set[str] = set()
        self.blocked = 0
        self.attempts: deque[tuple[str, str, int | None, bool]] = deque(maxlen=200)  # (kind, host, port, allowed)
        self._installed = False
        self._lock = threading.Lock()
        self._orig: dict[str, Any] = {}
        self._had_own: dict[str, bool] = {}
        self._loop_orig: list[tuple[type, str, bool, Any]] = []

    # --- policy ---
    def _allowed(self, host: str | bytes | None) -> bool:
        if is_local(host):
            return True
        name = host.decode("ascii", "ignore") if isinstance(host, bytes) else str(host)
        return name.strip().strip("[]").lower() in self.allow_hosts

    def check(self, kind: str, host: str | bytes | None, port: int | None) -> None:
        allowed = self._allowed(host)
        shown = host.decode("ascii", "ignore") if isinstance(host, bytes) else str(host)
        self.attempts.append((kind, shown, port, allowed))  # host and port only; kept so tests and the demo can show it
        if allowed or not self.active():
            return
        with self._lock:
            self.blocked += 1
        log.warning("on-prem mode blocked an outbound %s to %s:%s", kind, shown, port)
        raise EgressBlocked(f"on-prem mode: connecting to {shown} is not allowed")

    def _check_address(self, family: int, address: Any, kind: str) -> None:
        if family == getattr(socket, "AF_UNIX", -1) or not isinstance(address, tuple) or not address:
            return  # Unix sockets stay on this machine
        host = address[0]
        port = address[1] if len(address) > 1 and isinstance(address[1], int) else None
        self.check(kind, host, port)

    # --- install ---
    def install(self, active: Callable[[], bool], allow_hosts: Iterable[str] = ()) -> None:
        self.active = active
        self.allow_hosts = {h.strip().lower() for h in allow_hosts if h.strip()}
        if self._installed:
            return
        guard = self
        sock_cls = socket.socket
        for name in ("connect", "connect_ex"):
            self._had_own[name] = name in sock_cls.__dict__
            original = getattr(sock_cls, name)
            self._orig[name] = original

            def make(name: str, original: Any):
                def patched(sock: socket.socket, address: Any, *args: Any) -> Any:
                    guard._check_address(sock.family, address, "connection")
                    return guard._orig[name](sock, address, *args)

                patched.__name__ = name
                return patched

            setattr(sock_cls, name, make(name, original))
        for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex"):
            self._orig[name] = getattr(socket, name)

            def make_lookup(name: str):
                def patched(host: Any, *args: Any, **kwargs: Any) -> Any:
                    guard.check("DNS lookup", host, None)
                    return guard._orig[name](host, *args, **kwargs)

                patched.__name__ = name
                return patched

            setattr(socket, name, make_lookup(name))
        self._patch_sendto(sock_cls)
        self._patch_event_loops()
        self._installed = True

    def _patch_sendto(self, sock_cls: type) -> None:
        guard = self
        self._had_own["sendto"] = "sendto" in sock_cls.__dict__
        self._orig["sendto"] = sock_cls.sendto

        def sendto(sock: socket.socket, data: Any, *args: Any) -> Any:  # UDP without connect(): DNS, QUIC, ...
            address = args[-1] if args else None
            guard._check_address(sock.family, address, "datagram")
            return guard._orig["sendto"](sock, data, *args)

        sock_cls.sendto = sendto  # type: ignore[method-assign]

    def _patch_event_loops(self) -> None:
        """asyncio's Windows proactor loop connects without calling socket.connect, so the loops are patched too."""
        guard = self

        def wrap_create_connection(orig: Any):
            async def create_connection(loop: Any, protocol_factory: Any, host: Any = None, port: Any = None, **kw: Any) -> Any:
                if host is not None:
                    guard.check("connection", host, port if isinstance(port, int) else None)
                return await orig(loop, protocol_factory, host, port, **kw)

            return create_connection

        def wrap_sock_connect(orig: Any):
            async def sock_connect(loop: Any, sock: socket.socket, address: Any) -> Any:
                guard._check_address(sock.family, address, "connection")
                return await orig(loop, sock, address)

            return sock_connect

        def wrap_datagram(orig: Any):
            async def create_datagram_endpoint(
                loop: Any, protocol_factory: Any, local_addr: Any = None, remote_addr: Any = None, **kw: Any,
            ) -> Any:
                if remote_addr:
                    guard.check("datagram", remote_addr[0], remote_addr[1] if len(remote_addr) > 1 else None)
                return await orig(loop, protocol_factory, local_addr, remote_addr, **kw)

            return create_datagram_endpoint

        targets: list[tuple[type, str, Any]] = [
            (asyncio.base_events.BaseEventLoop, "create_connection", wrap_create_connection),
            (asyncio.base_events.BaseEventLoop, "create_datagram_endpoint", wrap_datagram),
            (asyncio.selector_events.BaseSelectorEventLoop, "sock_connect", wrap_sock_connect),
        ]
        if sys.platform == "win32":
            from asyncio import proactor_events

            targets.append((proactor_events.BaseProactorEventLoop, "sock_connect", wrap_sock_connect))
        for cls, name, wrapper in targets:
            original = getattr(cls, name)
            self._loop_orig.append((cls, name, name in cls.__dict__, original))
            setattr(cls, name, wrapper(original))

    def uninstall(self) -> None:
        if not self._installed:
            return
        for name in ("connect", "connect_ex"):
            if self._had_own[name]:
                setattr(socket.socket, name, self._orig[name])
            else:
                delattr(socket.socket, name)
        for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex"):
            setattr(socket, name, self._orig[name])
        if self._had_own.get("sendto"):
            socket.socket.sendto = self._orig["sendto"]  # type: ignore[method-assign]
        else:
            del socket.socket.sendto
        for cls, name, had_own, original in reversed(self._loop_orig):
            if had_own:
                setattr(cls, name, original)
            else:
                delattr(cls, name)
        self._loop_orig.clear()
        self._installed = False
        self.active = lambda: False

    @property
    def installed(self) -> bool:
        return self._installed


guard = PrivacyGuard()
