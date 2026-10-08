import socket

import pytest

from evora.core import privacy_guard


@pytest.fixture(autouse=True)
def _clean_guard():
    """These tests install and remove the guard themselves; start and end with the real sockets restored."""
    privacy_guard.guard.uninstall()
    privacy_guard.guard.blocked = 0
    privacy_guard.guard.attempts.clear()
    yield
    privacy_guard.guard.uninstall()
    privacy_guard.guard.blocked = 0
    privacy_guard.guard.attempts.clear()


@pytest.fixture()
def fresh_guard():
    """A private guard whose outside connections and lookups are recorders, so nothing touches the network.

    Loopback connections stay real: asyncio needs a local socketpair just to start an event loop on Windows.
    """
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    g = privacy_guard.PrivacyGuard()
    flag = {"on": True}
    g.install(lambda: flag["on"])
    reached: list = []

    def record(real):
        def connect(sock, address, *a):
            reached.append(address)  # every connection that got past the guard, real or not
            if isinstance(address, tuple) and privacy_guard.is_local(address[0]) and address[0] not in ("", "0.0.0.0"):
                return real(sock, address, *a)  # loopback is genuinely connected (a local server may or may not be there)
            return None if real is real_connect else 0

        return connect

    g._orig["connect"] = record(real_connect)
    g._orig["connect_ex"] = record(real_connect_ex)
    answer = [(socket.AF_INET, 1, 6, "", ("127.0.0.1", 0))]
    g._orig["getaddrinfo"] = lambda host, *a, **k: reached.append(("lookup", host)) or answer
    g._orig["gethostbyname"] = lambda host: reached.append(("lookup", host)) or "127.0.0.1"
    g._orig["gethostbyname_ex"] = lambda host: reached.append(("lookup", host)) or (host, [], ["127.0.0.1"])
    g.flag, g.reached = flag, reached
    yield g
    g.uninstall()
