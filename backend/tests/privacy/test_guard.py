import asyncio
import logging
import socket
import threading

import httpx
import pytest

from evora.core import privacy_guard
from evora.core.privacy_guard import EgressBlocked, PrivacyGuard, is_local


@pytest.mark.parametrize(
    "host,local",
    [
        ("127.0.0.1", True), ("127.5.5.5", True), ("::1", True), ("localhost", True), ("LOCALHOST", True),
        ("[::1]", True), ("::ffff:127.0.0.1", True), ("0.0.0.0", True), ("", True), (None, True), (b"127.0.0.1", True),
        ("8.8.8.8", False), ("93.184.216.34", False), ("2606:4700::1111", False), ("api.groq.com", False),
        ("10.0.0.5", False), ("192.168.1.10", False), ("localhost.evil.com", False), ("::ffff:8.8.8.8", False),
    ],
)
def test_what_counts_as_local(host, local):
    assert is_local(host) is local


def connect(host, port=443, family=socket.AF_INET):
    s = socket.socket(family, socket.SOCK_STREAM)
    try:
        return s.connect((host, port))
    finally:
        s.close()


@pytest.mark.parametrize("host", ["8.8.8.8", "93.184.216.34", "api.groq.com", "10.0.0.5"])
def test_active_guard_blocks_outside_connections(fresh_guard, host):
    with pytest.raises(EgressBlocked) as err:
        connect(host)
    assert isinstance(err.value, OSError)
    assert fresh_guard.blocked == 1 and not [a for a in fresh_guard.reached if not is_local(a[0])], "nothing got through"


def test_ipv6_outside_is_blocked(fresh_guard):
    with pytest.raises(EgressBlocked):
        s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        s.connect(("2606:4700::1111", 443, 0, 0))


@pytest.mark.parametrize("host", ["127.0.0.1", "127.9.9.9", "localhost", "::ffff:127.0.0.1"])
def test_loopback_passes(fresh_guard, host):
    try:
        connect(host, 11434)  # a local server may or may not be listening; either way the guard let it through
    except OSError as exc:
        assert not isinstance(exc, EgressBlocked)
    assert (host, 11434) in fresh_guard.reached and fresh_guard.blocked == 0


def test_ipv6_loopback_passes(fresh_guard):
    s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    try:
        s.connect(("::1", 8700, 0, 0))
    except OSError as exc:  # refused when nothing listens: that is the real socket answering, not the guard
        assert not isinstance(exc, EgressBlocked)
    finally:
        s.close()
    assert fresh_guard.blocked == 0 and any(a[0] == "::1" for a in fresh_guard.reached)


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="no Unix sockets on this platform")
def test_unix_sockets_stay_local(fresh_guard):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect("/tmp/some.sock")
    s.close()
    assert fresh_guard.reached == ["/tmp/some.sock"] and fresh_guard.blocked == 0


def test_connect_ex_is_guarded_too(fresh_guard):
    s = socket.socket()
    with pytest.raises(EgressBlocked):
        s.connect_ex(("8.8.8.8", 53))
    assert isinstance(s.connect_ex(("127.0.0.1", 9)), int)  # loopback reaches the real socket and returns a code
    s.close()


def test_dns_lookups_for_outside_names_are_refused(fresh_guard):
    for call in (socket.getaddrinfo, socket.gethostbyname, socket.gethostbyname_ex):
        with pytest.raises(EgressBlocked):
            call("example.com") if call is not socket.getaddrinfo else call("example.com", 443)
    socket.getaddrinfo("localhost", 80)
    socket.getaddrinfo(None, 80)
    socket.gethostbyname("127.0.0.1")
    assert fresh_guard.blocked == 3 and [r[1] for r in fresh_guard.reached] == ["localhost", None, "127.0.0.1"]


def test_an_allowed_host_is_the_only_exception():
    g = PrivacyGuard()
    g.install(lambda: True, allow_hosts=["10.0.0.5", "OLLAMA.lan"])
    g._orig["connect"] = lambda sock, address, *a: None
    try:
        connect("10.0.0.5")
        connect("ollama.lan")
        with pytest.raises(EgressBlocked):
            connect("10.0.0.6")
    finally:
        g.uninstall()


def test_inactive_guard_passes_everything_and_toggling_is_instant(fresh_guard):
    fresh_guard.flag["on"] = False
    connect("8.8.8.8")
    assert fresh_guard.reached == [("8.8.8.8", 443)] and fresh_guard.blocked == 0
    fresh_guard.flag["on"] = True
    with pytest.raises(EgressBlocked):
        connect("8.8.8.8")
    fresh_guard.flag["on"] = False
    connect("8.8.8.8")
    assert len(fresh_guard.reached) == 2


def test_attempts_are_recorded_even_when_off_with_host_and_port_only(fresh_guard):
    fresh_guard.flag["on"] = False
    connect("8.8.8.8", 443)
    assert list(fresh_guard.attempts) == [("connection", "8.8.8.8", 443, False)]


def test_logs_contain_host_and_port_only(fresh_guard, caplog):
    with caplog.at_level(logging.WARNING, logger="evora.privacy"), pytest.raises(EgressBlocked):
        s = socket.socket()
        s.connect(("93.184.216.34", 443))
    assert "93.184.216.34:443" in caplog.text and "on-prem" in caplog.text


def test_install_is_idempotent_and_uninstall_restores_the_originals():
    before = (socket.socket.connect, socket.getaddrinfo, "connect" in socket.socket.__dict__)
    g = PrivacyGuard()
    g.install(lambda: True)
    patched = socket.socket.connect
    g.install(lambda: True)
    assert socket.socket.connect is patched and g.installed
    g.uninstall()
    after = (socket.socket.connect, socket.getaddrinfo, "connect" in socket.socket.__dict__)
    assert after == before and not g.installed
    g.uninstall()  # twice is fine


def test_blocking_applies_in_worker_threads(fresh_guard):
    seen = []

    def work():
        try:
            connect("8.8.8.8")
        except EgressBlocked as exc:
            seen.append(exc)

    t = threading.Thread(target=work)
    t.start()
    t.join()
    assert len(seen) == 1 and fresh_guard.blocked == 1


def test_asyncio_connections_are_blocked(fresh_guard):
    async def go():
        await asyncio.open_connection("93.184.216.34", 80)

    with pytest.raises(OSError):
        asyncio.run(go())
    assert fresh_guard.blocked >= 1 and not [a for a in fresh_guard.reached if not is_local(a[0])]


def test_httpx_cannot_reach_an_outside_name(fresh_guard):
    async def go():
        async with httpx.AsyncClient(timeout=2) as client:
            await client.get("https://api.groq.com/openai/v1/models")

    with pytest.raises(httpx.HTTPError):
        asyncio.run(go())
    assert fresh_guard.blocked >= 1 and not any(isinstance(r, tuple) and r[0] == "api.groq.com" for r in fresh_guard.reached)


def test_the_real_listener_on_loopback_still_works_while_blocking():
    g = PrivacyGuard()
    g.install(lambda: True)
    try:
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        client = socket.socket()
        client.connect(server.getsockname())  # a real loopback connection through the patched connect
        conn, _ = server.accept()
        client.sendall(b"ok")
        assert conn.recv(2) == b"ok"
        for s in (client, conn, server):
            s.close()
        assert g.blocked == 0
    finally:
        g.uninstall()


def test_module_level_guard_is_the_one_the_app_installs():
    assert isinstance(privacy_guard.guard, PrivacyGuard)
