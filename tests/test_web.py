# Author: Tischenko A. (https://github.com/cruide)
import socket

import pytest

from aisha.errors import ToolPermissionError, ToolValidationError
from aisha.tools.web import WebFetchTool, _is_private_ip, check_url, html_to_text


def _getaddrinfo(ip):
    return lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 80))]


async def test_check_url_rejects_non_http():
    with pytest.raises(ToolValidationError):
        await check_url("ftp://example.com", False)


async def test_check_url_blocks_localhost():
    with pytest.raises(ToolPermissionError):
        await check_url("http://localhost:8080", False)


async def test_check_url_blocks_private_ip(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _getaddrinfo("10.0.0.1"))
    with pytest.raises(ToolPermissionError):
        await check_url("http://internal.corp", False)


async def test_check_url_allows_public_ip(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _getaddrinfo("93.184.216.34"))
    await check_url("http://example.com", False)


@pytest.mark.parametrize("ip", [
    "10.0.0.1", "127.0.0.1", "::1", "::ffff:127.0.0.1",
    "::ffff:169.254.169.254", "2002:a9fe:a9fe::", "2001:0000::1", "0.0.0.0",
])
def test_is_private_ip_blocks_restricted_addresses(ip):
    assert _is_private_ip(ip)


@pytest.mark.parametrize("ip", ["93.184.216.34", "8.8.8.8", "2606:4700:4700::1111"])
def test_is_private_ip_allows_public_addresses(ip):
    assert not _is_private_ip(ip)


def test_html_to_text_strips_scripts():
    html = ("<html><head><title>T</title></head><body><p>Hello</p>"
            "<script>var x = 1;</script></body></html>")
    title, text = html_to_text(html)
    assert title == "T"
    assert "Hello" in text
    assert "var x = 1" not in text


async def test_web_fetch_throttles_same_host(monkeypatch):
    clock = [10.0]
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr("aisha.tools.web.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("aisha.tools.web.asyncio.sleep", fake_sleep)
    tool = WebFetchTool()

    await tool._throttle("example.com")
    await tool._throttle("example.com")
    await tool._throttle("other.example")

    assert sleeps == [pytest.approx(0.2)]
