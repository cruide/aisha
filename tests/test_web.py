# Author: Tischenko A. (https://github.com/cruide)
import socket

import pytest

from aisha.errors import ToolPermissionError, ToolValidationError
from aisha.tools.web import _is_private_ip, check_url, html_to_text


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


def test_is_private_ip():
    assert _is_private_ip("10.0.0.1")
    assert _is_private_ip("127.0.0.1")
    assert not _is_private_ip("93.184.216.34")


def test_html_to_text_strips_scripts():
    html = ("<html><head><title>T</title></head><body><p>Hello</p>"
            "<script>var x = 1;</script></body></html>")
    title, text = html_to_text(html)
    assert title == "T"
    assert "Hello" in text
    assert "var x = 1" not in text
