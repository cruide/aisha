# Author: Tischenko A. (https://github.com/cruide)
"""web_search (ddgs) and web_fetch (httpx + BeautifulSoup) with SSRF protections."""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
import time
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from rich.markup import escape

from aisha.errors import ToolPermissionError, ToolValidationError
from aisha.tools.base import Tool, ToolContext, ToolResult

MAX_REDIRECTS = 5
MAX_CONCURRENT_FETCHES = 4
MIN_HOST_INTERVAL = 0.2
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) aisha/0.2"


def _is_private_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True
    if (addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved
            or addr.is_multicast or addr.is_unspecified):
        return True
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped is not None:
            return _is_private_ip(str(addr.ipv4_mapped))
        if addr.sixtofour is not None:
            return _is_private_ip(str(addr.sixtofour))
        if addr.teredo is not None:
            return True
    return False


async def check_url(url: str, allow_private: bool) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ToolValidationError(f"Only http/https URLs are allowed, got: {url}")
    host = parsed.hostname
    if not host:
        raise ToolValidationError(f"Invalid URL: {url}")
    if allow_private:
        return
    if host.lower() in ("localhost", "localhost.localdomain") or host.endswith(".local"):
        raise ToolPermissionError(f"Access to localhost is forbidden: {host}")
    try:
        infos = await asyncio.wait_for(
            asyncio.to_thread(socket.getaddrinfo, host, None, proto=socket.IPPROTO_TCP), 10
        )
    except (socket.gaierror, asyncio.TimeoutError) as exc:
        raise ToolValidationError(f"Failed to resolve hostname {host}: {exc}") from exc
    for info in infos:
        if _is_private_ip(info[4][0]):
            raise ToolPermissionError(f"Access to private addresses is forbidden: {host}")


def html_to_text(html: str) -> tuple[str, str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(["script", "style", "noscript", "template", "svg", "iframe", "head"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return title, text


class WebSearchTool(Tool):
    name = "web_search"
    read_only = True
    description = (
        "Search the web for a focused query. Returns ranked titles, URLs and snippets; "
        "use web_fetch "
        "on a promising URL when the full page content is needed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 25},
        },
        "required": ["query"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        cfg = ctx.config.web
        limit = max(1, min(int(args.get("max_results") or cfg.max_results), 25))
        query = args["query"].strip()

        def _search() -> list[dict[str, Any]]:
            from ddgs import DDGS

            return list(DDGS().text(query, max_results=limit))

        try:
            raw = await asyncio.wait_for(asyncio.to_thread(_search), cfg.timeout + 10)
        except asyncio.TimeoutError:
            return ToolResult.failure("ToolTimeoutError", "Search provider timed out")
        except Exception as exc:  # provider errors must be returned to the model
            return ToolResult.failure("SearchProviderError", f"Search error: {exc}")
        results = [
            {"position": i, "title": r.get("title", ""), "url": r.get("href") or r.get("url", ""),
             "snippet": r.get("body", "")}
            for i, r in enumerate(raw, 1)
        ]
        return ToolResult.success({"query": query, "results": results},
                                  f"{len(results)} results")


class WebFetchTool(Tool):
    name = "web_fetch"
    read_only = True
    description = (
        "Fetch a URL and extract readable page text. Use the exact URL from search results; "
        "output may be "
        "truncated, so request a relevant page or section rather than relying on the whole site."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "max_chars": {"type": "integer", "minimum": 1},
        },
        "required": ["url"],
    }

    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)
        self._host_locks: dict[str, asyncio.Lock] = {}
        self._last_request: dict[str, float] = {}

    async def _throttle(self, host: str) -> None:
        """Space request starts for one hostname to avoid accidental bursts."""
        lock = self._host_locks.setdefault(host, asyncio.Lock())
        async with lock:
            delay = MIN_HOST_INTERVAL - (time.monotonic() - self._last_request.get(host, 0.0))
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_request[host] = time.monotonic()

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        cfg = ctx.config.web
        url: str = args["url"].strip()
        max_chars = max(1, min(int(args.get("max_chars") or cfg.max_content_chars),
                                cfg.max_content_chars))
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/*;q=0.9,*/*;q=0.5",
        }
        async with self._semaphore:
            async with httpx.AsyncClient(timeout=cfg.timeout, follow_redirects=False,
                                         headers=headers) as client:
                for _ in range(MAX_REDIRECTS + 1):
                    await check_url(url, cfg.allow_private_hosts)
                    host = urlparse(url).hostname
                    if host is None:
                        raise ToolValidationError(f"Invalid URL: {url}")
                    await self._throttle(host.lower())
                    try:
                        async with client.stream("GET", url) as resp:
                            if resp.is_redirect and resp.headers.get("location"):
                                url = urljoin(url, resp.headers["location"])
                                continue
                            if resp.status_code >= 400:
                                return ToolResult.failure(
                                    "HTTPError", f"HTTP {resp.status_code}: {url}"
                                )
                            body = bytearray()
                            truncated = False
                            async for chunk in resp.aiter_bytes():
                                remaining = cfg.max_page_bytes - len(body)
                                body.extend(chunk[:remaining])
                                if len(body) >= cfg.max_page_bytes:
                                    truncated = True
                                    break
                            ctype = resp.headers.get("content-type", "")
                            encoding = resp.charset_encoding or "utf-8"
                    except httpx.HTTPError as exc:
                        return ToolResult.failure("HTTPError", f"Failed to fetch {url}: {exc}")
                    break
                else:
                    return ToolResult.failure("HTTPError", "Too many redirects")

        raw = bytes(body).decode(encoding, errors="replace")
        if "html" in ctype or raw.lstrip()[:200].lower().startswith(("<!doctype", "<html")):
            title, text = await asyncio.to_thread(html_to_text, raw)
        else:
            title, text = "", raw
        if len(text) > max_chars:
            text, truncated = text[:max_chars], True
        return ToolResult.success(
            {"url": url, "title": title, "content_type": ctype, "text": text},
            f"{escape(title[:60] or url)} · {len(text)} chars", truncated=truncated,
        )
