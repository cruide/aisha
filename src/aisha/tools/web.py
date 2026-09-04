# Author: Tischenko A. (https://github.com/cruide)
"""web_search (ddgs) and web_fetch (httpx + BeautifulSoup) with SSRF protections."""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from aisha.errors import ToolPermissionError, ToolValidationError
from aisha.tools.base import Tool, ToolContext, ToolResult

MAX_REDIRECTS = 5
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) aisha/0.2"


def _is_private_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True
    return (addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved
            or addr.is_multicast or addr.is_unspecified)


async def check_url(url: str, allow_private: bool) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ToolValidationError(f"Разрешены только http/https URL, получено: {url}")
    host = parsed.hostname
    if not host:
        raise ToolValidationError(f"Некорректный URL: {url}")
    if allow_private:
        return
    if host.lower() in ("localhost", "localhost.localdomain") or host.endswith(".local"):
        raise ToolPermissionError(f"Доступ к локальному хосту запрещён: {host}")
    try:
        infos = await asyncio.wait_for(
            asyncio.to_thread(socket.getaddrinfo, host, None, proto=socket.IPPROTO_TCP), 10
        )
    except (socket.gaierror, asyncio.TimeoutError) as exc:
        raise ToolValidationError(f"Не удалось разрешить имя хоста {host}: {exc}") from exc
    for info in infos:
        if _is_private_ip(info[4][0]):
            raise ToolPermissionError(f"Доступ к приватным адресам запрещён: {host}")


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
        "Поиск в интернете (DuckDuckGo). Обязательный аргумент: query — поисковый запрос. "
        "Необязательный: max_results (количество результатов). Возвращает заголовки, URL и "
        "сниппеты. Пример: web_search(query=\"как настроить llama.cpp\")."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
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
            return ToolResult.failure("ToolTimeoutError", "Поисковый провайдер не ответил вовремя")
        except Exception as exc:  # provider errors must be returned to the model
            return ToolResult.failure("SearchProviderError", f"Ошибка поиска: {exc}")
        results = [
            {"position": i, "title": r.get("title", ""), "url": r.get("href") or r.get("url", ""),
             "snippet": r.get("body", "")}
            for i, r in enumerate(raw, 1)
        ]
        return ToolResult.success({"query": query, "results": results},
                                  f"{len(results)} результатов")


class WebFetchTool(Tool):
    name = "web_fetch"
    read_only = True
    description = (
        "Загрузить веб-страницу по URL и вернуть извлечённый текст. Обязательный аргумент: url — "
        "полный адрес с http/https. Необязательный: max_chars (лимит символов текста). "
        "Пример: web_fetch(url=\"https://example.com/docs\")."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "max_chars": {"type": "integer"},
        },
        "required": ["url"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        cfg = ctx.config.web
        url: str = args["url"].strip()
        max_chars = min(int(args.get("max_chars") or cfg.max_content_chars), cfg.max_content_chars)
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/*;q=0.9,*/*;q=0.5",
        }
        async with httpx.AsyncClient(timeout=cfg.timeout, follow_redirects=False,
                                     headers=headers) as client:
            for _ in range(MAX_REDIRECTS + 1):
                await check_url(url, cfg.allow_private_hosts)
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
                            body.extend(chunk)
                            if len(body) >= cfg.max_page_bytes:
                                truncated = True
                                break
                        ctype = resp.headers.get("content-type", "")
                        encoding = resp.charset_encoding or "utf-8"
                except httpx.HTTPError as exc:
                    return ToolResult.failure("HTTPError", f"Ошибка загрузки {url}: {exc}")
                break
            else:
                return ToolResult.failure("HTTPError", "Слишком много редиректов")

        raw = bytes(body).decode(encoding, errors="replace")
        if "html" in ctype or raw.lstrip()[:200].lower().startswith(("<!doctype", "<html")):
            title, text = html_to_text(raw)
        else:
            title, text = "", raw
        if len(text) > max_chars:
            text, truncated = text[:max_chars], True
        return ToolResult.success(
            {"url": url, "title": title, "content_type": ctype, "text": text},
            f"{title[:60] or url} · {len(text)} символов", truncated=truncated,
        )
