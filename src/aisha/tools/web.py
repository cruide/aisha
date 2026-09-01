"""Web tools: web_search and web_fetch."""

from __future__ import annotations

__author__ = "Tischenko Alexander"

import asyncio
import ipaddress
import socket
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from aisha.tools.base import Tool, ToolResult, ToolValidationError


async def _is_private_host(hostname: str) -> bool:
    """Check if hostname resolves to a private/reserved address."""
    try:
        ip_str = await asyncio.to_thread(socket.gethostbyname, hostname)
        ip = ipaddress.ip_address(ip_str)
        return ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local
    except (socket.gaierror, ValueError):
        return False


class WebSearchTool(Tool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web using DuckDuckGo."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Max results (default 8)",
                    "default": 8,
                },
            },
            "required": ["query"],
        }

    def validate_args(self, args: dict) -> dict:
        if not args.get("query", "").strip():
            raise ToolValidationError("query must not be empty")
        return args

    async def execute(self, args: dict, context: dict) -> ToolResult:
        query = args["query"]
        max_results = min(args.get("max_results", 8), 20)

        try:
            from ddgs import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                        "position": i,
                    }
                )

            return ToolResult.success(
                {"query": query, "results": formatted, "count": len(formatted)}
            )
        except Exception as e:
            return ToolResult.failure("WebSearchError", f"Ошибка поиска: {e}")


class WebFetchTool(Tool):
    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch and extract text content from a web page."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
                "max_chars": {
                    "type": "integer",
                    "description": "Max content chars (default 50000)",
                    "default": 50000,
                },
            },
            "required": ["url"],
        }

    def validate_args(self, args: dict) -> dict:
        url = args.get("url", "")
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ToolValidationError(f"Only http/https URLs allowed: {url}")
        return args

    async def execute(self, args: dict, context: dict) -> ToolResult:
        url = args["url"]
        max_chars = args.get("max_chars", 50000)
        allow_private = context.get("allow_private_hosts", False)
        web_timeout = context.get("web_timeout", 20)
        max_page_bytes = context.get("max_page_bytes", 2097152)

        parsed = urlparse(url)
        hostname = parsed.hostname or ""

        # Check private hosts
        if not allow_private and await _is_private_host(hostname):
            return ToolResult.failure(
                "PermissionDenied",
                f"Обращение к приватным хостам запрещено: {hostname}",
            )

        try:
            async with httpx.AsyncClient(
                timeout=web_timeout,
                follow_redirects=True,
                max_redirects=5,
                limits=httpx.Limits(max_connections=5),
            ) as client:
                resp = await client.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; aisha/0.1)"},
                )
                resp.raise_for_status()

                # Limit response size
                content_bytes = resp.content[:max_page_bytes]
                truncated_bytes = len(resp.content) > max_page_bytes

                # Parse HTML
                content_type = resp.headers.get("content-type", "")
                if "text/html" in content_type:
                    soup = BeautifulSoup(content_bytes, "html.parser")
                    # Remove scripts, styles, nav, footer
                    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                        tag.decompose()

                    title = soup.title.string if soup.title else ""
                    text = soup.get_text(separator="\n", strip=True)
                else:
                    title = parsed.path.split("/")[-1] or url
                    text = content_bytes.decode("utf-8", errors="replace")

                # Truncate text
                truncated_text = False
                if len(text) > max_chars:
                    text = text[:max_chars]
                    truncated_text = True

                return ToolResult.success(
                    {
                        "url": url,
                        "title": title.strip() if title else "",
                        "content": text,
                        "status_code": resp.status_code,
                        "truncated": truncated_text or truncated_bytes,
                    }
                )
        except httpx.ConnectError as e:
            return ToolResult.failure("ConnectionError", f"Не удалось подключиться: {e}")
        except httpx.TimeoutException:
            return ToolResult.failure("TimeoutError", f"Таймаут при загрузке: {url}")
        except httpx.HTTPStatusError as e:
            return ToolResult.failure(
                "HTTPError", f"HTTP {e.response.status_code}: {url}"
            )
        except Exception as e:
            return ToolResult.from_exception(e)
