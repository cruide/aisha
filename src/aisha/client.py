"""HTTP/SSE client for llama-server OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger(__name__)


class ServerUnavailableError(Exception):
    """llama-server is unreachable."""


class ProtocolError(Exception):
    """Invalid SSE or JSON from server."""


class ContextOverflowError(Exception):
    """Request exceeds context window."""


@dataclass
class StreamDelta:
    """A single streaming delta."""

    text: str = ""
    reasoning: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict | None = None


@dataclass
class ChatResponse:
    """Complete chat response."""

    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    model: str = ""


class LlamaClient:
    """Async client for llama-server OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        connect_timeout: int = 5,
        request_timeout: int = 600,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout
        self._stream_options_supported: bool | None = None
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self.connect_timeout,
                    read=self.request_timeout,
                    write=self.connect_timeout,
                    pool=self.connect_timeout,
                ),
                limits=httpx.Limits(max_connections=10),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def check_health(self) -> dict:
        """Check /health endpoint."""
        client = await self._get_client()
        try:
            resp = await client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()
        except httpx.ConnectError as e:
            raise ServerUnavailableError(
                f"Не удалось подключиться к {self.base_url}: {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise ServerUnavailableError(
                f"Сервер вернул ошибку {e.response.status_code}: {e.response.text}"
            ) from e

    async def list_models(self) -> list[dict]:
        """Get available models from /v1/models."""
        client = await self._get_client()
        try:
            resp = await client.get(f"{self.base_url}/v1/models")
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", [])
        except httpx.ConnectError as e:
            raise ServerUnavailableError(
                f"Не удалось подключиться к {self.base_url}: {e}"
            ) from e

    async def verify_model(self) -> str:
        """Resolve the model id actually served by llama-server.

        llama-server serves the single loaded model regardless of the
        name in the request, so the configured name is only a hint. If
        the exact name is not advertised, fall back to the first
        advertised model id, or keep the configured name when the server
        advertises none.
        """
        models = await self.list_models()
        model_ids = [m.get("id", "") for m in models]
        if self.model in model_ids:
            return self.model
        if model_ids:
            logger.info(
                "Model '%s' not advertised; using '%s'", self.model, model_ids[0]
            )
            self.model = model_ids[0]
        return self.model

    async def get_model_context_size(self, model_id: str) -> int | None:
        """Return the advertised context size (meta.n_ctx) for a model id."""
        models = await self.list_models()
        for m in models:
            if m.get("id") == model_id:
                meta = m.get("meta") or {}
                n_ctx = meta.get("n_ctx")
                return n_ctx if isinstance(n_ctx, int) else None
        return None

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        stream: bool = True,
        tool_choice: str = "auto",
    ) -> ChatResponse | AsyncIterator[StreamDelta]:
        """Send chat completion request.

        Returns ChatResponse for non-streaming, AsyncIterator[StreamDelta] for streaming.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "tool_choice": tool_choice,
        }
        if tools:
            payload["tools"] = tools

        if stream:
            if self._stream_options_supported is not False:
                payload["stream_options"] = {"include_usage": True}
            return self._stream_chat(payload)
        else:
            return await self._non_stream_chat(payload)

    async def _non_stream_chat(self, payload: dict) -> ChatResponse:
        """Non-streaming chat completion."""
        client = await self._get_client()
        url = f"{self.base_url}/v1/chat/completions"

        for attempt in range(3):
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return self._parse_response(data)
            except httpx.ConnectError:
                if attempt < 2:
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue
                raise ServerUnavailableError("Сервер недоступен после 3 попыток")
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 502, 503, 504) and attempt < 2:
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue
                if e.response.status_code == 400:
                    error_data = {}
                    try:
                        error_data = e.response.json()
                    except Exception:
                        pass
                    error_msg = error_data.get("error", {}).get("message", "")
                    if "context" in error_msg.lower() or "length" in error_msg.lower():
                        raise ContextOverflowError(error_msg) from e
                raise

    async def _stream_chat(self, payload: dict) -> AsyncIterator[StreamDelta]:
        """Streaming chat completion via SSE."""
        client = await self._get_client()
        url = f"{self.base_url}/v1/chat/completions"

        retries = 0
        max_retries = 3

        while True:
            try:
                async with client.stream("POST", url, json=payload) as resp:
                    if resp.status_code != 200:
                        # Read error body
                        body = await resp.aread()
                        error_msg = ""
                        try:
                            error_data = json.loads(body)
                            error_msg = error_data.get("error", {}).get("message", "")
                        except Exception:
                            error_msg = body.decode(errors="replace")

                        if resp.status_code == 400 and self._stream_options_supported is None:
                            # Try without stream_options
                            self._stream_options_supported = False
                            payload.pop("stream_options", None)
                            async for delta in self._stream_chat(payload):
                                yield delta
                            return

                        if resp.status_code in (429, 502, 503, 504) and retries < max_retries - 1:
                            retries += 1
                            await asyncio.sleep(0.5 * (2 ** (retries - 1)))
                            continue

                        if resp.status_code == 400:
                            if "context" in error_msg.lower() or "length" in error_msg.lower():
                                raise ContextOverflowError(error_msg)
                        raise ProtocolError(f"HTTP {resp.status_code}: {error_msg}")

                    # Successfully connected — no more retries
                    if self._stream_options_supported is None:
                        self._stream_options_supported = True

                    async for line in self._iter_sse_lines(resp):
                        yield line
                    return

            except httpx.ConnectError:
                if retries < max_retries - 1:
                    retries += 1
                    await asyncio.sleep(0.5 * (2 ** (retries - 1)))
                    continue
                raise ServerUnavailableError("Сервер недоступен после 3 попыток")

    async def _iter_sse_lines(self, resp: httpx.Response) -> AsyncIterator[StreamDelta]:
        """Parse SSE stream from response."""
        # Accumulate tool calls by index
        tool_call_accum: dict[int, dict[str, Any]] = {}

        async for raw_line in resp.aiter_lines():
            if not raw_line:
                continue

            if raw_line.startswith("data: "):
                data_str = raw_line[6:].strip()
            else:
                continue

            if data_str == "[DONE]":
                return

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            choices = data.get("choices", [])
            usage = data.get("usage")

            for choice in choices:
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")

                # Text content
                content = delta.get("content", "") or ""

                # Reasoning content
                reasoning = delta.get("reasoning_content", "") or ""

                # Tool calls (may be fragmented)
                raw_tool_calls = delta.get("tool_calls", [])
                for tc in raw_tool_calls:
                    idx = tc.get("index", 0)
                    if idx not in tool_call_accum:
                        tool_call_accum[idx] = {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    entry = tool_call_accum[idx]
                    if tc.get("id"):
                        entry["id"] = tc["id"]
                    func = tc.get("function", {})
                    if func.get("name"):
                        entry["function"]["name"] = func["name"]
                    if func.get("arguments"):
                        entry["function"]["arguments"] += func["arguments"]

                # Build tool calls list for this delta
                delta_tool_calls = []
                for idx in sorted(tool_call_accum.keys()):
                    tc = tool_call_accum[idx]
                    if tc["function"]["name"]:
                        delta_tool_calls.append(tc)

                yield StreamDelta(
                    text=content,
                    reasoning=reasoning,
                    tool_calls=delta_tool_calls if delta_tool_calls else [],
                    finish_reason=finish_reason,
                    usage=usage,
                )

            # Final usage-only chunk (stream_options.include_usage) has empty choices
            if usage and not choices:
                yield StreamDelta(usage=usage)

    def _parse_response(self, data: dict) -> ChatResponse:
        """Parse a non-streaming response."""
        choices = data.get("choices", [])
        if not choices:
            raise ProtocolError("Пустой ответ от сервера (нет choices)")

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""
        tool_calls = message.get("tool_calls", [])
        finish_reason = choice.get("finish_reason", "")
        usage = data.get("usage", {})

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            model=data.get("model", self.model),
        )

    async def tool_call_test(self) -> bool:
        """Send a safe test request to verify tool calling works."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "description": "List files in a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Directory path"}
                        },
                        "required": ["path"],
                    },
                },
            }
        ]
        messages = [{"role": "user", "content": "List files in current directory"}]
        try:
            result = await self.chat(
                messages, tools=tools, stream=False, temperature=0.0
            )
            if isinstance(result, ChatResponse):
                return bool(result.tool_calls)
            return False
        except Exception:
            return False
