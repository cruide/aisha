"""Async OpenAI-compatible client for llama-server: SSE streaming, tool-call assembly, retries."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from aisha.errors import ContextOverflowError, ProtocolError, ServerUnavailableError

RETRY_DELAYS = (0.5, 1.0, 2.0)
RETRY_STATUSES = {429, 502, 503, 504}

EventCallback = Callable[[str, str], None]  # (kind: "text" | "reasoning", delta)


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str = ""

    def parse_arguments(self) -> dict[str, Any]:
        raw = self.arguments.strip() or "{}"
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("аргументы должны быть JSON-объектом")
        return obj

    def to_message(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments or "{}"},
        }


@dataclass(slots=True)
class ChatResponse:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] | None = None

    def to_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant"}
        if self.content or not self.tool_calls:
            msg["content"] = self.content
        if self.reasoning:
            msg["reasoning_content"] = self.reasoning
        if self.tool_calls:
            msg["tool_calls"] = [call.to_message() for call in self.tool_calls]
        return msg


class _Retryable(Exception):
    """Transient failure before any streamed data was received."""


class _StreamOptionsUnsupported(Exception):
    """Server rejected the `stream_options` field."""


class LlamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        connect_timeout: float = 5.0,
        request_timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(request_timeout, connect=connect_timeout),
        )
        self._stream_options_ok = True

    async def close(self) -> None:
        await self._http.aclose()

    # ----------------------------------------------------------------- probes
    async def health(self) -> dict[str, Any]:
        try:
            resp = await self._http.get("/health")
        except httpx.HTTPError as exc:
            raise ServerUnavailableError(f"Сервер {self.base_url} недоступен: {exc}") from exc
        if resp.status_code == 503:
            raise ServerUnavailableError("Сервер отвечает 503: модель ещё загружается")
        if resp.status_code != 200:
            raise ServerUnavailableError(f"/health вернул HTTP {resp.status_code}")
        try:
            return resp.json()
        except ValueError as exc:
            raise ProtocolError("/health вернул не JSON") from exc

    async def model_info(self) -> dict[str, dict[str, Any]]:
        """Return {model_id: meta} parsed from /v1/models."""
        try:
            resp = await self._http.get("/v1/models")
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPError as exc:
            raise ServerUnavailableError(f"Не удалось получить /v1/models: {exc}") from exc
        except ValueError as exc:
            raise ProtocolError("/v1/models вернул не JSON") from exc
        items = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise ProtocolError("Несовместимый ответ /v1/models")
        result: dict[str, dict[str, Any]] = {}
        for item in items:
            if isinstance(item, dict) and item.get("id"):
                meta = item.get("meta")
                result[str(item["id"])] = meta if isinstance(meta, dict) else {}
        return result

    async def models(self) -> list[str]:
        return list((await self.model_info()).keys())

    async def context_window(self, model: str) -> int | None:
        """Context window (meta.n_ctx) advertised for `model`, or None if unknown."""
        meta = (await self.model_info()).get(model, {})
        n_ctx = meta.get("n_ctx")
        return n_ctx if isinstance(n_ctx, int) and n_ctx > 0 else None

    async def resolve_model(self) -> tuple[str, bool]:
        """Return (model, matched). Falls back to the first available model."""
        model, matched, _ = await self.resolve_model_meta()
        return model, matched

    async def resolve_model_meta(self) -> tuple[str, bool, int | None]:
        """Resolve the model and fetch its context window in a single /v1/models request.

        Returns (model, matched, n_ctx). Falls back to the first available model.
        """
        await self.health()
        info = await self.model_info()
        names = list(info)
        if self.model in info:
            model, matched = self.model, True
        elif names:
            self.model = model = names[0]
            matched = False
        else:
            raise ServerUnavailableError("Сервер не вернул ни одной модели")
        meta = info.get(model, {})
        n_ctx = meta.get("n_ctx")
        n_ctx = n_ctx if isinstance(n_ctx, int) and n_ctx > 0 else None
        return model, matched, n_ctx

    # ------------------------------------------------------------------- chat
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float,
        max_tokens: int,
        on_event: EventCallback | None = None,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self._stream_options_ok:
            payload["stream_options"] = {"include_usage": True}

        attempt = 0
        while True:
            try:
                return await self._stream_once(payload, on_event)
            except _StreamOptionsUnsupported:
                self._stream_options_ok = False
                payload.pop("stream_options", None)
            except _Retryable as exc:
                if attempt >= len(RETRY_DELAYS):
                    raise ServerUnavailableError(f"Сервер недоступен: {exc}") from exc
                await asyncio.sleep(RETRY_DELAYS[attempt])
                attempt += 1

    async def _stream_once(
        self, payload: dict[str, Any], on_event: EventCallback | None
    ) -> ChatResponse:
        started = False
        result = ChatResponse()
        pending: dict[int, ToolCall] = {}
        try:
            async with self._http.stream("POST", "/v1/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    self._raise_for_status(resp.status_code, body, payload)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        break
                    started = True
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise ProtocolError(f"Некорректный JSON в SSE: {data[:200]}") from exc
                    self._apply_chunk(chunk, result, pending, on_event)
        except httpx.ReadTimeout as exc:
            if started:
                raise ServerUnavailableError("Таймаут ожидания ответа сервера") from exc
            raise _Retryable(str(exc)) from exc
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError) as exc:
            if started:
                raise ProtocolError(f"Соединение прервано во время ответа: {exc}") from exc
            raise _Retryable(str(exc)) from exc

        for idx in sorted(pending):
            call = pending[idx]
            if not call.id:
                call.id = f"call_{idx}"
            result.tool_calls.append(call)
        return result

    @staticmethod
    def _raise_for_status(status: int, body: str, payload: dict[str, Any]) -> None:
        if status == 400 and "stream_options" in body and "stream_options" in payload:
            raise _StreamOptionsUnsupported()
        if status in RETRY_STATUSES:
            raise _Retryable(f"HTTP {status}")
        lowered = body.lower()
        if status == 400 and ("context" in lowered or "exceed" in lowered or "n_ctx" in lowered):
            raise ContextOverflowError(f"Запрос превышает контекст модели: {body[:300]}")
        raise ProtocolError(f"HTTP {status}: {body[:500]}")

    @staticmethod
    def _apply_chunk(
        chunk: dict[str, Any],
        result: ChatResponse,
        pending: dict[int, ToolCall],
        on_event: EventCallback | None,
    ) -> None:
        if "error" in chunk:
            err = chunk["error"]
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise ProtocolError(f"Ошибка сервера: {msg}")
        usage = chunk.get("usage")
        if isinstance(usage, dict) and usage:
            result.usage = usage
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if text:
                result.content += text
                if on_event:
                    on_event("text", text)
            reasoning = delta.get("reasoning_content")
            if reasoning:
                result.reasoning += reasoning
                if on_event:
                    on_event("reasoning", reasoning)
            for tc in delta.get("tool_calls") or []:
                try:
                    idx = int(tc.get("index", 0))
                except (TypeError, ValueError):
                    idx = 0
                call = pending.setdefault(idx, ToolCall(id="", name=""))
                if tc.get("id"):
                    call.id = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name") and not call.name:
                    call.name = fn["name"]
                if fn.get("arguments"):
                    call.arguments += fn["arguments"]
            if choice.get("finish_reason"):
                result.finish_reason = choice["finish_reason"]
