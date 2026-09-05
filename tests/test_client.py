# Author: Tischenko A. (https://github.com/cruide)
import json

import httpx
import pytest

from aisha.client import LlamaClient
from aisha.errors import ServerUnavailableError


def sse(*events):
    lines = [f"data: {json.dumps(e)}\n\n" for e in events] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()


class ChunkedStream(httpx.AsyncByteStream):
    def __init__(self, data: bytes, size: int):
        self.data, self.size = data, size

    async def __aiter__(self):
        for i in range(0, len(self.data), self.size):
            yield self.data[i:i + self.size]


def make_client(handler):
    client = LlamaClient("http://test", "m")
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
    return client


async def test_text_and_fragmented_tool_calls():
    body = sse(
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1",
                                                "function": {"name": "read_file",
                                                             "arguments": '{"pa'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0,
                                                "function": {"arguments": 'th": "a.py"}'}}]},
                      "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    )

    def handler(request):
        return httpx.Response(200, stream=ChunkedStream(body, 7))

    deltas = []
    resp = await make_client(handler).chat([], temperature=0, max_tokens=10,
                                            on_event=lambda k, t: deltas.append(t))
    assert resp.content == "Hello" and "".join(deltas) == "Hello"
    assert resp.finish_reason == "tool_calls"
    assert resp.tool_calls[0].id == "c1"
    assert resp.tool_calls[0].parse_arguments() == {"path": "a.py"}
    assert resp.usage["prompt_tokens"] == 10


async def test_stream_options_fallback_and_retry(monkeypatch):
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if "stream_options" in payload:
            return httpx.Response(400, json={"error": {"message": "unknown field stream_options"}})
        if len(calls) == 2:
            return httpx.Response(503)
        return httpx.Response(200, content=sse({"choices": [{"delta": {"content": "ok"}}]}))

    import aisha.client as mod
    monkeypatch.setattr(mod, "RETRY_DELAYS", (0, 0, 0))
    client = make_client(handler)
    resp = await client.chat([], temperature=0, max_tokens=10)
    assert resp.content == "ok" and client._stream_options_ok is False and len(calls) == 3


async def test_retry_exhausted(monkeypatch):
    import aisha.client as mod
    monkeypatch.setattr(mod, "RETRY_DELAYS", (0, 0, 0))

    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ServerUnavailableError):
        await make_client(handler).chat([], temperature=0, max_tokens=10)


async def test_sampling_params_passed_in_payload():
    captured = {}

    def handler(request):
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, content=sse({"choices": [{"delta": {"content": "ok"}}]}))

    client = make_client(handler)
    await client.chat([], temperature=0.7, max_tokens=10,
                       sampling={"top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1,
                                 "frequency_penalty": 0.0})
    payload = captured["payload"]
    assert payload["top_p"] == 0.9
    assert payload["top_k"] == 40
    assert payload["repeat_penalty"] == 1.1
    assert payload["frequency_penalty"] == 0.0


async def test_no_sampling_by_default():
    captured = {}

    def handler(request):
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, content=sse({"choices": [{"delta": {"content": "ok"}}]}))

    client = make_client(handler)
    await client.chat([], temperature=0, max_tokens=10)
    payload = captured["payload"]
    assert "top_p" not in payload and "top_k" not in payload
    assert "repeat_penalty" not in payload and "frequency_penalty" not in payload


async def test_health_returns_none_on_html():
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, text="<html><body>Not Found</body></html>")
        return httpx.Response(200, json={"data": [{"id": "m", "meta": {}}]})

    client = make_client(handler)
    assert await client.health() is None


async def test_health_returns_none_on_non_200():
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(404, text="not found")
        return httpx.Response(200, json={"data": [{"id": "m", "meta": {}}]})

    client = make_client(handler)
    assert await client.health() is None


async def test_health_skip_returns_none():
    def handler(request):
        return httpx.Response(200, json={"status": "ok"})

    client = make_client(handler)
    client.skip_health = True
    assert await client.health() is None


async def test_health_still_raises_on_connection_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    client = make_client(handler)
    with pytest.raises(ServerUnavailableError):
        await client.health()


async def test_health_still_raises_on_503():
    def handler(request):
        return httpx.Response(503)

    client = make_client(handler)
    with pytest.raises(ServerUnavailableError):
        await client.health()
