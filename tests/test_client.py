"""Tests for OpenAI/SSE client."""

from __future__ import annotations

__author__ = "Tischenko Alexander"

import pytest

from aisha.client import (
    ChatResponse,
    LlamaClient,
    ProtocolError,
    StreamDelta,
)


@pytest.fixture
def client() -> LlamaClient:
    return LlamaClient(
        base_url="http://localhost:8088",
        model="test-model",
        connect_timeout=2,
        request_timeout=10,
    )


def test_stream_delta_defaults() -> None:
    d = StreamDelta()
    assert d.text == ""
    assert d.reasoning == ""
    assert d.tool_calls == []
    assert d.finish_reason is None
    assert d.usage is None


class _FakeSSEResponse:
    """Minimal stand-in for httpx.Response with aiter_lines()."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


async def test_iter_sse_usage_only_final_chunk(client: LlamaClient) -> None:
    lines = [
        'data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}',
        "data: [DONE]",
    ]
    resp = _FakeSSEResponse(lines)
    deltas = [d async for d in client._iter_sse_lines(resp)]
    usage_deltas = [d for d in deltas if d.usage]
    assert len(usage_deltas) == 1
    assert usage_deltas[0].usage == {"prompt_tokens": 10, "completion_tokens": 5}
    assert "".join(d.text for d in deltas) == "Hello"


async def test_iter_sse_usage_attached_to_content_chunk(client: LlamaClient) -> None:
    lines = [
        'data: {"choices":[{"delta":{"content":"Hi"}}],'
        '"usage":{"prompt_tokens":7,"completion_tokens":2}}',
        "data: [DONE]",
    ]
    resp = _FakeSSEResponse(lines)
    deltas = [d async for d in client._iter_sse_lines(resp)]
    assert deltas[-1].usage == {"prompt_tokens": 7, "completion_tokens": 2}


def test_chat_response_defaults() -> None:
    r = ChatResponse()
    assert r.content == ""
    assert r.tool_calls == []
    assert r.finish_reason == ""


def test_chat_response_with_tool_calls() -> None:
    tc = [{"id": "1", "function": {"name": "test", "arguments": "{}"}}]
    r = ChatResponse(tool_calls=tc, finish_reason="tool_calls")
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0]["function"]["name"] == "test"


def test_parse_response(client: LlamaClient) -> None:
    data = {
        "choices": [
            {
                "message": {"content": "Hello", "role": "assistant"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    resp = client._parse_response(data)
    assert resp.content == "Hello"
    assert resp.finish_reason == "stop"
    assert resp.usage["prompt_tokens"] == 10


def test_parse_response_empty_choices(client: LlamaClient) -> None:
    data = {"choices": []}
    with pytest.raises(ProtocolError):
        client._parse_response(data)


def test_parse_response_with_tool_calls(client: LlamaClient) -> None:
    data = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "list_dir",
                                "arguments": '{"path": "."}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    resp = client._parse_response(data)
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0]["function"]["name"] == "list_dir"


async def test_verify_model_exact_match(client: LlamaClient) -> None:
    async def fake_list_models() -> list[dict]:
        return [{"id": "test-model"}, {"id": "other-model"}]

    client.list_models = fake_list_models
    resolved = await client.verify_model()
    assert resolved == "test-model"
    assert client.model == "test-model"


async def test_verify_model_falls_back_to_first_available(
    client: LlamaClient,
) -> None:
    async def fake_list_models() -> list[dict]:
        return [{"id": "Qwen3.5-9B-Q4_K_XL"}]

    client.list_models = fake_list_models
    resolved = await client.verify_model()
    assert resolved == "Qwen3.5-9B-Q4_K_XL"
    assert client.model == "Qwen3.5-9B-Q4_K_XL"


async def test_verify_model_keeps_name_when_none_advertised(
    client: LlamaClient,
) -> None:
    async def fake_list_models() -> list[dict]:
        return []

    client.list_models = fake_list_models
    resolved = await client.verify_model()
    assert resolved == "test-model"
    assert client.model == "test-model"


async def test_get_model_context_size(client: LlamaClient) -> None:
    async def fake_list_models() -> list[dict]:
        return [{"id": "test-model", "meta": {"n_ctx": 8192}}]

    client.list_models = fake_list_models
    assert await client.get_model_context_size("test-model") == 8192
    assert await client.get_model_context_size("unknown") is None


async def test_get_model_context_size_missing_meta(client: LlamaClient) -> None:
    async def fake_list_models() -> list[dict]:
        return [{"id": "test-model"}]

    client.list_models = fake_list_models
    assert await client.get_model_context_size("test-model") is None
