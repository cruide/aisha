"""Tests for agent loop."""

from __future__ import annotations

__author__ = "Tischenko Alexander"

from pathlib import Path

from aisha.agent import PARALLEL_TOOLS
from aisha.config import Config
from aisha.context import ContextManager


def test_parallel_tools_list() -> None:
    expected = {
        "read_file", "list_dir", "glob", "grep",
        "web_search", "web_fetch", "memory_get",
    }
    assert PARALLEL_TOOLS == expected


def _make_context(tmp_path: Path, messages: list[dict]) -> ContextManager:
    ctx = ContextManager(tmp_path, Config())
    ctx.messages = messages
    return ctx


def _tool_chain_msg(identity: str) -> tuple[dict, dict]:
    assistant = {
        "role": "assistant",
        "tool_calls": [{"id": identity, "function": {"name": "read_file", "arguments": "{}"}}],
    }
    tool = {"role": "tool", "tool_call_id": identity, "content": "result"}
    return assistant, tool


def test_safe_split_does_not_break_tool_chain(tmp_path: Path) -> None:
    a1, t1 = _tool_chain_msg("1")
    a2, t2 = _tool_chain_msg("2")
    messages = [
        {"role": "user", "content": "q1"},
        a1, t1,
        a2, t2,
        {"role": "user", "content": "q3"},
    ]
    ctx = _make_context(tmp_path, messages)
    # keep=2 lands on tool message of chain 2 (index 4), must move back to 3
    assert ctx._safe_split_index(2) == 3


def test_safe_split_keeps_boundary_when_not_tool(tmp_path: Path) -> None:
    a1, t1 = _tool_chain_msg("1")
    messages = [
        {"role": "user", "content": "q1"},
        a1, t1,
        {"role": "user", "content": "q2"},
    ]
    ctx = _make_context(tmp_path, messages)
    assert ctx._safe_split_index(1) == 3


def test_safe_split_respects_lower_bound(tmp_path: Path) -> None:
    ctx = _make_context(tmp_path, [{"role": "user", "content": "q"}])
    assert ctx._safe_split_index(0) == 1
    assert ctx._safe_split_index(10) == 0


def test_format_token_status_session_and_last(tmp_path: Path) -> None:
    ctx = ContextManager(tmp_path, Config())
    ctx.update_token_counts({"prompt_tokens": 1234, "completion_tokens": 56})
    ctx.update_token_counts({"prompt_tokens": 80, "completion_tokens": 10})
    assert (
        ctx.format_token_status()
        == "session: ↑ 1 314 ↓ 66 | last: ↑ 80 ↓ 10"
    )


def test_format_token_status_reset_on_clear(tmp_path: Path) -> None:
    ctx = ContextManager(tmp_path, Config())
    ctx.update_token_counts({"prompt_tokens": 100, "completion_tokens": 5})
    ctx.clear()
    assert ctx.format_token_status() == "session: ↑ 0 ↓ 0 | last: ↑ 0 ↓ 0"
