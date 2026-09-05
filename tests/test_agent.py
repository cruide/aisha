# Author: Tischenko A. (https://github.com/cruide)
import json
from pathlib import Path

import pytest

import aisha.agent as agent_mod
from aisha.agent import AgentLoop
from aisha.client import ChatResponse, ToolCall
from aisha.config import load_config
from aisha.context import ConversationContext, build_tool_guide
from aisha.skills import SkillIndex
from aisha.tools.base import Tool, ToolContext, ToolRegistry, ToolResult
from aisha.tools.files import EditFileTool, ReadFileTool


@pytest.fixture
def config(workspace: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: workspace.parent / "home"))
    return load_config(workspace, env={})


@pytest.fixture
def skills(workspace: Path) -> SkillIndex:
    return SkillIndex(workspace / ".aisha" / "skills", workspace / ".aisha" / "skills")


class EchoTool(Tool):
    name = "echo"
    read_only = True
    description = "echo"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def run(self, args, ctx):
        return ToolResult.success({"text": args["text"]}, "ok")


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat(self, messages, tools=None, *, temperature, max_tokens, on_event=None,
                   sampling=None):
        self.calls.append((messages, tools))
        resp = self.responses.pop(0)
        return resp(messages) if callable(resp) else resp


class FakeEvents:
    def __init__(self):
        self.notices = []
        self.starts = []
        self.ends = []
        self.debugs = []

    def on_stream_start(self):
        pass

    def on_text(self, delta):
        pass

    def on_reasoning(self, delta):
        pass

    def on_stream_end(self, response):
        pass

    def on_tool_start(self, call, args):
        self.starts.append(call.name)

    def on_tool_end(self, call, result):
        self.ends.append(call.name)

    def on_notice(self, text, level="info"):
        self.notices.append((level, text))

    def on_debug(self, title, body):
        self.debugs.append((title, body))


def make_agent(config, skills, workspace, client, events=None):
    context = ConversationContext(config, None, skills)
    registry = ToolRegistry()
    registry.register(EchoTool())
    tool_ctx = ToolContext(
        workspace=workspace, config=config, memory=None, skills=skills, todos=context.todos,
        on_system_change=context.invalidate,
    )
    return AgentLoop(config, client, registry, context, tool_ctx, events or FakeEvents())


async def test_agent_executes_tool_and_returns_content(config, skills, workspace):
    client = FakeClient([
        ChatResponse(tool_calls=[ToolCall(id="c1", name="echo", arguments='{"text": "hi"}')]),
        ChatResponse(content="done"),
    ])
    agent = make_agent(config, skills, workspace, client)
    result = await agent.run("call echo")
    assert result == "done"
    roles = [m["role"] for m in agent.context.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]


async def test_parallel_read_only_calls_both_execute(config, skills, workspace, monkeypatch):
    monkeypatch.setattr(agent_mod, "PARALLEL_TOOLS", agent_mod.PARALLEL_TOOLS | {"echo"})
    client = FakeClient([
        ChatResponse(tool_calls=[
            ToolCall(id="c1", name="echo", arguments='{"text": "a"}'),
            ToolCall(id="c2", name="echo", arguments='{"text": "b"}'),
        ]),
        ChatResponse(content="ok"),
    ])
    agent = make_agent(config, skills, workspace, client)
    result = await agent.run("twice")
    assert result == "ok"
    tool_msgs = [m for m in agent.context.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 2


async def test_compaction_does_not_loop_when_still_over_limit(config, skills, workspace):
    client = FakeClient([ChatResponse(content="ok")])
    events = FakeEvents()
    agent = make_agent(config, skills, workspace, client, events)
    compact_calls = {"n": 0}

    async def fake_compact(*, force: bool = False) -> bool:
        compact_calls["n"] += 1
        if compact_calls["n"] > 5:
            raise AssertionError("compaction loop")
        return True

    agent.compact = fake_compact  # type: ignore[method-assign]
    agent.context.needs_compaction = lambda: True  # type: ignore[method-assign]
    result = await agent.run("x")
    assert result == "ok"
    assert compact_calls["n"] == 1
    assert any("compaction" in text.lower() for _, text in events.notices)


async def test_max_tokens_capped_to_remaining_context(config, skills, workspace):
    class CaptureClient(FakeClient):
        async def chat(self, messages, tools=None, *, temperature, max_tokens, on_event=None,
                       sampling=None):
            self.calls.append(max_tokens)
            return self.responses.pop(0)

    config.llm.context_window = 1000
    config.llm.max_output_tokens = 1000
    client = CaptureClient([ChatResponse(content="ok")])
    agent = make_agent(config, skills, workspace, client)
    await agent.run("hi")
    sent = client.calls[0]
    assert 256 <= sent < config.llm.context_window


async def test_iteration_limit_stops_tool_loop(config, skills, workspace):
    config.llm.max_tool_iterations = 1
    client = FakeClient([
        ChatResponse(tool_calls=[ToolCall(id=f"c{i}", name="echo", arguments='{"text": "x"}')])
        for i in range(10)
    ])
    events = FakeEvents()
    agent = make_agent(config, skills, workspace, client, events)
    result = await agent.run("loop")
    assert result == ""
    assert len(client.calls) == 3  # tool, tool (limit hit), no-tools refusal
    assert any("limit" in text.lower() for _, text in events.notices)


def test_large_agents_md_does_not_force_compaction(config, skills, workspace):
    (workspace / "AGENTS.md").write_text("x" * 70_000, encoding="utf-8")
    context = ConversationContext(config, None, skills)
    assert context.needs_compaction() is False


def test_system_prompt_cached_and_invalidated(config, skills, workspace):
    context = ConversationContext(config, None, skills)
    first = context.system_prompt()
    assert context.system_prompt() is first
    context.invalidate()
    assert context.system_prompt() is not first


def test_sent_chars_matches_manual(config, skills, workspace):
    context = ConversationContext(config, None, skills)
    context.add_user("hello")
    context.add_assistant(ChatResponse(content="response"))
    manual = len(json.dumps({"role": "system", "content": context.system_prompt()},
                            ensure_ascii=False))
    manual += sum(len(json.dumps(m, ensure_ascii=False)) for m in context.messages)
    assert context.sent_chars() == manual


def test_close_dangling_tool_calls(config, skills, workspace):
    context = ConversationContext(config, None, skills)
    context.add_user("x")
    context.add_assistant(ChatResponse(tool_calls=[ToolCall(id="c1", name="echo",
                                                           arguments="{}")]))
    context.close_dangling_tool_calls("cancelled")
    tools = [m for m in context.messages if m.get("role") == "tool"]
    assert len(tools) == 1 and tools[0]["tool_call_id"] == "c1"


def test_tool_guide_off_by_default(config, skills, workspace):
    context = ConversationContext(config, None, skills)
    assert "Tool reference" not in context.system_prompt()


def test_tool_guide_injected_when_enabled(config, skills, workspace):
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(EditFileTool())
    guide = build_tool_guide(registry.schemas())
    context = ConversationContext(config, None, skills, tool_guide=guide)
    prompt = context.system_prompt()
    assert "Tool reference" in prompt
    assert "read_file" in prompt and "edit_file" in prompt
    assert "old_text" in prompt


def test_system_md_replaces_base_prompt(config, skills, workspace):
    (workspace / ".aisha").mkdir(exist_ok=True)
    (workspace / ".aisha" / "SYSTEM.md").write_text("You are a custom assistant.", encoding="utf-8")
    context = ConversationContext(config, None, skills)
    prompt = context.system_prompt()
    assert prompt.startswith("You are a custom assistant.")
    assert "You are Aisha" not in prompt


def test_system_md_still_appends_agents_md(config, skills, workspace):
    (workspace / ".aisha").mkdir(exist_ok=True)
    (workspace / ".aisha" / "SYSTEM.md").write_text("CUSTOM", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("AGENTS CONTENT", encoding="utf-8")
    context = ConversationContext(config, None, skills)
    prompt = context.system_prompt()
    assert prompt.startswith("CUSTOM")
    assert "AGENTS CONTENT" in prompt
    assert "## Project instructions" in prompt


async def test_silent_tool_skips_events(config, skills, workspace):
    class Silent(Tool):
        name = "silent"
        read_only = True
        silent = True
        parameters = {"type": "object", "properties": {}}

        async def run(self, args, ctx):
            return ToolResult.success(None, "ok")

    events = FakeEvents()
    registry = ToolRegistry()
    registry.register(Silent())
    context = ConversationContext(config, None, skills)
    tool_ctx = ToolContext(workspace=workspace, config=config, memory=None, skills=skills,
                           todos=context.todos, on_system_change=context.invalidate)
    client = FakeClient([
        ChatResponse(tool_calls=[ToolCall(id="c1", name="silent", arguments="{}")]),
        ChatResponse(content="done"),
    ])
    agent = AgentLoop(config, client, registry, context, tool_ctx, events)
    result = await agent.run("go")
    assert result == "done"
    assert events.starts == [] and events.ends == []
    tool_msgs = [m for m in context.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1


async def test_debug_emits_request_response_and_tool_dumps(config, skills, workspace):
    config.ui.debug = True
    client = FakeClient([
        ChatResponse(tool_calls=[ToolCall(id="c1", name="echo", arguments='{"text": "hi"}')]),
        ChatResponse(content="done", reasoning="thinking"),
    ])
    events = FakeEvents()
    agent = make_agent(config, skills, workspace, client, events)
    result = await agent.run("call echo")
    assert result == "done"
    titles = [title for title, _ in events.debugs]
    assert "→ model" in titles
    assert "← model" in titles
    assert any(title.startswith("tool:") for title in titles)
    request = [body for title, body in events.debugs if title == "→ model"][0]
    assert "user" in request and "call echo" in request
    response = [body for title, body in events.debugs if title == "← model"][-1]
    assert "thinking" in response
