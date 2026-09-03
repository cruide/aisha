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

    async def chat(self, messages, tools=None, *, temperature, max_tokens, on_event=None):
        self.calls.append((messages, tools))
        resp = self.responses.pop(0)
        return resp(messages) if callable(resp) else resp


class FakeEvents:
    def __init__(self):
        self.notices = []

    def on_stream_start(self):
        pass

    def on_text(self, delta):
        pass

    def on_reasoning(self, delta):
        pass

    def on_stream_end(self, response):
        pass

    def on_tool_start(self, call, args):
        pass

    def on_tool_end(self, call, result):
        pass

    def on_notice(self, text, level="info"):
        self.notices.append((level, text))


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
        ChatResponse(content="готово"),
    ])
    agent = make_agent(config, skills, workspace, client)
    result = await agent.run("позови echo")
    assert result == "готово"
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
    result = await agent.run("дважды")
    assert result == "ok"
    tool_msgs = [m for m in agent.context.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 2


async def test_iteration_limit_stops_tool_loop(config, skills, workspace):
    config.llm.max_tool_iterations = 1
    client = FakeClient([
        ChatResponse(tool_calls=[ToolCall(id=f"c{i}", name="echo", arguments='{"text": "x"}')])
        for i in range(10)
    ])
    events = FakeEvents()
    agent = make_agent(config, skills, workspace, client, events)
    result = await agent.run("зацикли")
    assert result == ""
    assert len(client.calls) == 3  # tool, tool (limit hit), no-tools refusal
    assert any("лимит" in text for _, text in events.notices)


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
    context.add_user("привет")
    context.add_assistant(ChatResponse(content="ответ"))
    manual = len(json.dumps({"role": "system", "content": context.system_prompt()},
                            ensure_ascii=False))
    manual += sum(len(json.dumps(m, ensure_ascii=False)) for m in context.messages)
    assert context.sent_chars() == manual


def test_close_dangling_tool_calls(config, skills, workspace):
    context = ConversationContext(config, None, skills)
    context.add_user("x")
    context.add_assistant(ChatResponse(tool_calls=[ToolCall(id="c1", name="echo",
                                                           arguments="{}")]))
    context.close_dangling_tool_calls("отменено")
    tools = [m for m in context.messages if m.get("role") == "tool"]
    assert len(tools) == 1 and tools[0]["tool_call_id"] == "c1"


def test_tool_guide_off_by_default(config, skills, workspace):
    context = ConversationContext(config, None, skills)
    assert "Справочник инструментов" not in context.system_prompt()


def test_tool_guide_injected_when_enabled(config, skills, workspace):
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(EditFileTool())
    guide = build_tool_guide(registry.schemas())
    context = ConversationContext(config, None, skills, tool_guide=guide)
    prompt = context.system_prompt()
    assert "Справочник инструментов" in prompt
    assert "read_file" in prompt and "edit_file" in prompt
    assert "old_text" in prompt
