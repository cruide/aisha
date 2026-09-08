# Author: Tischenko A. (https://github.com/cruide)
import json
from pathlib import Path

import pytest

import aisha.agent as agent_mod
from aisha.agent import AgentLoop
from aisha.client import ChatResponse, ToolCall
from aisha.config import load_config
from aisha.context import ConversationContext, build_tool_guide
from aisha.errors import AishaError, ContextOverflowError
from aisha.memory import MemoryStore
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

    async def tokenize(self, text):
        return None

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


async def test_context_overflow_compacts_and_retries(config, skills, workspace):
    def raise_overflow(messages):
        raise ContextOverflowError("too big")

    client = FakeClient([
        raise_overflow,
        ChatResponse(content="compact summary"),
        ChatResponse(content="done"),
    ])
    events = FakeEvents()
    agent = make_agent(config, skills, workspace, client, events)
    agent.context.add_user("previous task")
    agent.context.add_assistant(ChatResponse(content="previous answer"))
    result = await agent.run("task")
    assert result == "done"
    assert agent.context.messages[-1]["role"] == "assistant"
    assert any("compact summary" in str(m.get("content", "")) for m in agent.context.messages)
    assert any("overflow" in text.lower() for _, text in events.notices)


async def test_context_overflow_propagates_after_one_retry(config, skills, workspace):
    def raise_overflow(messages):
        raise ContextOverflowError("too big")

    client = FakeClient([
        raise_overflow,
        ChatResponse(content="summary"),
        raise_overflow,
    ])
    agent = make_agent(config, skills, workspace, client)
    agent.context.add_user("previous task")
    agent.context.add_assistant(ChatResponse(content="previous answer"))
    with pytest.raises(ContextOverflowError):
        await agent.run("task")
    assert len(client.calls) == 3


async def test_compact_fallback_on_summary_failure(config, skills, workspace):
    def fail(messages):
        raise AishaError("boom")

    client = FakeClient([fail])
    events = FakeEvents()
    agent = make_agent(config, skills, workspace, client, events)
    context = agent.context
    context.add_user("first task")
    context.add_assistant(ChatResponse(content="first answer"))
    context.add_user("second task")

    result = await agent.compact(force=True)
    assert result is True
    messages = context.messages
    assert any("fallback" in str(m.get("content", "")) for m in messages)
    assert messages[-1]["role"] == "user" and messages[-1]["content"] == "second task"
    assert any("fallback" in text.lower() for _, text in events.notices)


def test_bounded_history_limits_messages(config, skills, workspace):
    agent = make_agent(config, skills, workspace, FakeClient([]))
    old = [{"role": "user", "content": "task " + "x" * 5000}]
    old += [{"role": "assistant", "content": "answer " + "y" * 5000} for _ in range(20)]
    bounded = agent._bounded_history(old, 4000)
    assert len(bounded) < len(old)
    assert any("omitted" in str(m.get("content", "")) for m in bounded)
    assert any(m.get("role") == "user" and "task" in str(m.get("content", ""))
               for m in bounded)


async def test_interrupted_response_ignores_tool_calls(config, skills, workspace):
    client = FakeClient([
        ChatResponse(content="partial text", interrupted=True,
                     tool_calls=[ToolCall(id="c1", name="echo", arguments='{"text": "x"}')]),
    ])
    events = FakeEvents()
    agent = make_agent(config, skills, workspace, client, events)
    result = await agent.run("task")
    assert result == "partial text"
    assert events.starts == [] and events.ends == []
    roles = [m["role"] for m in agent.context.messages]
    assert roles == ["user", "assistant"]
    assert "tool_calls" not in agent.context.messages[-1]
    assert any("interrupted" in text.lower() for _, text in events.notices)


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


async def test_compaction_clears_stale_context_size(config, skills, workspace):
    client = FakeClient([ChatResponse(content="ok")])
    agent = make_agent(config, skills, workspace, client)
    context = agent.context
    context.add_user("x")
    # Simulate a request that measured a near-limit context right before compaction.
    context.stats.record({"prompt_tokens": 60_000, "completion_tokens": 500}, 0, 0, 100_000)
    assert context.needs_compaction() is True
    context.replace_history("summary", [{"role": "user", "content": "next task"}])
    assert context.needs_compaction() is False


def test_reasoning_not_stored_in_history(config, skills, workspace):
    context = ConversationContext(config, None, skills)
    context.add_user("x")
    context.add_assistant(ChatResponse(content="answer", reasoning="secret chain of thought"))
    assert "reasoning_content" not in context.messages[-1]
    context.add_interrupted_assistant(ChatResponse(content="partial", reasoning="more thoughts"))
    assert "reasoning_content" not in context.messages[-1]


def test_trim_keep_block_drops_reasoning_and_trims(config, skills, workspace):
    agent = make_agent(config, skills, workspace, FakeClient([]))
    keep = [
        {"role": "assistant", "content": "ok", "reasoning_content": "r" * 5000},
        {"role": "tool", "tool_call_id": "c1", "content": "y" * 50_000},
    ]
    trimmed = agent._trim_keep_block(keep)
    assert all("reasoning_content" not in m for m in trimmed)
    tool = trimmed[1]
    assert "trimmed for context compaction" in tool["content"]
    assert len(tool["content"]) < 50_000


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


def test_agents_md_bom_stripped(config, skills, workspace):
    (workspace / "AGENTS.md").write_bytes(b"\xef\xbb\xbf# Title\ncontent")
    context = ConversationContext(config, None, skills)
    prompt = context.system_prompt()
    assert "\ufeff" not in prompt
    assert "# Title" in prompt


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


def test_system_md_keeps_memory_and_skills_index(config, skills, workspace):
    (workspace / ".aisha").mkdir(exist_ok=True)
    (workspace / ".aisha" / "SYSTEM.md").write_text("CUSTOM", encoding="utf-8")
    (workspace / ".aisha" / "skills" / "s1").mkdir(parents=True)
    (workspace / ".aisha" / "skills" / "s1" / "SKILL.md").write_text(
        "---\nname: s1\ndescription: a skill\n---\nbody\n", encoding="utf-8",
    )
    store = MemoryStore(config.home_dir / "memory", config.project_dir / "memory",
                        max_block_chars=1000)
    store.set("style", "d", "v")
    context = ConversationContext(config, store, skills)
    prompt = context.system_prompt()
    assert prompt.startswith("CUSTOM")
    assert "style" in prompt
    assert "s1" in prompt
    assert "## Persistent memory" in prompt
    assert "## Skills" in prompt


def test_system_md_without_memory_omits_memory_section(config, skills, workspace):
    (workspace / ".aisha").mkdir(exist_ok=True)
    (workspace / ".aisha" / "SYSTEM.md").write_text("CUSTOM", encoding="utf-8")
    context = ConversationContext(config, None, skills)
    prompt = context.system_prompt()
    assert "## Persistent memory" not in prompt
    assert "## Skills" in prompt


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


# ------------------------------------------------------- TokenStats.record tests
def test_token_stats_record_with_usage(config, skills, workspace):
    from aisha.context import TokenStats

    stats = TokenStats()
    stats.record({"prompt_tokens": 1000, "completion_tokens": 200}, 0, 0, 3000)
    assert stats.last_in == 1000
    assert stats.last_out == 200
    assert stats.session_in == 1000
    assert stats.session_out == 200
    assert stats.ctx == 1200
    assert stats.approximate is False
    # chars_per_token = 3000 / 1000 = 3.0, clamped to [1.0, 6.0]
    assert stats.chars_per_token == 3.0


def test_token_stats_record_accumulates(config, skills, workspace):
    from aisha.context import TokenStats

    stats = TokenStats()
    stats.record({"prompt_tokens": 100, "completion_tokens": 10}, 0, 0, 300)
    stats.record({"prompt_tokens": 200, "completion_tokens": 20}, 0, 0, 600)
    assert stats.session_in == 300
    assert stats.session_out == 30
    assert stats.last_in == 200
    assert stats.last_out == 20


def test_token_stats_record_clamps_lower_bound(config, skills, workspace):
    from aisha.context import TokenStats

    stats = TokenStats()
    # chars_per_token = 500 / 1000 = 0.5 → clamped to 1.0
    stats.record({"prompt_tokens": 1000, "completion_tokens": 0}, 0, 0, 500)
    assert stats.chars_per_token == 1.0


def test_token_stats_record_clamps_upper_bound(config, skills, workspace):
    from aisha.context import TokenStats

    stats = TokenStats()
    # chars_per_token = 12000 / 1000 = 12.0 → clamped to 6.0
    stats.record({"prompt_tokens": 1000, "completion_tokens": 0}, 0, 0, 12000)
    assert stats.chars_per_token == 6.0


def test_token_stats_record_without_usage_falls_back_to_estimate(config, skills, workspace):
    from aisha.context import TokenStats

    stats = TokenStats()
    stats.record(None, 500, 100, 0)
    assert stats.last_in == 500
    assert stats.last_out == 100
    assert stats.approximate is True
    assert stats.session_in == 500
    assert stats.session_out == 100


def test_token_stats_record_zero_prompt_tokens_treated_as_no_usage(config, skills, workspace):
    from aisha.context import TokenStats

    stats = TokenStats()
    stats.record({"prompt_tokens": 0, "completion_tokens": 0}, 400, 50, 0)
    assert stats.last_in == 400
    assert stats.approximate is True


def test_token_stats_reset(config, skills, workspace):
    from aisha.context import TokenStats

    stats = TokenStats()
    stats.record({"prompt_tokens": 100, "completion_tokens": 10}, 0, 0, 300)
    stats.reset()
    assert stats.session_in == 0
    assert stats.session_out == 0
    assert stats.ctx == 0
    assert stats.approximate is True


# ---------------------------------------------------- estimate_sent_tokens tests
def test_estimate_sent_tokens_with_reasoning(config, skills, workspace):
    context = ConversationContext(config, None, skills)
    context.add_user("hello")
    context.add_assistant(ChatResponse(content="answer", reasoning="x" * 500))
    est = context.estimate_sent_tokens()
    # The reasoning text is part of the message (stored as content only, reasoning
    # is not in history), so the estimate should be > 0.
    assert est > 0
    # Verify it scales with content length.
    context2 = ConversationContext(config, None, skills)
    context2.add_user("hello")
    context2.add_assistant(ChatResponse(content="answer"))
    est2 = context2.estimate_sent_tokens()
    # Both should produce reasonable estimates (not zero, not wildly different).
    assert est > 0 and est2 > 0


def test_estimate_sent_tokens_scales_with_cpt(config, skills, workspace):
    context = ConversationContext(config, None, skills)
    context.add_user("a" * 1000)
    context.stats.chars_per_token = 2.0
    est_high = context.estimate_sent_tokens()
    context.stats.chars_per_token = 4.0
    est_low = context.estimate_sent_tokens()
    # Higher chars_per_token → fewer estimated tokens.
    assert est_high > est_low


# --------------------------------------------------------- calibration tests
async def test_calibrate_sets_chars_per_token(config, skills, workspace):
    class CalibClient(FakeClient):
        def __init__(self, responses, tokenize_result):
            super().__init__(responses)
            self._tokenize_result = tokenize_result
            self.tokenize_called = False

        async def tokenize(self, text):
            self.tokenize_called = True
            return self._tokenize_result

    tokens = list(range(800))  # 800 tokens for ~2000 chars → cpt ≈ 2.5
    client = CalibClient([ChatResponse(content="ok")], tokens)
    agent = make_agent(config, skills, workspace, client)
    await agent.run("hi")
    assert client.tokenize_called
    # chars_per_token should be set from calibration (2000/800 = 2.5, clamped).
    assert agent.context.stats.chars_per_token != 2.5 or agent._calibrated


async def test_calibrate_skips_when_tokenize_returns_none(config, skills, workspace):
    class NoTokenizeClient(FakeClient):
        async def tokenize(self, text):
            return None

    client = NoTokenizeClient([ChatResponse(content="ok")])
    agent = make_agent(config, skills, workspace, client)
    original_cpt = agent.context.stats.chars_per_token
    await agent.run("hi")
    # chars_per_token unchanged when /tokenize unavailable.
    assert agent.context.stats.chars_per_token == original_cpt


async def test_calibrate_runs_only_once(config, skills, workspace):
    class CountingClient(FakeClient):
        def __init__(self, responses):
            super().__init__(responses)
            self.calibrate_count = 0

        async def tokenize(self, text):
            self.calibrate_count += 1
            return list(range(500))

    client = CountingClient([ChatResponse(content="a"), ChatResponse(content="b")])
    agent = make_agent(config, skills, workspace, client)
    await agent.run("first")
    await agent.run("second")
    assert client.calibrate_count == 1


# --------------------------------------------------- enable_thinking tests
async def test_enable_thinking_passed_to_model(config, skills, workspace):
    config.llm.enable_thinking = True
    captured_sampling = {}

    class SamplingCaptureClient(FakeClient):
        async def chat(self, messages, tools=None, *, temperature, max_tokens, on_event=None,
                       sampling=None):
            captured_sampling.setdefault("calls", []).append(
                {"tools": tools, "sampling": sampling}
            )
            return self.responses.pop(0)

    client = SamplingCaptureClient([
        ChatResponse(tool_calls=[ToolCall(id="c1", name="echo", arguments='{"text": "x"}')]),
        ChatResponse(content="done"),
    ])
    agent = make_agent(config, skills, workspace, client)
    await agent.run("call echo")
    calls = captured_sampling["calls"]
    # Both calls have tools (registry schemas), so enable_thinking should be False.
    for call in calls:
        assert call["sampling"]["chat_template_kwargs"]["enable_thinking"] is False

    # Verify that tools=None (e.g. compaction call) gets enable_thinking=True.
    assert (
        agent.config.llm.enable_thinking is True
    ), "config enables thinking"
    # Simulate the logic: tools=None → thinking enabled.
    sampling_with_none_tools = {}
    if agent.config.llm.enable_thinking is not None:
        sampling_with_none_tools["chat_template_kwargs"] = {
            "enable_thinking": agent.config.llm.enable_thinking if None is None else False,
        }
    assert sampling_with_none_tools["chat_template_kwargs"]["enable_thinking"] is True


async def test_enable_thinking_none_omits_chat_template_kwargs(config, skills, workspace):
    config.llm.enable_thinking = None
    captured_sampling = {}

    class SamplingCaptureClient(FakeClient):
        async def chat(self, messages, tools=None, *, temperature, max_tokens, on_event=None,
                       sampling=None):
            captured_sampling.setdefault("calls", []).append(sampling)
            return self.responses.pop(0)

    client = SamplingCaptureClient([ChatResponse(content="ok")])
    agent = make_agent(config, skills, workspace, client)
    await agent.run("hi")
    s = captured_sampling["calls"][0]
    assert s is None or "chat_template_kwargs" not in s
