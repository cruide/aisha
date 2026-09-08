# Author: Tischenko A. (https://github.com/cruide)
"""AgentLoop: model -> tool calls -> tool results -> model, with limits and compaction."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Protocol

from aisha.client import ChatResponse, LlamaClient, ToolCall
from aisha.config import Config
from aisha.context import ConversationContext
from aisha.errors import AishaError, ContextOverflowError
from aisha.logger import debug_logger
from aisha.tools.base import ToolContext, ToolRegistry, ToolResult

# Read-only tools without side effects: consecutive calls run concurrently.
PARALLEL_TOOLS = frozenset({
    "read_file", "list_dir", "glob", "grep", "web_search", "web_fetch", "memory_get", "memory_list",
})

SUMMARY_SYSTEM = (
    "You are compressing the conversation history of an AI agent for a developer. "
    "Produce a structured summary: 1) user goal; 2) what was done (files, commands, results); "
    "3) important facts and decisions; 4) unfinished tasks and next steps. No fluff, no "
    "tool calls, plain text only."
)
SUMMARY_REQUEST = "Summarise the conversation above following the structure described."
SUMMARY_OUTPUT_TOKENS = 4096


class AgentEvents(Protocol):
    def on_stream_start(self) -> None: ...
    def on_text(self, delta: str) -> None: ...
    def on_reasoning(self, delta: str) -> None: ...
    def on_stream_end(self, response: ChatResponse) -> None: ...
    def on_tool_start(self, call: ToolCall, args: dict[str, Any] | None) -> None: ...
    def on_tool_end(self, call: ToolCall, result: ToolResult) -> None: ...
    def on_notice(self, text: str, level: str = "info") -> None: ...
    def on_debug(self, title: str, body: str) -> None: ...


class AgentLoop:
    def __init__(
        self,
        config: Config,
        client: LlamaClient,
        registry: ToolRegistry,
        context: ConversationContext,
        tool_ctx: ToolContext,
        events: AgentEvents,
    ) -> None:
        self.config = config
        self.client = client
        self.registry = registry
        self.context = context
        self.tool_ctx = tool_ctx
        self.events = events
        self._calibrated = False

    # ------------------------------------------------------------------ turn
    async def run(self, user_text: str) -> str:
        self.context.add_user(user_text)
        try:
            return await self._run_turn()
        except BaseException:
            # Keep history valid if we were interrupted between tool_calls and results.
            self.context.close_dangling_tool_calls("Operation interrupted by user.")
            raise

    async def _calibrate(self) -> None:
        """Calibrate chars_per_token via the server's /tokenize endpoint (once)."""
        if self._calibrated:
            return
        self._calibrated = True
        snippet = self.context.system_prompt()
        # Use a representative slice (up to 2000 chars) to avoid huge payloads.
        sample = snippet[:2000]
        if not sample:
            return
        tokens = await self.client.tokenize(sample)
        if tokens and len(tokens) > 0:
            cpt = len(sample) / len(tokens)
            # Clamp to the same bounds used by TokenStats.record.
            cpt = max(1.0, min(6.0, cpt))
            self.context.stats.chars_per_token = cpt
            self.events.on_notice(
                f"Tokeniser calibrated: {cpt:.2f} chars/token "
                f"({len(sample)} chars → {len(tokens)} tokens).",
            )

    async def _run_turn(self) -> str:
        llm = self.config.llm
        iterations = 0
        limit_hit = False
        skip_compact = False
        overflow_retried = False
        while True:
            if not skip_compact and self.context.needs_compaction():
                await self.compact()
                if self.context.needs_compaction():
                    skip_compact = True
                    self.events.on_notice(
                        "Compaction did not free enough context; "
                        "continuing without retrying.",
                        "warn",
                    )
            tools = None if limit_hit else self.registry.schemas(read_only=self.config.read_only)
            try:
                response = await self._call_model(tools)
            except ContextOverflowError:
                if overflow_retried:
                    self.events.on_notice(
                        "Context still exceeds the model limit after compaction. "
                        "Start a new session (/new) or reduce the task size.",
                        "warn",
                    )
                    raise
                overflow_retried = True
                self.events.on_notice(
                    "Context overflow; compacting conversation history before retrying.",
                    "warn",
                )
                if not await self.compact(force=True):
                    raise
                continue
            if response.interrupted:
                self.context.add_interrupted_assistant(response)
                self.events.on_notice(
                    "Response interrupted; partial text was saved. Tool calls were ignored.",
                    "warn",
                )
                return response.content
            self.context.add_assistant(response)
            if response.finish_reason == "length":
                if response.tool_calls:
                    self.events.on_notice(
                        "Response truncated: max_output_tokens limit reached; tool calls "
                        "may be incomplete.", "warn",
                    )
                else:
                    self.events.on_notice(
                        "Response truncated: max_output_tokens limit reached.", "warn",
                    )
            if not response.tool_calls:
                return response.content
            if limit_hit:
                self._refuse_calls(
                    response.tool_calls, "Tools unavailable: iteration limit reached."
                )
                return response.content
            iterations += 1
            if iterations > llm.max_tool_iterations:
                self.events.on_notice(
                    f"Tool iteration limit reached ({llm.max_tool_iterations}); "
                    "requesting final answer.", "warn",
                )
                self._refuse_calls(
                    response.tool_calls,
                    "Tool iteration limit reached. Provide a final answer to "
                    "the user without new tool calls.",
                )
                limit_hit = True
                continue
            await self._execute_calls(response.tool_calls,
                                      truncated=response.finish_reason == "length")
            skip_compact = False

    def _refuse_calls(self, calls: list[ToolCall], message: str) -> None:
        for call in calls:
            result = ToolResult.failure("IterationLimit", message)
            self.context.add_tool_result(call.id, call.name, result.to_json())

    async def _call_model(self, tools: list[dict[str, Any]] | None) -> ChatResponse:
        if not self._calibrated:
            await self._calibrate()
        llm = self.config.llm
        messages = self.context.all_messages()
        chars_in = self.context.sent_chars()
        est_in = self.context.estimate_sent_tokens()
        sampling = {
            key: value for key, value in (
                ("top_p", llm.top_p),
                ("top_k", llm.top_k),
                ("repeat_penalty", llm.repeat_penalty),
                ("frequency_penalty", llm.frequency_penalty),
            ) if value is not None
        }
        # Per-request thinking control for Qwen-style models.
        # When enable_thinking is configured, disable thinking on tool-calling
        # turns (mechanical work) and enable it on final-answer turns.
        if llm.enable_thinking is not None:
            sampling["chat_template_kwargs"] = {
                "enable_thinking": llm.enable_thinking if tools is None else False,
            }
        # est_in already includes the system prompt and the message history. Only
        # the tool schemas (sent with every request) and the chat-template special
        # tokens are missing from it. Reserve space for those plus a safety margin
        # for estimate error, otherwise max_tokens overshoots and the server
        # truncates tool-call JSON mid-stream.
        cpt = self.context.stats.chars_per_token
        overhead = int(len(json.dumps(tools, ensure_ascii=False)) / cpt) if tools else 0
        overhead += max(512, est_in // 20)
        remaining = llm.context_window - est_in - overhead
        max_tokens = max(256, min(llm.max_output_tokens, remaining))
        if self.config.ui.debug:
            self.events.on_debug("→ model", self._format_request(messages, est_in))
        if debug_logger.path:
            debug_logger.log_request(
                {
                    "model": self.client.model,
                    "messages": messages,
                    "tools": tools,
                    "temperature": llm.temperature,
                    "max_tokens": max_tokens,
                    **(sampling or {}),
                },
                est_tokens=est_in,
                message_count=len(messages),
            )
        self.events.on_stream_start()
        try:
            response = await self.client.chat(
                messages, tools, temperature=llm.temperature,
                max_tokens=max_tokens, on_event=self._on_event,
                sampling=sampling or None,
            )
        except BaseException:
            self.events.on_stream_end(ChatResponse())
            raise
        produced = response.content + response.reasoning + "".join(
            c.arguments for c in response.tool_calls
        )
        self.context.stats.record(response.usage, est_in, self.context.estimate_text(produced),
                                  chars_in)
        self.events.on_stream_end(response)
        if self.config.ui.debug:
            self.events.on_debug("← model", self._format_response(response))
        if debug_logger.path:
            debug_logger.log_response(
                response,
                produced_chars=len(produced),
            )
        return response

    def _on_event(self, kind: str, delta: str) -> None:
        if kind == "text":
            self.events.on_text(delta)
        elif kind == "reasoning":
            self.events.on_reasoning(delta)

    @staticmethod
    def _clip(text: str, limit: int) -> str:
        text = text.strip()
        return text if len(text) <= limit else text[:limit] + "…"

    def _format_request(self, messages: list[dict[str, Any]], est_tokens: int) -> str:
        lines = [f"messages: {len(messages)}, ~{est_tokens} tokens"]
        for m in messages:
            role = m.get("role")
            if m.get("tool_calls"):
                names = [c.get("function", {}).get("name", "?") for c in m["tool_calls"]]
                lines.append(f"  {role}: [tool_calls] {', '.join(names)}")
                continue
            content = m.get("content")
            body = content if isinstance(content, str) else json.dumps(content,
                                                                       ensure_ascii=False)
            lines.append(f"  {role}: {self._clip(body, 240)}")
        return "\n".join(lines)

    def _format_response(self, response: ChatResponse) -> str:
        parts: list[str] = []
        if response.reasoning:
            parts.append(f"reasoning: {self._clip(response.reasoning, 600)}")
        if response.content:
            parts.append(f"content: {self._clip(response.content, 600)}")
        for c in response.tool_calls:
            parts.append(f"tool_call: {c.name}({self._clip(c.arguments, 300)})")
        if response.finish_reason:
            parts.append(f"finish_reason: {response.finish_reason}")
        if response.usage:
            parts.append(f"usage: {response.usage}")
        return "\n".join(parts) or "(empty)"

    # ----------------------------------------------------------------- tools
    async def _execute_calls(self, calls: list[ToolCall], *, truncated: bool = False) -> None:
        i = 0
        while i < len(calls):
            if calls[i].name in PARALLEL_TOOLS:
                j = i
                while j < len(calls) and calls[j].name in PARALLEL_TOOLS:
                    j += 1
                await asyncio.gather(*(self._run_call(c, truncated=truncated) for c in calls[i:j]))
                i = j
            else:
                await self._run_call(calls[i], truncated=truncated)
                i += 1

    async def _run_call(self, call: ToolCall, *, truncated: bool = False) -> None:
        tool = self.registry.get(call.name)
        silent = bool(tool and tool.silent)
        try:
            args = call.parse_arguments()
        except ValueError as exc:
            if not silent:
                self.events.on_tool_start(call, None)
            message = f"Invalid arguments JSON: {exc}"
            if truncated:
                message += (
                    ". Model output was truncated by the max_tokens limit. Break "
                    "the content into smaller parts and retry: skeleton via write_file, then "
                    "extend with edit_file or additional write_file calls."
                )
            result = ToolResult.failure("ToolValidationError", message)
        else:
            if not silent:
                self.events.on_tool_start(call, args)
            if debug_logger.path:
                debug_logger.log_tool_call(call.name, args, call.id)
            result = await self.registry.execute(call.name, args, self.tool_ctx)
        if not silent:
            self.events.on_tool_end(call, result)
        self.context.add_tool_result(call.id, call.name, result.to_json())
        if debug_logger.path:
            debug_logger.log_tool_result(call.name, call.id, result.to_json(), ok=result.ok)
        if self.config.ui.debug and not silent:
            self.events.on_debug(f"tool: {call.name}", self._clip(result.to_json(), 2000))

    # ------------------------------------------------------------ compaction
    _KEEP_TOOL_HEAD = 2000
    _KEEP_TOOL_TAIL = 1000
    _KEEP_BLOCK_FRACTION = 0.25

    async def compact(self, *, force: bool = False) -> bool:
        blocks = self.context.turn_blocks()
        if len(blocks) < 2:
            if force:
                return await self._aggressive_compact(blocks[0] if blocks else [])
            return False
        old = [m for block in blocks[:-1] for m in block]
        keep = blocks[-1]
        self.events.on_notice("Compacting conversation history…")
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM},
            *self._summary_input(old),
            {"role": "user", "content": SUMMARY_REQUEST},
        ]
        summary: str | None = None
        try:
            if debug_logger.path:
                debug_logger.log_request(
                    {
                        "model": self.client.model,
                        "messages": messages,
                        "tools": None,
                        "temperature": 0.1,
                        "max_tokens": SUMMARY_OUTPUT_TOKENS,
                    },
                    est_tokens=0,
                    message_count=len(messages),
                )
            response = await self.client.chat(
                messages, None, temperature=0.1, max_tokens=SUMMARY_OUTPUT_TOKENS,
            )
            summary = response.content.strip() or None
            if debug_logger.path:
                debug_logger.log_response(response, produced_chars=len(response.content))
            # Record usage so chars_per_token stays accurate after compaction.
            if response.usage:
                chars_in = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
                self.context.stats.record(
                    response.usage, 0, self.context.estimate_text(response.content), chars_in,
                )
        except AishaError as exc:
            self.events.on_notice(
                f"Summary failed ({exc}); retained a compact fallback.", "warn",
            )
        if not summary:
            summary = self._fallback_summary(old)
        # Trim oversized tool-results inside the keep block so that the
        # compacted history actually fits in the context window.
        keep = self._trim_keep_block(keep)
        self.context.replace_history(summary, keep)
        self.events.on_notice(f"History compacted: {len(old)} messages → summary.")
        return True

    def _trim_keep_block(self, keep: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Trim the last-turn block so the compacted history fits comfortably.

        ``reasoning_content`` is dropped (the model must not replay its own
        chain-of-thought) and oversized tool results are reduced to a head + tail,
        earliest first, until the block fits ``_KEEP_BLOCK_FRACTION`` of the
        context window. This leaves room for the summary prefix and further work.
        """
        cpt = self.context.stats.chars_per_token
        budget_chars = int(self.config.llm.context_window * self._KEEP_BLOCK_FRACTION * cpt)
        for msg in keep:
            msg.pop("reasoning_content", None)
        total = sum(self.context._chars(m) for m in keep)
        if total <= budget_chars:
            return keep
        trimmed = False
        for msg in keep:
            if total <= budget_chars:
                break
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if len(content) <= self._KEEP_TOOL_HEAD + self._KEEP_TOOL_TAIL:
                continue
            msg["content"] = (
                content[:self._KEEP_TOOL_HEAD]
                + "\n[… trimmed for context compaction …]\n"
                + content[-self._KEEP_TOOL_TAIL:]
            )
            total -= len(content) - len(msg["content"])
            trimmed = True
        if trimmed:
            self.events.on_notice("Trimmed large tool results in the keep block.", "warn")
        return keep

    async def _aggressive_compact(self, block: list[dict[str, Any]]) -> bool:
        """Trim large tool-results within a single turn block.

        Used when ``force=True`` but the history contains only one block and
        normal compaction (which needs ≥2 blocks) is impossible.
        """
        trimmed = 0
        for msg in block:
            msg.pop("reasoning_content", None)
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if len(content) > 4000:
                msg["content"] = (
                    content[:2000]
                    + "\n[… trimmed for context compaction …]\n"
                    + content[-1000:]
                )
                trimmed += 1
        if trimmed:
            self.context.replace_history(None, block)
            self.events.on_notice(
                f"Aggressive compaction: trimmed {trimmed} tool result(s) in the current block.",
                "warn",
            )
            return True
        self.events.on_notice("History is too short, nothing to compact.")
        return False

    @staticmethod
    def _is_summary_message(msg: dict[str, Any]) -> bool:
        return (
            msg.get("role") == "user"
            and isinstance(msg.get("content"), str)
            and msg["content"].startswith("[Summary of the previous conversation]")
        )

    def _summary_input(self, old: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build the summarizer input, bounded to a conservative token budget."""
        llm = self.config.llm
        cpt = self.context.stats.chars_per_token
        fixed = [
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": SUMMARY_REQUEST},
        ]
        fixed_chars = sum(self.context._chars(m) for m in fixed)
        reserve = max(1024, llm.context_window // 32)
        budget_chars = max(
            1024,
            int((llm.context_window - reserve - SUMMARY_OUTPUT_TOKENS) * cpt) - fixed_chars,
        )
        return self._bounded_history(old, budget_chars)

    def _bounded_history(
        self, old: list[dict[str, Any]], budget_chars: int
    ) -> list[dict[str, Any]]:
        """Trim `old` to its critical fragments when it exceeds `budget_chars`."""
        if sum(self.context._chars(m) for m in old) <= budget_chars:
            return list(old)
        picked: dict[int, dict[str, Any]] = {}
        # Existing previous summary and its acknowledgement.
        for i, m in enumerate(old):
            if self._is_summary_message(m):
                picked[i] = m
                if i + 1 < len(old) and old[i + 1].get("role") == "assistant":
                    picked[i + 1] = old[i + 1]
        # The original task (first non-summary user message), clipped.
        for i, m in enumerate(old):
            if m.get("role") == "user" and i not in picked:
                picked[i] = {"role": "user",
                             "content": self._clip(str(m.get("content", "")), 2000)}
                break
        # The most recent user/assistant context.
        for i in range(len(old) - 1, -1, -1):
            if i in picked or old[i].get("role") not in ("user", "assistant"):
                continue
            picked[i] = old[i]
            if len(picked) >= 6:
                break
        result = [m for _, m in sorted(picked.items())]
        skipped = len(old) - len(picked)
        if skipped > 0:
            result.append({
                "role": "user",
                "content": f"[{skipped} earlier messages omitted from the summary input]",
            })
        return result

    def _fallback_summary(self, old: list[dict[str, Any]]) -> str:
        """Deterministic summary used when the summarizer model call fails or returns empty."""
        lines = ["Previous conversation (compressed fallback):"]
        previous = next((str(m["content"]) for m in old if self._is_summary_message(m)), None)
        if previous:
            lines.append(f"- Prior summary: {self._clip(previous, 800)}")
        first_user = next(
            (str(m["content"]) for m in old
             if m.get("role") == "user" and not self._is_summary_message(m)),
            None,
        )
        if first_user:
            lines.append(f"- Goal: {self._clip(first_user, 500)}")
        recent = [m for m in old if m.get("role") in ("user", "assistant")][-4:]
        for m in recent:
            content = m.get("content")
            if isinstance(content, str):
                lines.append(f"- {m['role']}: {self._clip(content, 300)}")
        lines.append(f"- ({len(old)} messages compressed)")
        return "\n".join(lines)
