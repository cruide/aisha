# Author: Tischenko A. (https://github.com/cruide)
"""AgentLoop: model -> tool calls -> tool results -> model, with limits and compaction."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Protocol

from aisha.client import ChatResponse, LlamaClient, ToolCall
from aisha.config import Config
from aisha.context import ConversationContext
from aisha.errors import AishaError
from aisha.tools.base import ToolContext, ToolRegistry, ToolResult

# Read-only tools without side effects: consecutive calls run concurrently.
PARALLEL_TOOLS = frozenset({
    "read_file", "list_dir", "glob", "grep", "web_search", "web_fetch", "memory_get", "memory_list",
})

SUMMARY_SYSTEM = (
    "Ты сжимаешь историю диалога AI-агента для программиста. Составь структурированную сводку "
    "на русском: 1) цель пользователя; 2) что уже сделано (файлы, команды, результаты); "
    "3) важные факты и решения; 4) незавершённые задачи и следующие шаги. Без воды, без "
    "инструментов, только текст."
)
SUMMARY_REQUEST = "Сделай сводку диалога выше по указанной структуре."


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

    # ------------------------------------------------------------------ turn
    async def run(self, user_text: str) -> str:
        self.context.add_user(user_text)
        try:
            return await self._run_turn()
        except BaseException:
            # Keep history valid if we were interrupted between tool_calls and results.
            self.context.close_dangling_tool_calls("Операция прервана пользователем.")
            raise

    async def _run_turn(self) -> str:
        llm = self.config.llm
        iterations = 0
        limit_hit = False
        skip_compact = False
        while True:
            if not skip_compact and self.context.needs_compaction():
                await self.compact()
                if self.context.needs_compaction():
                    skip_compact = True
                    self.events.on_notice(
                        "Сжатие не освободило достаточно контекста; "
                        "продолжаю без повторной попытки.",
                        "warn",
                    )
            tools = None if limit_hit else self.registry.schemas(read_only=self.config.read_only)
            response = await self._call_model(tools)
            self.context.add_assistant(response)
            if response.finish_reason == "length":
                self.events.on_notice("Ответ обрезан: достигнут лимит max_output_tokens.", "warn")
            if not response.tool_calls:
                return response.content
            if limit_hit:
                self._refuse_calls(response.tool_calls, "Инструменты недоступны: лимит исчерпан.")
                return response.content
            iterations += 1
            if iterations > llm.max_tool_iterations:
                self.events.on_notice(
                    f"Достигнут лимит итераций инструментов ({llm.max_tool_iterations}); "
                    "запрашиваю итоговый ответ.", "warn",
                )
                self._refuse_calls(
                    response.tool_calls,
                    "Лимит итераций инструментов исчерпан. Сформируй итоговый ответ для "
                    "пользователя без новых вызовов инструментов.",
                )
                limit_hit = True
                continue
            await self._execute_calls(response.tool_calls)
            skip_compact = False

    def _refuse_calls(self, calls: list[ToolCall], message: str) -> None:
        for call in calls:
            result = ToolResult.failure("IterationLimit", message)
            self.context.add_tool_result(call.id, call.name, result.to_json())

    async def _call_model(self, tools: list[dict[str, Any]] | None) -> ChatResponse:
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
        remaining = llm.context_window - est_in
        max_tokens = max(256, min(llm.max_output_tokens, remaining))
        if self.config.ui.debug:
            self.events.on_debug("→ model", self._format_request(messages, est_in))
        self.events.on_stream_start()
        response = await self.client.chat(
            messages, tools, temperature=llm.temperature,
            max_tokens=max_tokens, on_event=self._on_event,
            sampling=sampling or None,
        )
        produced = response.content + response.reasoning + "".join(
            c.arguments for c in response.tool_calls
        )
        self.context.stats.record(response.usage, est_in, self.context.estimate_text(produced),
                                  chars_in)
        self.events.on_stream_end(response)
        if self.config.ui.debug:
            self.events.on_debug("← model", self._format_response(response))
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
        lines = [f"сообщений: {len(messages)}, ~{est_tokens} токенов"]
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
        return "\n".join(parts) or "(пусто)"

    # ----------------------------------------------------------------- tools
    async def _execute_calls(self, calls: list[ToolCall]) -> None:
        i = 0
        while i < len(calls):
            if calls[i].name in PARALLEL_TOOLS:
                j = i
                while j < len(calls) and calls[j].name in PARALLEL_TOOLS:
                    j += 1
                await asyncio.gather(*(self._run_call(c) for c in calls[i:j]))
                i = j
            else:
                await self._run_call(calls[i])
                i += 1

    async def _run_call(self, call: ToolCall) -> None:
        tool = self.registry.get(call.name)
        silent = bool(tool and tool.silent)
        try:
            args = call.parse_arguments()
        except ValueError as exc:
            if not silent:
                self.events.on_tool_start(call, None)
            result = ToolResult.failure("ToolValidationError",
                                        f"Некорректный JSON аргументов: {exc}")
        else:
            if not silent:
                self.events.on_tool_start(call, args)
            result = await self.registry.execute(call.name, args, self.tool_ctx)
        if not silent:
            self.events.on_tool_end(call, result)
        self.context.add_tool_result(call.id, call.name, result.to_json())
        if self.config.ui.debug and not silent:
            self.events.on_debug(f"tool: {call.name}", self._clip(result.to_json(), 2000))

    # ------------------------------------------------------------ compaction
    async def compact(self, *, force: bool = False) -> bool:
        blocks = self.context.turn_blocks()
        if len(blocks) < 2:
            if force:
                self.events.on_notice("История слишком короткая, сжимать нечего.")
            return False
        old = [m for block in blocks[:-1] for m in block]
        keep = blocks[-1]
        self.events.on_notice("Сжимаю историю диалога…")
        summary: str | None = None
        try:
            response = await self.client.chat(
                [{"role": "system", "content": SUMMARY_SYSTEM}, *old,
                 {"role": "user", "content": SUMMARY_REQUEST}],
                None, temperature=0.1, max_tokens=2048,
            )
            summary = response.content.strip() or None
        except AishaError as exc:
            self.events.on_notice(f"Сводка не удалась ({exc}); старые сообщения удалены.", "warn")
        self.context.replace_history(summary, keep)
        self.events.on_notice(
            f"История сжата: {len(old)} сообщений → {'сводка' if summary else 'удалены'}."
        )
        return True
