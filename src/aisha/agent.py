"""Main agent loop: model interaction, tool calling, iteration limits."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable

from aisha.client import (
    ChatResponse,
    LlamaClient,
    StreamDelta,
)
from aisha.config import Config
from aisha.context import ContextManager
from aisha.tools.base import (
    Tool,
    ToolRegistry,
    ToolResult,
    ToolValidationError,
)

logger = logging.getLogger(__name__)

# Tools that can run in parallel (read-only, no side effects)
PARALLEL_TOOLS = frozenset({
    "read_file",
    "list_dir",
    "glob",
    "grep",
    "web_search",
    "web_fetch",
    "memory_get",
})


class AgentLoop:
    """Main agent loop handling model calls and tool execution."""

    def __init__(
        self,
        client: LlamaClient,
        config: Config,
        context: ContextManager,
        registry: ToolRegistry,
        ui_callback: Callable[[str, Any], Awaitable[None]] | None = None,
        tool_status_fn: Callable[[str, str, str], None] | None = None,
        ask_fn: Callable[[str, list[str] | None], Awaitable[str | None]] | None = None,
        confirm_fn: Callable[..., Awaitable[str | None]] | None = None,
        thinking_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.context = context
        self.registry = registry
        self.ui_callback = ui_callback
        self.tool_status_fn = tool_status_fn
        self.ask_fn = ask_fn
        self.confirm_fn = confirm_fn
        self.thinking_callback = thinking_callback
        self._cancelled = False

    def cancel(self) -> None:
        """Cancel current operation."""
        self._cancelled = True

    def reset_cancel(self) -> None:
        self._cancelled = False

    async def run(self, user_input: str) -> str:
        """Run agent loop for a single user input.

        Returns the final text response.
        """
        self.reset_cancel()
        self.context.add_user_message(user_input)
        self.context._load_agents_md()

        tools = self.registry.openai_tools()
        iteration = 0
        max_iter = self.config.llm.max_tool_iterations
        final_text = ""

        while iteration < max_iter:
            if self._cancelled:
                return "[Отменено]"

            messages = self.context.get_messages()

            try:
                stream = await self.client.chat(
                    messages,
                    tools=tools if tools else None,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_output_tokens,
                    stream=self.config.ui.stream,
                )
            except Exception as e:
                error_msg = f"Ошибка сервера: {e}"
                self.context.add_assistant_message(error_msg)
                return error_msg

            if self.config.ui.stream:
                response = await self._collect_stream(stream)
            else:
                response = stream if isinstance(stream, ChatResponse) else await stream

            # Update token counts
            if response.usage:
                self.context.update_token_counts(response.usage)

            # Store reasoning if present
            if response.reasoning and self.config.ui.show_reasoning:
                if self.thinking_callback:
                    self.thinking_callback(response.reasoning)

            # Handle tool calls
            if response.tool_calls:
                iteration += 1
                self.context.add_assistant_message(
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                )

                # Execute tool calls
                tool_results = await self._execute_tool_calls(response.tool_calls)

                # Add tool results to context
                for call_id, result_str in tool_results:
                    self.context.add_tool_result(call_id, result_str)

                # Continue loop for model to process results
                if response.content:
                    final_text = response.content
                continue
            else:
                # No tool calls — final text response
                if response.content:
                    final_text = response.content
                    self.context.add_assistant_message(content=response.content)
                break

        if iteration >= max_iter:
            # Notify model about iteration limit
            limit_msg = (
                f"Достигнут лимит итераций инструментов ({max_iter}). "
                "Пожалуйста, сформируйте итоговый ответ без вызова инструментов."
            )
            self.context.add_user_message(limit_msg)
            messages = self.context.get_messages()
            try:
                result = await self.client.chat(
                    messages, tools=None, stream=False
                )
                if isinstance(result, ChatResponse) and result.content:
                    final_text = result.content
                    self.context.add_assistant_message(content=result.content)
            except Exception:
                pass

        return final_text

    async def _collect_stream(self, stream: Any) -> ChatResponse:
        """Collect streaming response into a ChatResponse."""
        full_text = ""
        full_reasoning = ""
        tool_calls: list[dict] = []
        finish_reason = ""
        usage = {}

        async for delta in stream:
            if self._cancelled:
                break

            if isinstance(delta, StreamDelta):
                if delta.text:
                    full_text += delta.text
                    if self.ui_callback:
                        await self.ui_callback("text", delta.text)
                if delta.reasoning:
                    full_reasoning += delta.reasoning
                    if self.ui_callback:
                        await self.ui_callback("reasoning", delta.reasoning)
                if delta.tool_calls:
                    tool_calls = delta.tool_calls
                if delta.finish_reason:
                    finish_reason = delta.finish_reason
                if delta.usage:
                    usage = delta.usage

        return ChatResponse(
            content=full_text,
            reasoning=full_reasoning,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    async def _execute_tool_calls(
        self, tool_calls: list[dict]
    ) -> list[tuple[str, str]]:
        """Execute tool calls and return (call_id, result_json) pairs."""
        results: list[tuple[str, str]] = []

        # Determine which tools can run in parallel
        parallel_batch: list[tuple[str, dict]] = []
        sequential_batch: list[tuple[str, dict]] = []

        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            call_id = tc.get("id", "")
            args_str = func.get("arguments", "{}")

            try:
                args = json.loads(args_str)
            except json.JSONDecodeError as e:
                result = ToolResult.failure(
                    "ToolValidationError",
                    f"Некорректный JSON аргументов: {e}",
                )
                results.append((call_id, result.to_json()))
                continue

            tool = self.registry.get(name)
            if tool is None:
                result = ToolResult.failure(
                    "ToolNotFoundError",
                    f"Инструмент '{name}' не найден",
                )
                results.append((call_id, result.to_json()))
                continue

            # Validate args
            try:
                args = tool.validate_args(args)
            except ToolValidationError as e:
                result = ToolResult.failure("ToolValidationError", str(e))
                results.append((call_id, result.to_json()))
                continue

            if name in PARALLEL_TOOLS:
                parallel_batch.append((call_id, {"tool": tool, "args": args, "name": name}))
            else:
                sequential_batch.append((call_id, {"tool": tool, "args": args, "name": name}))

        # Execute parallel batch
        if parallel_batch:
            tasks = []
            for call_id, info in parallel_batch:
                tasks.append(self._exec_single(call_id, info["tool"], info["args"]))
            parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
            for (call_id, _), result in zip(parallel_batch, parallel_results):
                if isinstance(result, Exception):
                    r = ToolResult.from_exception(result)
                    results.append((call_id, r.to_json()))
                else:
                    results.append((call_id, result))

        # Execute sequential batch
        for call_id, info in sequential_batch:
            result = await self._exec_single(call_id, info["tool"], info["args"])
            results.append((call_id, result))

        return results

    async def _exec_single(
        self, call_id: str, tool: Tool, args: dict
    ) -> str:
        """Execute a single tool call."""
        if self.tool_status_fn:
            self.tool_status_fn(tool.name, "running", call_id)

        start = time.monotonic()

        context = {
            "workspace": self.context.workspace,
            "config": self.config,
            "permission_mode": self.config.tools.permission,
            "read_only": self.config.tools.read_only,
            "shell_type": self.config.tools.shell_type,
            "shell_timeout": self.config.tools.shell_timeout,
            "max_output_chars": self.config.tools.max_output_chars,
            "allow_write_outside_workspace": self.config.tools.allow_write_outside_workspace,
            "allow_private_hosts": self.config.web.allow_private_hosts,
            "web_timeout": self.config.web.timeout,
            "max_page_bytes": self.config.web.max_page_bytes,
            "max_block_chars": self.config.memory.max_block_chars,
            "ask_fn": self.ask_fn,
            "confirm_fn": self.confirm_fn,
        }

        try:
            result = await tool.execute(args, context)
        except Exception as e:
            result = ToolResult.from_exception(e)

        duration_ms = int((time.monotonic() - start) * 1000)
        result.meta.setdefault("duration_ms", duration_ms)

        status = "success" if result.ok else "error"
        if self.tool_status_fn:
            self.tool_status_fn(tool.name, status, call_id)

        return result.to_json()

    def get_token_status(self) -> str:
        """Get formatted token status."""
        return self.context.format_token_status()
