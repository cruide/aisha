# Author: Tischenko A. (https://github.com/cruide)
"""Entry point: argument parsing, wiring, --doctor, one-shot and REPL modes."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from aisha import __version__
from aisha.agent import AgentLoop
from aisha.client import LlamaClient
from aisha.config import Config, load_config
from aisha.context import ConversationContext, build_tool_guide
from aisha.errors import AishaError, ConfigurationError
from aisha.memory import MemoryStore
from aisha.skills import SkillIndex
from aisha.tools.base import ToolContext, ToolRegistry
from aisha.tools.extras import (
    AskUserTool,
    MemoryGetTool,
    MemoryListTool,
    MemoryReplaceTool,
    MemorySetTool,
    SkillTool,
    TodoWriteTool,
)
from aisha.tools.files import (
    EditFileTool,
    GlobTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from aisha.tools.shell import RunCommandTool
from aisha.tools.web import WebFetchTool, WebSearchTool
from aisha.ui import ConsoleUI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aisha", description="Локальный консольный AI-агент.")
    p.add_argument("prompt", nargs="*", help="одноразовый запрос (без него — REPL)")
    p.add_argument("--server", help="URL llama-server, например http://localhost:8088")
    p.add_argument("--model", help="имя модели (alias -a на сервере)")
    p.add_argument("--api-key", help="API-ключ для сервера (если требуется авторизация)")
    p.add_argument("--skip-health", action="store_true",
                   help="пропустить проверку /health (для несовместимых серверов)")
    p.add_argument("-r", "--read-only", action="store_true", help="режим только для чтения")
    p.add_argument("--permission", choices=("auto", "ask", "deny"), help="режим shell")
    p.add_argument("--shell", choices=("powershell", "cmd"), help="оболочка по умолчанию")
    p.add_argument("--tools-only", action="store_true", help="показать инструменты и выйти")
    p.add_argument("--doctor", action="store_true", help="диагностика подключения")
    p.add_argument("--tool-call-test", action="store_true",
                   help="с --doctor: проверить tool calling")
    p.add_argument("--no-color", action="store_true", help="отключить цвета")
    p.add_argument("--debug", action="store_true",
                   help="режим отладки: reasoning модели, дампы запросов/ответов, traceback")
    p.add_argument("--version", action="version", version=f"aisha {__version__}")
    return p


def cli_overrides(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    over: dict[str, dict[str, Any]] = {}
    if args.server:
        over.setdefault("server", {})["base_url"] = args.server
    if args.model:
        over.setdefault("server", {})["model"] = args.model
    if args.api_key:
        over.setdefault("server", {})["api_key"] = args.api_key
    if args.skip_health:
        over.setdefault("server", {})["skip_health"] = True
    if args.permission:
        over.setdefault("tools", {})["permission"] = args.permission
    if args.shell:
        over.setdefault("tools", {})["shell_type"] = args.shell
    if args.debug:
        over.setdefault("ui", {})["debug"] = True
    return over


def build_registry(config: Config) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (ReadFileTool(), WriteFileTool(), EditFileTool(), ListDirTool(), GlobTool(),
                 GrepTool()):
        registry.register(tool)
    if config.tools.shell:
        registry.register(RunCommandTool())
    if config.tools.web_search:
        registry.register(WebSearchTool())
    registry.register(WebFetchTool())
    registry.register(TodoWriteTool())
    registry.register(AskUserTool())
    registry.register(SkillTool())
    if config.memory.enabled:
        for tool in (MemoryListTool(), MemoryGetTool(), MemorySetTool(), MemoryReplaceTool()):
            registry.register(tool)
    return registry


async def run_doctor(config: Config, client: LlamaClient, ui: ConsoleUI,
                     tool_call_test: bool = False) -> bool:
    ok_all = True

    def report(ok: bool, label: str, detail: str = "", warn: bool = False) -> None:
        nonlocal ok_all
        if not ok and not warn:
            ok_all = False
        mark = "[green]✓[/]" if ok else ("[yellow]⚠[/]" if warn else "[red]✗[/]")
        ui.console.print(f"  {mark} {label}" + (f" [dim]— {detail}[/]" if detail else ""))

    ui.console.print(f"[bold]Диагностика[/] {config.server.base_url}")
    if config.server.skip_health:
        report(True, "/health", "пропущена (--skip-health)", warn=True)
    else:
        try:
            health = await client.health()
            if health is not None:
                report(True, "/health", str(health.get("status", "ok")))
            else:
                report(True, "/health", "не JSON — пропущена", warn=True)
        except AishaError as exc:
            report(False, "/health", str(exc))
            ui.info("Проверьте, что llama-server запущен (команда — в README.md).")
            return False
    try:
        info = await client.model_info()
        names = list(info)
        report(bool(names), "/v1/models", ", ".join(names) or "пусто")
        if not names:
            return False
        if client.model in info:
            model, matched = client.model, True
        else:
            model, matched = names[0], False
            client.model = model
        report(matched, "модель", model if matched else
               f"'{config.server.model}' не найдена, используется '{model}'", warn=True)
    except AishaError as exc:
        report(False, "/v1/models", str(exc))
        return False
    try:
        resp = await client.chat([{"role": "user", "content": "Ответь одним словом: ok"}],
                                 None, temperature=0.0, max_tokens=64)
        report(bool(resp.content.strip()), "/v1/chat/completions",
               f"ответ: {resp.content.strip()[:40]!r}, usage: {'есть' if resp.usage else 'нет'}")
    except AishaError as exc:
        report(False, "/v1/chat/completions", str(exc))
    if tool_call_test:
        echo = {"type": "function", "function": {
            "name": "echo", "description": "Вернуть переданный текст без изменений",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                           "required": ["text"]}}}
        try:
            resp = await client.chat(
                [{"role": "user", "content": "Вызови инструмент echo с текстом 'ping'."}],
                [echo], temperature=0.0, max_tokens=1024,
            )
            calls = [f"{c.name}({c.arguments})" for c in resp.tool_calls]
            report(any(c.name == "echo" for c in resp.tool_calls), "tool calling",
                   ", ".join(calls) or f"инструмент не вызван: {resp.content[:60]!r}")
        except AishaError as exc:
            report(False, "tool calling", str(exc))
    ui.console.print("[green]Готово: всё в порядке.[/]" if ok_all else
                     "[red]Обнаружены проблемы.[/]")
    return ok_all


async def _amain(args: argparse.Namespace) -> int:
    workspace = Path.cwd().resolve()
    no_color = args.no_color or bool(os.environ.get("NO_COLOR"))
    ui = ConsoleUI(no_color=no_color, debug=args.debug)
    try:
        config = load_config(workspace, cli=cli_overrides(args), read_only=args.read_only)
    except ConfigurationError as exc:
        ui.error(f"Ошибка конфигурации: {exc}")
        return 2

    registry = build_registry(config)
    if args.tools_only:
        ui.print_tools(registry)
        return 0

    client = LlamaClient(config.server.base_url, config.server.model,
                         api_key=config.server.api_key,
                         skip_health=config.server.skip_health,
                         connect_timeout=config.server.connect_timeout,
                         request_timeout=config.server.request_timeout)
    try:
        if args.doctor:
            return 0 if await run_doctor(config, client, ui, args.tool_call_test) else 1
        try:
            model, matched, n_ctx = await client.resolve_model_meta()
        except AishaError as exc:
            ui.error(str(exc), exc)
            ui.info("Подсказка: aisha --doctor покажет подробности; сервер должен слушать "
                    f"{config.server.base_url}.")
            return 1

        # if not matched:
        #     ui.warn(f"Модель '{config.server.model}' не найдена на сервере, "
        #             f"используется '{model}'.")

        if n_ctx:
            config.llm.context_window = n_ctx
            config.llm.max_output_tokens = n_ctx

        memory = (MemoryStore(config.home_dir / "memory", config.project_dir / "memory",
                              max_block_chars=config.memory.max_block_chars)
                  if config.memory.enabled else None)
        skills = SkillIndex(config.home_dir / "skills", config.project_dir / "skills")
        tool_guide = (build_tool_guide(registry.schemas(read_only=config.read_only))
                      if config.llm.tool_guide else "")
        context = ConversationContext(config, memory, skills, tool_guide)
        ui.attach(config, context, client)
        tool_ctx = ToolContext(
            workspace=workspace, config=config, memory=memory, skills=skills,
            todos=context.todos, confirm=ui.confirm if ui.interactive else None,
            ask=ui.ask_user if ui.interactive else None, interactive=ui.interactive,
            on_system_change=context.invalidate,
        )
        agent = AgentLoop(config, client, registry, context, tool_ctx, ui)

        prompt = " ".join(args.prompt).strip()
        if prompt:
            return await ui.run_once(agent, prompt)
        await ui.run_repl(agent, registry, lambda: run_doctor(config, client, ui))
        return 0
    finally:
        await client.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if os.name == "nt":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
            except (AttributeError, ValueError):
                pass
    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        return 130
