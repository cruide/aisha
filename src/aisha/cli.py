"""CLI entry point: argument parsing, REPL, one-shot queries, doctor."""

from __future__ import annotations

__author__ = "Tischenko Alexander"

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from aisha.agent import AgentLoop
from aisha.client import (
    LlamaClient,
    ServerUnavailableError,
)
from aisha.config import Config, ConfigurationError, load_config
from aisha.context import ContextManager
from aisha.memory import MemoryManager
from aisha.skills import SkillManager
from aisha.tools.base import ToolRegistry
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
from aisha.ui import AishaUI

INIT_PROMPT = (
    "Изучи внимательно данный проект и опиши как можно подробнее его архитектуру, "
    "структуру, идею и особенности в файл AGENTS.md. Проанализируй структуру "
    "каталогов, ключевые файлы, точку входа, используемые технологии и зависимости, "
    "принятые конвенции кода, конфигурацию, сборку и запуск. Итоговый AGENTS.md "
    "должен быть подробным руководством, которое позволит другому агенту быстро "
    "разобраться в проекте и эффективно в нём работать. Сохрани результат в файл "
    "AGENTS.md в корне проекта."
)


def format_context_size(n_ctx: int | None) -> str:
    """Format a context size as a human-readable string (e.g. 8192 -> '8K')."""
    if not n_ctx:
        return ""
    if n_ctx % 1024 == 0:
        return f"{n_ctx // 1024}K"
    return f"{n_ctx / 1024:.1f}K"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="aisha",
        description="Local console AI agent",
    )
    parser.add_argument("query", nargs="*", help="One-shot query")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--server", help="Override server URL")
    parser.add_argument(
        "-r", "--read-only", action="store_true", help="Read-only mode"
    )
    parser.add_argument(
        "--permission",
        choices=["auto", "ask", "deny"],
        help="Permission mode",
    )
    parser.add_argument(
        "--shell",
        choices=["powershell", "cmd"],
        help="Shell type",
    )
    parser.add_argument("--doctor", action="store_true", help="Run diagnostics")
    parser.add_argument(
        "--tool-call-test",
        action="store_true",
        help="Test tool calling (with --doctor)",
    )
    parser.add_argument("--tools-only", action="store_true", help="List tools and exit")
    parser.add_argument("--no-color", action="store_true", help="Disable colors")
    parser.add_argument("--debug", action="store_true", help="Show full tracebacks")
    return parser.parse_args()


def build_registry(
    skill_manager: SkillManager, memory_manager: MemoryManager
) -> tuple[ToolRegistry, TodoWriteTool]:
    """Create and populate the tool registry."""
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(EditFileTool())
    registry.register(ListDirTool())
    registry.register(GlobTool())
    registry.register(GrepTool())
    registry.register(RunCommandTool())
    registry.register(WebSearchTool())
    registry.register(WebFetchTool())
    todo_tool = TodoWriteTool()
    registry.register(todo_tool)
    registry.register(AskUserTool())
    registry.register(MemoryListTool(memory_manager))
    registry.register(MemoryGetTool(memory_manager))
    registry.register(MemorySetTool(memory_manager))
    registry.register(MemoryReplaceTool(memory_manager))
    registry.register(SkillTool(skill_manager))
    return registry, todo_tool


async def run_doctor(client: LlamaClient, tool_call_test: bool = False) -> bool:
    """Run diagnostics."""
    console = Console()
    console.print("[bold]aisha --doctor[/bold]\n")

    # Health check
    console.print("Checking /health ...", end=" ")
    try:
        health = await client.check_health()
        status = health.get("status", "unknown")
        console.print(f"[green]OK[/green] (status: {status})")
    except ServerUnavailableError as e:
        console.print(f"[red]FAILED[/red]\n  {e}")
        return False

    # Model check
    console.print("Checking /v1/models ...", end=" ")
    try:
        models = await client.list_models()
        model_ids = [m.get("id", "") for m in models]
        console.print(f"[green]OK[/green] ({len(models)} models)")
        if client.model not in model_ids and model_ids:
            client.model = model_ids[0]
        n_ctx = await client.get_model_context_size(client.model)
        ctx = format_context_size(n_ctx)
        suffix = f" | Контекст: {ctx}" if ctx else ""
        console.print(f"  Model [green]{client.model}[/green]{suffix}")
    except ServerUnavailableError as e:
        console.print(f"[red]FAILED[/red]\n  {e}")
        return False

    if tool_call_test:
        console.print("Testing tool calling ...", end=" ")
        ok = await client.tool_call_test()
        if ok:
            console.print("[green]OK[/green]")
        else:
            console.print("[yellow]FAILED or not supported[/yellow]")

    console.print("\n[green]All checks passed[/green]")
    return True


async def run_repl(config: Config, workspace: Path, args: argparse.Namespace) -> None:
    """Run interactive REPL."""
    # Initialize components
    client = LlamaClient(
        base_url=config.server.base_url,
        model=config.server.model,
        connect_timeout=config.server.connect_timeout,
        request_timeout=config.server.request_timeout,
    )

    # UI
    ui = AishaUI(no_color=args.no_color, input_history=config.ui.input_history)

    # Skills
    global_skills = Path.home() / ".aisha" / "skills"
    project_skills = workspace / ".aisha" / "skills"
    skill_manager = SkillManager(global_skills, project_skills)

    # Memory
    global_memory = Path.home() / ".aisha" / "memory"
    project_memory = workspace / ".aisha" / "memory"
    memory_manager = MemoryManager(global_memory, project_memory)

    # Tool registry
    registry, todo_tool = build_registry(skill_manager, memory_manager)

    # Context
    skills_index = skill_manager.get_index_lines()
    memory_summary = memory_manager.get_summary()
    context = ContextManager(workspace, config, skills_index, memory_summary)

    # Confirmation handlers
    async def ask_confirmation(
        question: str, options: list[str] | None = None
    ) -> str | None:
        return await ui.ask_question(question, options)

    async def confirm_danger(
        command: str, shell: str, cwd: str, reason: str
    ) -> str | None:
        return await ui.ask_confirmation(
            question=f"Опасная команда: {reason}",
            command=command,
            shell=shell,
            cwd=cwd,
            reason=reason,
        )

    # Agent
    agent = AgentLoop(
        client=client,
        config=config,
        context=context,
        registry=registry,
        ask_fn=ask_confirmation,
        confirm_fn=confirm_danger,
        thinking_callback=(
            lambda text: ui.print_thinking(text) if config.ui.show_reasoning else None
        ),
    )

    def tool_status(name: str, status: str, call_id: str, detail: str) -> None:
        ui.print_tool_status(name, status, call_id, detail)

    agent.tool_status_fn = tool_status

    # Print welcome
    ui.print_welcome()

    try:
        # Resolve the model served by llama-server (fallback to first available)
        resolved = await client.verify_model()
    except ServerUnavailableError as e:
        ui.print_error(str(e))
        await client.close()
        return

    n_ctx = await client.get_model_context_size(resolved)
    ctx = format_context_size(n_ctx)
    suffix = f" | Контекст: {ctx}" if ctx else ""
    ui.print_info(f"Server: {config.server.base_url} | Model: {resolved}{suffix}")

    # REPL loop
    try:
        while True:
            try:
                # Get user input
                user_input = await ui._input_async("> ")
                user_input = user_input.strip()

                if not user_input:
                    continue

                # Handle local commands
                if user_input.startswith("/"):
                    cmd = user_input.split()[0].lower()
                    if cmd in ("/quit", "/exit"):
                        break
                    elif cmd == "/help":
                        ui.print_help()
                    elif cmd == "/new":
                        context.clear()
                        todo_tool.items.clear()
                        skill_manager.rebuild_index()
                        context.skills_index = skill_manager.get_index_lines()
                        context.memory_summary = memory_manager.get_summary()
                        ui.print_info("New session started")
                    elif cmd == "/status":
                        ui.print_status(config, workspace, context.format_token_status())
                    elif cmd == "/tools":
                        ui.print_tools(registry.openai_tools())
                    elif cmd == "/skills":
                        for line in skill_manager.get_index_lines():
                            ui.print_info(line)
                    elif cmd == "/memory":
                        blocks = memory_manager.list_blocks()
                        if blocks:
                            for b in blocks:
                                ui.print_info(
                                    f"  {b['label']} [{b['scope']}] — {b['description']}"
                                )
                        else:
                            ui.print_info("No memory blocks")
                    elif cmd == "/compact":
                        ui.print_info("Compacting ...")
                        ok = await context.compact(client)
                        if ok:
                            ui.print_info("Compacted")
                        else:
                            ui.print_warning("Nothing to compact")
                    elif cmd == "/doctor":
                        await run_doctor(client)
                    elif cmd == "/clear":
                        ui.clear_screen()
                    elif cmd == "/init":
                        ui.print_info("Изучаю проект и создаю AGENTS.md ...")

                        async def on_stream(type_: str, text: str) -> None:
                            if type_ == "text":
                                ui.print_assistant_text(text)

                        ui.print_assistant_start()
                        agent.ui_callback = on_stream
                        await agent.run(INIT_PROMPT)
                        ui.flush_assistant()
                        ui.print_token_status(context.format_token_status())
                    else:
                        ui.print_warning(f"Unknown command: {cmd}")
                    continue

                # Process user input
                ui.print_assistant_start()

                # Streaming callback
                async def on_stream(type_: str, text: str) -> None:
                    if type_ == "text":
                        ui.print_assistant_text(text)

                agent.ui_callback = on_stream
                await agent.run(user_input)

                # Flush and display
                ui.flush_assistant()

                # Token status
                ui.print_token_status(context.format_token_status())

            except KeyboardInterrupt:
                ui.print_cancelled()
                continue
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        await client.close()
        ui.print_info("Goodbye!")


async def run_one_shot(
    config: Config, workspace: Path, query: str, args: argparse.Namespace
) -> None:
    """Run a single query and print the response."""
    client = LlamaClient(
        base_url=config.server.base_url,
        model=config.server.model,
        connect_timeout=config.server.connect_timeout,
        request_timeout=config.server.request_timeout,
    )

    ui = AishaUI(no_color=args.no_color, input_history=config.ui.input_history)

    # Skills and memory
    global_skills = Path.home() / ".aisha" / "skills"
    project_skills = workspace / ".aisha" / "skills"
    skill_manager = SkillManager(global_skills, project_skills)

    global_memory = Path.home() / ".aisha" / "memory"
    project_memory = workspace / ".aisha" / "memory"
    memory_manager = MemoryManager(global_memory, project_memory)

    # Registry
    registry, _todo_tool = build_registry(skill_manager, memory_manager)

    # Context
    skills_index = skill_manager.get_index_lines()
    memory_summary = memory_manager.get_summary()
    context = ContextManager(workspace, config, skills_index, memory_summary)

    agent = AgentLoop(
        client=client,
        config=config,
        context=context,
        registry=registry,
        thinking_callback=lambda text: None,
    )

    try:
        response = await agent.run(query)
        if response:
            ui.console.print(Markdown(response))
    except Exception as e:
        if args.debug:
            raise
        ui.print_error(str(e))
    finally:
        await client.close()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Determine workspace
    workspace = Path.cwd().resolve()

    # Build CLI overrides
    cli_overrides: dict[str, str] = {}
    if args.model:
        cli_overrides["model"] = args.model
    if args.server:
        cli_overrides["server_url"] = args.server
    if args.permission:
        cli_overrides["permission"] = args.permission
    if args.shell:
        cli_overrides["shell"] = args.shell
    if args.read_only:
        cli_overrides["read_only"] = True

    # Load config
    try:
        config = load_config(workspace, cli_overrides)
    except ConfigurationError as e:
        Console().print(f"[red]Configuration error:[/red] {e}")
        sys.exit(1)

    # Tools-only mode
    if args.tools_only:
        console = Console()
        tools = [
            "read_file", "write_file", "edit_file",
            "list_dir", "glob", "grep",
            "run_command",
            "web_search", "web_fetch",
            "todowrite", "ask_user",
            "memory_list", "memory_get", "memory_set", "memory_replace",
            "skill",
        ]
        for t in tools:
            console.print(t)
        return

    # Doctor mode
    if args.doctor:
        client = LlamaClient(
            base_url=config.server.base_url,
            model=config.server.model,
            connect_timeout=config.server.connect_timeout,
            request_timeout=config.server.request_timeout,
        )
        try:
            asyncio.run(run_doctor(client, args.tool_call_test))
        finally:
            asyncio.run(client.close())
        return

    # One-shot or REPL
    if args.query:
        query = " ".join(args.query)
        asyncio.run(run_one_shot(config, workspace, query, args))
    else:
        try:
            asyncio.run(run_repl(config, workspace, args))
        except KeyboardInterrupt:
            pass
