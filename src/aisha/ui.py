# Author: Tischenko A. (https://github.com/cruide)
"""Terminal UI: rich output, prompt_toolkit input, REPL commands, confirmations."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import sys
import traceback
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from aisha import __version__
from aisha.agent import AgentLoop
from aisha.client import ChatResponse, LlamaClient, ToolCall
from aisha.config import Config
from aisha.context import ConversationContext
from aisha.errors import AishaError, ToolCancelledError
from aisha.tools.base import ConfirmRequest, ToolRegistry, ToolResult

COMMANDS = {
    "/help": "справка по командам",
    "/new": "новая сессия (очистить историю диалога)",
    "/status": "сервер, модель, workspace, режим, токены",
    "/tools": "доступные инструменты",
    "/skills": "индекс скиллов",
    "/memory": "блоки постоянной памяти",
    "/compact": "сжать историю диалога",
    "/doctor": "проверить соединение с сервером",
    "/init": "изучить проект и создать AGENTS.md",
    "/clear": "очистить экран",
    "/quit": "выход (также /exit, Ctrl+D)",
}

PT_STYLE = Style.from_dict({
    "prompt": "bold #d75fff",
    "bottom-toolbar": "noreverse bg:default #8a8a8a",
    "bottom-toolbar.rule": "noreverse bg:default #444444",
})


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def fmt_ctx(n: int) -> str:
    if n % 1024 == 0:
        return f"{n // 1024}K"
    return fmt_int(n)


def fmt_short(n: int) -> str:
    if n < 1024:
        return fmt_int(n)
    value = n / 1024
    return f"{value:.0f}K" if n % 1024 == 0 else f"{value:.1f}K"


class AishaCompleter(Completer):
    """Slash commands at line start, filesystem paths for the last word otherwise."""

    def __init__(self) -> None:
        self._paths = PathCompleter(expanduser=True)

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/") and " " not in text:
            for cmd in COMMANDS:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text), display_meta=COMMANDS[cmd])
            return
        word = document.get_word_before_cursor(WORD=True)
        if word:
            yield from self._paths.get_completions(Document(word, len(word)), complete_event)


class ConsoleUI:
    def __init__(self, *, no_color: bool = False, debug: bool = False) -> None:
        self.console = Console(no_color=no_color, highlight=False)
        self.debug = debug
        self.interactive = sys.stdin.isatty() and sys.stdout.isatty()
        self.config: Config | None = None
        self.context: ConversationContext | None = None
        self.client: LlamaClient | None = None
        self.session: PromptSession | None = None
        self.sent: list[str] = []
        self._nav: int | None = None
        self._draft = ""
        self._stream_live: Live | None = None
        self._tail: str = ""
        self._rtail: str = ""
        self._tool_live: Live | None = None
        self._pending: dict[str, str] = {}

    # ------------------------------------------------------------------ setup
    def attach(self, config: Config, context: ConversationContext, client: LlamaClient) -> None:
        self.config, self.context, self.client = config, context, client
        history_path = Path(os.path.expanduser(config.ui.input_history))
        history_path.parent.mkdir(parents=True, exist_ok=True)
        self.session = PromptSession(
            history=FileHistory(str(history_path)),
            completer=AishaCompleter(),
            complete_while_typing=False,
            key_bindings=self._key_bindings(),
            bottom_toolbar=self._toolbar,
            style=PT_STYLE,
        )

    def _key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("c-up")
        def _prev(event) -> None:
            self._navigate(event.current_buffer, -1)

        @kb.add("c-down")
        def _next(event) -> None:
            self._navigate(event.current_buffer, +1)

        return kb

    def _navigate(self, buffer, delta: int) -> None:
        if not self.sent:
            return
        if self._nav is None:
            self._draft = buffer.text
            self._nav = len(self.sent)
        target = max(0, min(len(self.sent), self._nav + delta))
        if target == self._nav and target != len(self.sent):
            return
        self._nav = target
        text = self._draft if target == len(self.sent) else self.sent[target]
        if target == len(self.sent):
            self._nav = None
        buffer.document = Document(text, cursor_position=len(text))

    def _toolbar(self):
        stats = self.context.stats if self.context else None
        width = shutil.get_terminal_size((80, 24)).columns
        if not stats or not self.config:
            status = "CTX: — | Last: — | Session: —"
        else:
            pct = round(stats.ctx * 100 / self.config.llm.context_window)
            status = (
                f"CTX: {fmt_short(stats.ctx)} (~{pct}%)"
                f" | Last: ↑{fmt_short(stats.last_in)} ↓{fmt_short(stats.last_out)}"
                f" | Session: ↑{fmt_short(stats.session_in)} ↓{fmt_short(stats.session_out)}"
            )
            if self.config.read_only:
                status += " | read-only"
        return [("class:bottom-toolbar.rule", "─" * width), ("", "\n"),
                ("class:bottom-toolbar", status)]

    # --------------------------------------------------------------- printing
    def error(self, text: str, exc: BaseException | None = None) -> None:
        self._stop_lives()
        self.console.print(f"[bold red]✗[/] {escape(text)}")
        if exc is not None and (self.debug or (self.config and self.config.ui.debug)):
            self.console.print(Text("".join(traceback.format_exception(exc)), style="dim"))

    def warn(self, text: str) -> None:
        self.console.print(f"[yellow]⚠[/] {escape(text)}")

    def info(self, text: str) -> None:
        self.console.print(f"[dim]{escape(text)}[/]")

    def print_banner(self, model: str) -> None:
        cfg = self.config

        assert cfg is not None

        mode = "read-only" if cfg.read_only else f"permission={cfg.tools.permission}"

        body = Text()

        body.append("AISHA ", style="bold yellow")
        body.append(f"v{__version__}", style="white")
        body.append(" · MODEL: ", style="cyan")
        body.append(f"{model}", style="green")
        body.append(" · N_CTX: ", style="cyan")
        body.append(f"{fmt_ctx(cfg.llm.context_window)}\n", style="green")
        body.append(f"Workspace: {cfg.workspace}\n", style="dim")
        body.append(
            f"{cfg.server.base_url} · {mode} · Shell: {cfg.tools.shell_type}\n", style="dim"
        )
        body.append(
            "/help — команды · Tab — пути · Ctrl+↑/↓ — прошлые запросы · Ctrl+C — прервать",
            style="dim",
        )

        self.console.print(Panel(body, border_style="magenta", padding=(0, 1)))

    def print_help(self) -> None:
        table = Table(box=None, show_header=False, padding=(0, 2))

        for cmd, desc in COMMANDS.items():
            table.add_row(f"[bold cyan]{cmd}[/]", desc)

        self.console.print(Panel(table, title="Команды", title_align="left", border_style="cyan"))

    def print_tools(self, registry: ToolRegistry) -> None:
        table = Table(box=None, show_header=True, header_style="bold", padding=(0, 2))
        table.add_column("Инструмент")
        table.add_column("Режим")
        table.add_column("Описание")
        for tool in registry:
            table.add_row(f"[cyan]{tool.name}[/]", "read" if tool.read_only else "[yellow]write[/]",
                          tool.description)
        self.console.print(Panel(table, title="Инструменты", title_align="left"))

    def print_status(self, model: str, registry: ToolRegistry) -> None:
        cfg, ctx = self.config, self.context
        assert cfg and ctx
        s = ctx.stats
        table = Table.grid(padding=(0, 2))
        rows = [
            ("Сервер", cfg.server.base_url), ("Модель", model),
            ("Workspace", str(cfg.workspace)),
            ("Режим", "read-only" if cfg.read_only else f"permission={cfg.tools.permission}"),
            ("Shell", cfg.tools.shell_type if cfg.tools.shell else "отключён"),
            ("Контекст", f"{'~' if s.approximate else ''}{fmt_int(s.ctx)} / "
                         f"{fmt_int(cfg.llm.context_window)} "
                         f"(бюджет {fmt_int(ctx.input_budget())})"),
            ("Последний запрос", f"↑ {fmt_int(s.last_in)}  ↓ {fmt_int(s.last_out)}"),
            ("Сессия", f"↑ {fmt_int(s.session_in)}  ↓ {fmt_int(s.session_out)}"),
            ("Сообщений", str(len(ctx.messages))),
            ("Инструментов", str(len(registry.names()))),
            ("AGENTS.md", "загружен" if ctx.agents_md else "нет"),
            ("SYSTEM.md", "заменяет базовый промпт" if ctx.system_md else "нет"),
            ("Конфиги", ", ".join(cfg.sources) or "по умолчанию"),
        ]
        for k, v in rows:
            table.add_row(f"[bold]{k}[/]", escape(v))
        self.console.print(Panel(table, title="Статус", title_align="left", border_style="cyan"))

    def print_skills(self) -> None:
        assert self.context
        idx = self.context.skills
        if not idx.skills and not idx.errors:
            # self.info("Скиллы не найдены (~/.aisha/skills, <workspace>/.aisha/skills).")
            return
        table = Table.grid(padding=(0, 2))
        for s in idx.skills.values():
            table.add_row(f"[cyan]{s.name}[/]", f"[dim]{s.scope}[/]", escape(s.description))
        self.console.print(Panel(table, title="Скиллы", title_align="left"))
        for err in idx.errors:
            self.warn(err)

    def print_memory(self) -> None:
        assert self.context
        if self.context.memory is None:
            self.info("Память отключена.")
            return
        blocks = self.context.memory.list()
        if not blocks:
            self.info("Блоков памяти пока нет.")
            return
        table = Table.grid(padding=(0, 2))
        for b in blocks:
            table.add_row(f"[cyan]{b.label}[/]", f"[dim]{b.scope}[/]",
                          escape(b.description or "—"), f"[dim]{len(b.value)} симв.[/]")
        self.console.print(Panel(table, title="Память", title_align="left"))

    def print_todos(self) -> None:
        assert self.context
        if not self.context.todos:
            return
        icons = {"pending": "○", "in_progress": "◐", "done": "●", "cancelled": "✕"}
        styles = {
            "pending": "",
            "in_progress": "yellow",
            "done": "green",
            "cancelled": "dim strike",
        }
        text = Text()
        for t in self.context.todos:
            text.append(f"{icons[t['status']]} {t['text']}\n", style=styles[t["status"]])
        self.console.print(Panel(text, title="Задачи", title_align="left", border_style="blue"))

    # ----------------------------------------------------------- agent events
    def on_stream_start(self) -> None:
        self._tail = ""
        self._rtail = ""
        if self.interactive and self.config and self.config.ui.stream:
            self._stream_live = Live(self._render_stream(), console=self.console,
                                     refresh_per_second=8, transient=True,
                                     vertical_overflow="crop")
            self._stream_live.start()

    def _stream_height(self) -> int:
        return max(4, self.console.size.height - 8)

    @staticmethod
    def _grow_tail(current: str, delta: str, height: int) -> str:
        """Append `delta` keeping only the last `height` lines (O(height), not O(total))."""
        if not delta:
            return current
        return "\n".join((current + delta).splitlines()[-height:])

    def _show_reasoning(self) -> bool:
        return bool(self.config and (self.config.ui.show_reasoning or self.config.ui.debug))

    def _render_stream(self):
        label = "aisha отвечает…" if self._tail else "aisha думает…"
        parts: list[Any] = [Spinner("dots", text=Text(label, style="yellow3"))]
        if not self._tail and self._rtail and self._show_reasoning():
            parts.append(Text(self._rtail, style="dim italic"))
        if self._tail:
            parts.append(Text(self._tail, style="dim"))
        return Group(*parts)

    def on_text(self, delta: str) -> None:
        if self._stream_live:
            self._tail = self._grow_tail(self._tail, delta, self._stream_height())
            self._stream_live.update(self._render_stream())

    def on_reasoning(self, delta: str) -> None:
        if self._stream_live:
            self._rtail = self._grow_tail(self._rtail, delta, self._stream_height())
            self._stream_live.update(self._render_stream())

    def on_stream_end(self, response: ChatResponse) -> None:
        if self._stream_live:
            self._stream_live.stop()
            self._stream_live = None
        if not self.interactive:
            return
        if response.reasoning and self._show_reasoning():
            self.console.print(Panel(Text(response.reasoning.strip(), style="dim italic"),
                                     title="reasoning", title_align="left", border_style="dim"))
        if response.content.strip():
            self.console.print(Panel(Markdown(response.content, code_theme="monokai"),
                                     title="aisha", title_align="left", border_style="magenta"))
    def on_tool_start(self, call: ToolCall, args: dict[str, Any] | None) -> None:
        self._pending[call.id] = self._fmt_call(call.name, args)
        if not self.interactive:
            return
        if self._tool_live is None:
            self._tool_live = Live(self._render_pending(), console=self.console,
                                   refresh_per_second=8, transient=True)
            self._tool_live.start()
        else:
            self._tool_live.update(self._render_pending())

    def _render_pending(self):
        return Group(*(Text.assemble(("  ⟳ ", "yellow"), (line, "dim"))
                       for line in self._pending.values()))

    def on_tool_end(self, call: ToolCall, result: ToolResult) -> None:
        label = self._pending.pop(call.id, self._fmt_call(call.name, None))
        if call.name != "skill" or result.ok or (result.error or {}).get("type") != "NotFound":
            ms = result.meta.get("duration_ms")
            suffix = f" · {ms} ms" if ms is not None else ""
            if result.ok:
                line = Text.assemble(("  ✓ ", "green"), (call.name, "bold"),
                                     (f" — {result.summary or 'ok'}", ""), (suffix, "dim"))
                if result.meta.get("truncated"):
                    line.append(" (обрезано)", style="yellow")
            else:
                line = Text.assemble(("  ✗ ", "red"), (call.name, "bold"),
                                     (f" — {result.summary}", "red"), (suffix, "dim"))
            line.append(f"\n      {label}", style="dim")
            self.console.print(line)
        if call.name == "todowrite" and result.ok:
            self.print_todos()
        if self._tool_live:
            if self._pending:
                self._tool_live.update(self._render_pending())
            else:
                self._tool_live.stop()
                self._tool_live = None

    def on_notice(self, text: str, level: str = "info") -> None:
        (self.warn if level == "warn" else self.info)(text)

    def on_debug(self, title: str, body: str) -> None:
        self.console.print(Panel(Text(body, style="dim"), title=f"debug · {title}",
                                 title_align="left", border_style="blue"))

    @staticmethod
    def _fmt_call(name: str, args: dict[str, Any] | None) -> str:
        if args is None:
            return f"{name}(…)"
        parts = []
        for key, value in args.items():
            s = json.dumps(value, ensure_ascii=False)
            if len(s) > 70:
                s = s[:67] + "…" + ('"' if s.startswith('"') else "")
            parts.append(f"{key}={s}")
        text = f"{name}({', '.join(parts)})"
        return text if len(text) <= 120 else text[:117] + "…)"

    def _stop_lives(self) -> None:
        for attr in ("_stream_live", "_tool_live"):
            live = getattr(self, attr)
            if live is not None:
                live.stop()
                setattr(self, attr, None)

    # ------------------------------------------------------- confirm / ask
    async def _plain_prompt(self, message: str) -> str:
        try:
            return await PromptSession(style=PT_STYLE).prompt_async([("class:prompt", message)])
        except (KeyboardInterrupt, EOFError) as exc:
            raise ToolCancelledError("отменено пользователем") from exc

    async def confirm(self, request: ConfirmRequest) -> str | None:
        self._stop_lives()
        table = Table.grid(padding=(0, 1))
        for key, value in request.details:
            table.add_row(f"[bold]{key}:[/]", escape(value))
        table.add_row("[bold]Причина:[/]", f"[yellow]{escape(request.reason)}[/]")
        self.console.print(Panel(table, title=f"⚠ {request.title}", title_align="left",
                                 border_style="yellow"))
        answer = await self._plain_prompt("[y] один раз  [a] до конца сессии  [n] отказать › ")
        return {"y": "y", "yes": "y", "д": "y", "a": "a", "all": "a", "в": "a"}.get(
            answer.strip().lower()
        )

    async def ask_user(self, question: str, options: list[str], allow_free_text: bool) -> str:
        self._stop_lives()
        body = Text(question)
        for i, option in enumerate(options, 1):
            body.append(f"\n  {i}. {option}", style="cyan")
        self.console.print(Panel(body, title="❔ Вопрос", title_align="left", border_style="cyan"))
        while True:
            answer = (await self._plain_prompt("Ответ › ")).strip()
            if not answer:
                continue
            if options and answer.isdigit() and 1 <= int(answer) <= len(options):
                return options[int(answer) - 1]
            if not options or allow_free_text or answer in options:
                return answer
            self.warn("Выберите номер одного из вариантов.")

    # ------------------------------------------------------------------ REPL
    async def _run_cancellable(self, coro: Awaitable[Any]) -> Any:
        """Run agent work; Ctrl+C cancels it but keeps the REPL alive."""
        loop = asyncio.get_running_loop()
        task = asyncio.ensure_future(coro)

        def on_sigint(signum, frame) -> None:
            loop.call_soon_threadsafe(task.cancel)

        previous = signal.signal(signal.SIGINT, on_sigint)
        try:
            return await task
        except asyncio.CancelledError:
            self._stop_lives()
            self.warn("Прервано пользователем.")
        except AishaError as exc:
            self.error(str(exc), exc)
        except Exception as exc:  # never let a bug kill the REPL
            self.error(f"{type(exc).__name__}: {exc}", exc)
        finally:
            self._stop_lives()
            signal.signal(signal.SIGINT, previous)
        return None

    async def run_once(self, agent: AgentLoop, prompt: str) -> int:
        result = await self._run_cancellable(agent.run(prompt))
        if not self.interactive and isinstance(result, str):
            print(result)
        return 0 if result is not None else 1

    async def run_repl(
        self,
        agent: AgentLoop,
        registry: ToolRegistry,
        doctor: Callable[[], Awaitable[bool]],
    ) -> None:
        assert self.session and self.client
        self.print_banner(self.client.model)
        while True:
            try:
                text = await self.session.prompt_async([("class:prompt", "❯ ")])
            except KeyboardInterrupt:
                self.info("Выход.")
                return
            except EOFError:
                return
            text = text.strip()
            if not text:
                continue
            if text.startswith("/"):
                if not await self._command(text, agent, registry, doctor):
                    return
                continue
            self.sent.append(text)
            self._nav, self._draft = None, ""
            await self._run_cancellable(agent.run(text))

    async def _command(self, text: str, agent: AgentLoop, registry: ToolRegistry,
                       doctor: Callable[[], Awaitable[bool]]) -> bool:
        assert self.context and self.client
        cmd, _, _arg = text.partition(" ")
        cmd = cmd.lower()
        if cmd in ("/quit", "/exit"):
            return False
        if cmd == "/help":
            self.print_help()
        elif cmd == "/new":
            self.context.reset()
            self.info("Новая сессия: история и счётчики сброшены, AGENTS.md перечитан.")
        elif cmd == "/status":
            self.print_status(self.client.model, registry)
        elif cmd == "/tools":
            self.print_tools(registry)
        elif cmd == "/skills":
            self.print_skills()
        elif cmd == "/memory":
            self.print_memory()
        elif cmd == "/compact":
            await self._run_cancellable(agent.compact(force=True))
        elif cmd == "/doctor":
            await self._run_cancellable(doctor())
        elif cmd == "/clear":
            self.console.clear()
        elif cmd == "/init":
            prompt = (
                "Изучи очень внимательно структуру этого проекта (list_dir, glob, read_file "
                "ключевых файлов: README, конфиги сборки, точки входа) и создай в корне файл "
                "AGENTS.md: подробное назначение проекта, команды сборки/тестов/линта, "
                "архитектура по каталогам, конвенции кода и неочевидные особенности. "
                "По возможности, до 25 КБ. Вся информация в AGENTS.md должна быть "
                "на английском языке."
            )
            await self._run_cancellable(agent.run(prompt))
        else:
            self.warn(f"Неизвестная команда {cmd}. /help — список команд.")
        return True
