"""UI module: REPL, rich rendering, prompt_toolkit integration, confirmations."""

from __future__ import annotations

__author__ = "Tischenko Alexander"

import asyncio
import os
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text
from rich.theme import Theme

_theme = Theme(
    {
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "red bold",
        "tool.running": "cyan italic",
        "tool.success": "green",
        "tool.error": "red",
        "tool.warning": "yellow",
        "user.label": "bold blue",
        "assistant.label": "bold green",
        "status.line": "dim",
    }
)


def _truncate_detail(detail: str, max_chars: int = 120) -> str:
    """Collapse whitespace and truncate a tool detail line for display."""
    if not detail:
        return ""
    text = " ".join(detail.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


class AishaUI:
    """Terminal UI using rich + prompt_toolkit."""

    def __init__(
        self,
        console: Console | None = None,
        no_color: bool = False,
        input_history: str = "~/.aisha/input_history.txt",
    ) -> None:
        self.console = console or Console(
            theme=_theme,
            force_terminal=True,
            color_system="truecolor" if not no_color else None,
            highlight=False,
        )
        self.no_color = no_color
        self.input_history = input_history
        self._stream_buffer: list[str] = []
        self._prompt_session: Any = None

    def print_welcome(self) -> None:
        """Print welcome message."""
        self.console.print("[bold]aisha[/bold] — local AI agent. By Cruide (https://github.com/cruide)")
        self.console.print("Type your message or /help for commands.")

    def print_help(self) -> None:
        """Print help with available commands."""
        help_text = (
            "[bold]Available commands:[/bold]\n\n"
            "/help    — show this help\n"
            "/new     — new session, clear history\n"
            "/status  — server, model, workspace, permissions, tokens\n"
            "/tools   — list available tools\n"
            "/skills  — list available skills\n"
            "/memory  — list memory blocks\n"
            "/compact — force compact history\n"
            "/doctor  — check server connection\n"
            "/init    — study project and create AGENTS.md\n"
            "/clear   — clear screen\n"
            "/quit    — exit\n"
            "/exit    — exit"
        )
        self.console.print(help_text)

    def print_status(self, config: Any, workspace: Any, token_status: str) -> None:
        """Print status information."""
        status_text = (
            f"[bold]Server:[/bold] {config.server.base_url}\n"
            f"[bold]Model:[/bold] {config.server.model}\n"
            f"[bold]Workspace:[/bold] {workspace}\n"
            f"[bold]Permission:[/bold] {config.tools.permission}\n"
            f"[bold]Shell:[/bold] {config.tools.shell_type}\n"
            f"[bold]Tokens:[/bold] {token_status}"
        )
        self.console.print(status_text)

    def print_tools(self, tools: list[dict]) -> None:
        """Print available tools."""
        lines = []
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "?")
            desc = func.get("description", "")
            lines.append(f"  [bold]{name}[/bold] — {desc}")
        self.console.print("\n".join(lines))

    def print_assistant_start(self) -> None:
        """Start assistant response area."""
        self._stream_buffer = []

    def print_assistant_text(self, text: str) -> None:
        """Append text to assistant streaming buffer."""
        self._stream_buffer.append(text)

    def flush_assistant(self) -> None:
        """Flush accumulated assistant text as markdown."""
        full = "".join(self._stream_buffer)
        if full:
            self.console.print(Markdown(full))
        self._stream_buffer.clear()

    def print_thinking(self, text: str) -> None:
        """Display reasoning/thinking content."""
        if text:
            self.console.print(Text(text, style="dim italic"))

    def print_tool_status(
        self, name: str, status: str, call_id: str = "", detail: str = ""
    ) -> None:
        """Print tool execution status."""
        icons = {
            "running": "[tool.running]⟳[/tool.running]",
            "success": "[tool.success]✓[/tool.success]",
            "error": "[tool.error]✗[/tool.error]",
            "warning": "[tool.warning]⚠[/tool.warning]",
        }
        icon = icons.get(status, "?")
        detail_text = _truncate_detail(detail)
        if detail_text:
            self.console.print(f"  {icon} [bold]{name}[/bold] [dim]{detail_text}[/dim]")
        else:
            self.console.print(f"  {icon} {name}")

    def print_error(self, message: str) -> None:
        """Print error message."""
        self.console.print(f"[error]✗ {message}[/error]")

    def print_warning(self, message: str) -> None:
        """Print warning message."""
        self.console.print(f"[warning]⚠ {message}[/warning]")

    def print_info(self, message: str) -> None:
        """Print info message."""
        self.console.print(f"[info]{message}[/info]")

    def print_token_status(self, status: str) -> None:
        """Print token status line."""
        self.console.print(f"[status.line]{'─' * 80}[/status.line]")
        self.console.print(f"[status.line]{status}[/status.line]")
        # self.console.print("[status.line]© Tischenko Alexander[/status.line]")
        self.console.print(f"[status.line]{'─' * 80}[/status.line]")

    async def ask_confirmation(
        self,
        question: str,
        command: str = "",
        shell: str = "",
        cwd: str = "",
        reason: str = "",
    ) -> str:
        """Ask user for confirmation of dangerous operation.

        Returns 'y', 'a', or 'n'.
        """
        self.console.print(f"\n[warning]⚠ {question}[/warning]")
        if command:
            self.console.print(f"  [bold]Command:[/bold] {command}")
        if shell:
            self.console.print(f"  [bold]Shell:[/bold] {shell}")
        if cwd:
            self.console.print(f"  [bold]CWD:[/bold] {cwd}")
        if reason:
            self.console.print(f"  [bold]Reason:[/bold] {reason}")

        self.console.print(
            "\n  [y] выполнить один раз  "
            "[a] разрешить подобные  "
            "[n] отказать"
        )

        try:
            answer = await self._input_async("> ")
            answer = answer.strip().lower()
            if answer in ("y", "a", "n"):
                return answer
            return "n"
        except (EOFError, KeyboardInterrupt):
            return "n"

    async def ask_question(
        self,
        question: str,
        options: list[str] | None = None,
    ) -> str | None:
        """Ask user a question via ask_user tool."""
        self.console.print(f"\n[bold blue]Вопрос:[/bold blue] {question}")

        if options:
            for i, opt in enumerate(options, 1):
                self.console.print(f"  [dim]{i}.[/dim] {opt}")

        try:
            answer = await self._input_async("> ")
            answer = answer.strip()
            if not answer:
                return None
            # Try to match option number
            if options:
                try:
                    idx = int(answer) - 1
                    if 0 <= idx < len(options):
                        return options[idx]
                except ValueError:
                    pass
            return answer
        except (EOFError, KeyboardInterrupt):
            return None

    async def _input_async(self, prompt: str = "> ") -> str:
        """Get input asynchronously using prompt_toolkit."""
        session = self._get_prompt_session()
        if session is None:
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: input(prompt)
            )
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: session.prompt(prompt))

    def _get_prompt_session(self) -> Any | None:
        """Create (once) and return the prompt_toolkit session, or None if unavailable."""
        if self._prompt_session is not None:
            return self._prompt_session
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import FileHistory

            history_path = os.path.expanduser(self.input_history)
            hist_dir = os.path.dirname(history_path)
            if hist_dir:
                os.makedirs(hist_dir, exist_ok=True)
            self._prompt_session = PromptSession(history=FileHistory(history_path))
        except ImportError:
            self._prompt_session = None
        return self._prompt_session

    def clear_screen(self) -> None:
        """Clear terminal screen."""
        self.console.clear()

    def print_cancelled(self) -> None:
        """Print cancellation message."""
        self.console.print("[warning]Операция отменена[/warning]")
