# Author: Tischenko A. (https://github.com/cruide)
import io

from rich.console import Console

from aisha.client import ToolCall
from aisha.tools.base import ToolResult
from aisha.ui import ConsoleUI


def _render(call_name: str, result: ToolResult) -> str:
    ui = ConsoleUI(no_color=True)
    ui.console = Console(file=io.StringIO(), no_color=True, highlight=False, width=80)
    ui.on_tool_end(ToolCall(id="c1", name=call_name, arguments="{}"), result)
    return ui.console.file.getvalue()


def test_on_tool_end_failure_renders_brackets_literally():
    out = _render("run_command", ToolResult.failure("SomeError", "broken [/] and [error] text"))
    assert "[/]" in out
    assert "[error] text" in out


def test_on_tool_end_truncated_indicator_shown():
    out = _render("read_file", ToolResult.success({}, "did a thing", truncated=True))
    assert "[truncated]" in out


def test_on_tool_end_success_escaped_filename():
    out = _render("write_file", ToolResult.success({}, "[bright_cyan]a\\[/]b.txt[/], created"))
    assert "a[/]b.txt" in out
