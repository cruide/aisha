# Author: Tischenko A. (https://github.com/cruide)
import io
from types import SimpleNamespace

from rich.console import Console

from aisha.client import ToolCall
from aisha.tools.base import ToolResult
from aisha.ui import ConsoleUI, fmt_sampling


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


def _banner(llm) -> str:
    ui = ConsoleUI(no_color=True)
    ui.config = SimpleNamespace(
        llm=llm,
        read_only=False,
        tools=SimpleNamespace(permission="ask", shell_type="powershell"),
        workspace="D:\\ws",
        server=SimpleNamespace(base_url="http://localhost:8088"),
    )
    ui.console = Console(file=io.StringIO(), no_color=True, highlight=False, width=200)
    ui.print_banner("test-model")
    return ui.console.file.getvalue()


def test_banner_shows_sampling_params():
    llm = SimpleNamespace(temperature=0.3, top_p=0.9, top_k=30, repeat_penalty=1.03,
                          frequency_penalty=0.5, context_window=32768)
    out = _banner(llm)
    assert "Temp: 0.3" in out
    assert "Top-p: 0.9" in out
    assert "Top-k: 30" in out
    assert "Repeat penalty: 1.03" in out
    assert "Frequency penalty: 0.5" in out


def test_banner_shows_server_for_none_sampling():
    llm = SimpleNamespace(temperature=0.3, top_p=None, top_k=None, repeat_penalty=None,
                          frequency_penalty=None, context_window=32768)
    out = _banner(llm)
    assert "Top-p: server" in out
    assert "Top-k: server" in out
    assert "Repeat penalty: server" in out
    assert "Frequency penalty: server" in out


def test_fmt_sampling_formats_values():
    llm = SimpleNamespace(temperature=0.3, top_p=0.9, top_k=30, repeat_penalty=1.03,
                          frequency_penalty=0.5)
    assert fmt_sampling(llm) == (
        "Temp: 0.3 · Top-p: 0.9 · Top-k: 30 · Repeat penalty: 1.03 · "
        "Frequency penalty: 0.5"
    )
