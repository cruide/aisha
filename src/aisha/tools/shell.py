# Author: Tischenko A. (https://github.com/cruide)
"""run_command: PowerShell / cmd execution with timeouts, output limits and confirmations."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
from typing import Any

from aisha.errors import ToolPermissionError, ToolTimeoutError, ToolValidationError
from aisha.tools.base import ConfirmRequest, Tool, ToolContext, ToolResult, require_confirmation
from aisha.tools.files import resolve_path

# (regex, human reason). Checked case-insensitively; not a sandbox — an extra safety net.
DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"\brm\s+(-[a-z]*r[a-z]*f|-[a-z]*f[a-z]*r)\b", "recursive deletion (rm -rf)"),
    (r"\b(del|erase)\b[^&|\n]*\s/s\b", "recursive deletion (del /s)"),
    (r"\b(rd|rmdir)\b[^&|\n]*\s/s\b", "recursive directory deletion (rd /s)"),
    (r"\bremove-item\b[^&|\n]*-recurse", "Remove-Item -Recurse"),
    (r"(^|[\s;&|])format(\.com|\.exe)?\s", "disk formatting"),
    (r"\bformat-volume\b", "volume formatting"),
    (r"\bclear-disk\b", "disk clearing"),
    (r"\breg(\.exe)?\s+delete\b", "registry key deletion"),
    (r"\b(shutdown|stop-computer|restart-computer)\b", "shutdown/restart"),
    (r"\btaskkill\b[^&|\n]*\s/f\b", "forced process termination"),
    (r"\bgit\s+reset\s+--hard\b", "git reset --hard"),
    (r"\bgit\s+clean\s+-[a-z]*[fx]", "git clean -f"),
    (r"\bgit\s+push\b[^&|\n]*(--force\b|\s-f\b|\s\+\S+)", "forced push"),
    (r"\bgit\s+branch\s+-D\b", "forced branch deletion"),
    (r"\b(iwr|irm|invoke-webrequest|invoke-restmethod|curl|wget)\b[^&|\n]*\|\s*"
     r"(iex|invoke-expression|sh|bash|pwsh|powershell|cmd)\b", "executing script from network"),
    (r"\b(winget|choco|scoop|msiexec|apt(-get)?|yum|dnf|brew)\s+(install|remove|uninstall)\b",
     "installing/removing system packages"),
    (r"\b(sc(\.exe)?\s+(delete|stop|config)|set-service|stop-service|remove-service|"
     r"new-service|net\s+stop)\b", "modifying system services"),
    (r"\b(password|passwd|token|secret|api[_-]?key)\s*[=:]\s*\S+", "possible secret in command"),
    (r"\b(sk|ghp|gho|glpat|xox[abp])[-_][A-Za-z0-9_-]{16,}", "looks like an access token"),
    (r"\b(pip3?|npm|pnpm|yarn)\s+(install|add)\b", "package installation"),
    (r"\bcomposer\s+require\b", "package installation (composer)"),
    (r"\bgit\s+(checkout|restore)\s+\.(?!\S)", "discarding local changes (git checkout/restore .)"),
    (r"\bgit\s+stash\s+(drop|clear)\b", "dropping stashed changes"),
    (r"\b(invoke-expression|iex)\b", "executing an expression (Invoke-Expression)"),
    (r"\bpowershell\b[^&|\n]*-enc(odedcommand)?\b", "encoded PowerShell command"),
    (r"\bfrombase64string\b", "decoding base64 payload"),
    (r"\bcmd(\.exe)?\s+/c\b", "nested cmd shell"),
    (r"\b(powershell|pwsh)\b[^&|\n]*-(c|command)\b", "nested PowerShell shell"),
]
_COMPILED = [(re.compile(p, re.IGNORECASE), reason) for p, reason in DANGEROUS_PATTERNS]

PS_PRELUDE = (
    "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
    "$OutputEncoding=[System.Text.Encoding]::UTF8; $ErrorActionPreference='Continue'; "
)


def find_danger(command: str) -> str | None:
    """Return the reason if the command matches a dangerous pattern."""
    for regex, reason in _COMPILED:
        if regex.search(command):
            return reason
    return None


def build_argv(shell: str, command: str) -> list[str]:
    if os.name != "nt":
        return ["/bin/sh", "-c", command]
    if shell == "cmd":
        return ["cmd.exe", "/d", "/s", "/c", f"chcp 65001>nul & {command}"]
    return ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
            "-Command", PS_PRELUDE + command]


def truncate_output(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    head = max_chars * 2 // 3
    tail = max_chars - head
    skipped = len(text) - head - tail
    return f"{text[:head]}\n… [{skipped} chars skipped] …\n{text[-tail:]}", True


MAX_SHELL_TIMEOUT = 600.0


def normalize_timeout(raw: Any, default: float) -> float:
    """Normalize a model-supplied timeout: non-positive is invalid, otherwise cap at 600 s."""
    if raw is None:
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ToolValidationError(f"Invalid timeout_seconds: {raw!r}") from exc
    if value <= 0:
        raise ToolValidationError("timeout_seconds must be greater than zero")
    return min(value, MAX_SHELL_TIMEOUT)


async def _bounded_read(stream: Any, limit: int) -> tuple[bytes, bool]:
    """Read at most `limit` bytes from an async stream; returns (data, truncated)."""
    buf = bytearray()
    while len(buf) < limit:
        chunk = await stream.read(min(65536, limit - len(buf)))
        if not chunk:
            return bytes(buf), False
        buf.extend(chunk)
    return bytes(buf), True


async def kill_tree(proc: asyncio.subprocess.Process) -> None:
    """Terminate the process and all of its children."""
    if proc.returncode is not None:
        return
    if os.name == "nt":
        try:
            killer = await asyncio.create_subprocess_exec(
                "taskkill", "/F", "/T", "/PID", str(proc.pid),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.wait(), 10)
        except (OSError, asyncio.TimeoutError):
            pass
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(proc.wait(), 5)
    except asyncio.TimeoutError:
        pass


async def run_process(
    argv: list[str], cwd: str, timeout: float, stream_limit: int
) -> tuple[int, bytes, bytes, bool]:
    """Run a command, draining stdout/stderr concurrently with a hard byte limit.

    Returns (returncode, stdout, stderr, truncated). Both pipes are drained in
    parallel so a full stderr cannot block the child; when either stream exceeds
    `stream_limit` the process tree is killed and accumulated bytes are returned.
    """
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=cwd, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **kwargs,
    )
    out_buf = bytearray()
    err_buf = bytearray()
    truncated: list[bool] = []

    async def drain() -> None:
        out, err = await asyncio.gather(
            _bounded_read(proc.stdout, stream_limit),
            _bounded_read(proc.stderr, stream_limit),
        )
        out_buf.extend(out[0])
        err_buf.extend(err[0])
        truncated.append(out[1] or err[1])
        if out[1] or err[1]:
            await kill_tree(proc)
        else:
            await proc.wait()

    try:
        await asyncio.wait_for(drain(), timeout=timeout)
    except asyncio.TimeoutError:
        await kill_tree(proc)
        raise ToolTimeoutError(f"Command did not finish within {timeout:g} s and was stopped")
    except asyncio.CancelledError:
        await kill_tree(proc)
        raise
    return proc.returncode or 0, bytes(out_buf), bytes(err_buf), bool(truncated and truncated[0])


class RunCommandTool(Tool):
    name = "run_command"
    description = (
        "Run one non-interactive command and return stdout, stderr and exit_code. Example: run_command(command=\"pytest\")."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "shell": {"type": "string", "enum": ["powershell", "cmd"]},
            "cwd": {"type": "string", "description": "Working directory, default workspace"},
            "timeout_seconds": {"type": "integer"},
        },
        "required": ["command"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        cfg = ctx.config.tools
        if not cfg.shell or cfg.permission == "deny":
            raise ToolPermissionError("Shell command execution is disabled by settings")
        command: str = args["command"].strip()
        if not command:
            return ToolResult.failure("ToolValidationError", "Empty command")
        shell = args.get("shell") or cfg.shell_type
        cwd, _ = resolve_path(args.get("cwd") or ".", ctx, write=False)
        if not cwd.is_dir():
            return ToolResult.failure("NotADirectoryError", f"Directory not found: {cwd}")
        timeout = normalize_timeout(args.get("timeout_seconds"), cfg.shell_timeout)

        danger = find_danger(command)
        if danger is not None:
            key = f"shell:danger:{danger}"
            reason = f"dangerous command — {danger}"
        else:
            key = "shell:any"
            reason = "permission=ask mode"
        if danger is not None or cfg.permission == "ask":
            await require_confirmation(ctx, ConfirmRequest(
                title="Run command",
                details=[("Command", command), ("Shell", shell), ("Working directory", str(cwd))],
                reason=reason,
                key=key,
            ))

        code, out, err, stream_truncated = await run_process(
            build_argv(shell, command), str(cwd), timeout, 4 * cfg.max_output_chars
        )
        stdout, t1 = truncate_output(out.decode("utf-8", "replace"), cfg.max_output_chars)
        stderr, t2 = truncate_output(err.decode("utf-8", "replace"), cfg.max_output_chars // 4)
        lines = stdout.count("\n") + (1 if stdout and not stdout.endswith("\n") else 0)
        summary = f"exit {code}, {lines} lines of output"
        result = ToolResult.success(
            {"exit_code": code, "stdout": stdout, "stderr": stderr, "shell": shell},
            summary, truncated=stream_truncated or t1 or t2,
        )
        if code != 0:
            result.summary = f"exit {code}" + (f": {stderr.strip().splitlines()[-1][:120]}"
                                              if stderr.strip() else "")
        return result
