"""Shell tool: execute PowerShell and cmd commands."""

from __future__ import annotations

__author__ = "Tischenko Alexander"

import asyncio
import os
import signal
import subprocess
import time
from pathlib import Path

from aisha.tools.base import Tool, ToolResult, ToolValidationError

# Patterns that require user confirmation
DANGEROUS_PATTERNS = [
    "rm -rf",
    "del /s",
    "erase /s",
    "rd /s",
    "rmdir /s",
    "Remove-Item -Recurse",
    "Remove-Item -Force",
    "format",
    "Format-Volume",
    "Clear-Disk",
    "reg delete",
    "shutdown",
    "Stop-Computer",
    "Restart-Computer",
    "taskkill /f",
    "git reset --hard",
    "git clean -fd",
    "git clean -fdx",
    "git push --force",
    "git push -f",
    "Invoke-WebRequest",
    "Invoke-RestMethod",
    "curl",
    "wget",
    "pip install",
    "pip uninstall",
    "npm install",
    "npm uninstall",
    "choco install",
    "winget install",
    "Set-Service",
    "New-Service",
    "sc create",
]

DANGEROUS_REASONS = {
    "rm -rf": "Рекурсивное удаление файлов",
    "del /s": "Рекурсивное удаление файлов",
    "erase /s": "Рекурсивное удаление файлов",
    "rd /s": "Рекурсивное удаление директорий",
    "rmdir /s": "Рекурсивное удаление директорий",
    "Remove-Item -Recurse": "Рекурсивное удаление файлов",
    "Remove-Item -Force": "Принудительное удаление файлов",
    "format": "Форматирование диска",
    "Format-Volume": "Форматирование тома",
    "Clear-Disk": "Очистка диска",
    "reg delete": "Удаление записей реестра",
    "shutdown": "Выключение компьютера",
    "Stop-Computer": "Выключение компьютера",
    "Restart-Computer": "Перезагрузка компьютера",
    "taskkill /f": "Принудительное завершение процесса",
    "git reset --hard": "Сброс изменений Git",
    "git clean -fd": "Удаление untracked файлов Git",
    "git clean -fdx": "Удаление untracked файлов Git (включая игнорируемые)",
    "git push --force": "Принудительный push в Git",
    "git push -f": "Принудительный push в Git",
    "Invoke-WebRequest": "Загрузка из интернета",
    "Invoke-RestMethod": "HTTP-запрос из интернета",
    "curl": "HTTP-запрос из интернета",
    "wget": "Загрузка из интернета",
    "pip install": "Установка Python-пакетов",
    "pip uninstall": "Удаление Python-пакетов",
    "npm install": "Установка npm-пакетов",
    "npm uninstall": "Удаление npm-пакетов",
    "choco install": "Установка пакетов через Chocolatey",
    "winget install": "Установка пакетов через WinGet",
    "Set-Service": "Изменение системных служб",
    "New-Service": "Создание системных служб",
    "sc create": "Создание системных служб",
}


def _check_dangerous(command: str) -> str | None:
    """Check if command matches dangerous patterns. Returns reason or None."""
    cmd_lower = command.lower()
    for pattern in DANGEROUS_PATTERNS:
        if pattern.lower() in cmd_lower:
            return DANGEROUS_REASONS.get(pattern, f"Опасная команда: содержит '{pattern}'")
    return None


class RunCommandTool(Tool):
    def __init__(self) -> None:
        self._allowed_reasons: set[str] = set()

    @property
    def name(self) -> str:
        return "run_command"

    @property
    def description(self) -> str:
        return "Execute a shell command (PowerShell or cmd)."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to execute"},
                "shell": {
                    "type": "string",
                    "enum": ["powershell", "cmd"],
                    "description": "Shell type",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory (default: workspace)",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Timeout in seconds",
                },
            },
            "required": ["command"],
        }

    def validate_args(self, args: dict) -> dict:
        if not args.get("command", "").strip():
            raise ToolValidationError("command must not be empty")
        return args

    async def execute(self, args: dict, context: dict) -> ToolResult:
        workspace: Path = context["workspace"]
        permission_mode: str = context.get("permission_mode", "ask")
        read_only: bool = context.get("read_only", False)
        shell_type: str = context.get("shell_type", "powershell")
        shell_timeout: int = context.get("shell_timeout", 120)
        confirm_fn = context.get("confirm_fn")

        command = args["command"]
        shell = args.get("shell", shell_type)
        cwd = _resolve_cwd(args.get("cwd"), workspace)
        timeout = args.get("timeout_seconds", shell_timeout)

        # Check read-only
        if read_only:
            return ToolResult.failure(
                "ToolPermissionError",
                "Режим read-only: выполнение команд запрещено",
            )

        # Check deny mode
        if permission_mode == "deny":
            return ToolResult.failure(
                "ToolPermissionError",
                "Shell команды запрещены (permission=deny)",
            )

        # Check dangerous commands (confirmation required in both "ask" and "auto")
        danger = _check_dangerous(command)
        if danger and danger not in self._allowed_reasons:
            if confirm_fn:
                confirmed = await confirm_fn(
                    command=command,
                    shell=shell,
                    cwd=str(cwd),
                    reason=danger,
                )
                if confirmed == "a":
                    self._allowed_reasons.add(danger)
                elif confirmed in ("n", None):
                    return ToolResult.failure(
                        "ToolCancelledError",
                        "Команда отменена пользователем",
                    )
            else:
                return ToolResult.failure(
                    "ToolPermissionError",
                    f"Требуется подтверждение для опасной команды: {danger}",
                )

        # Build process
        start_time = time.monotonic()
        try:
            if shell == "cmd":
                exe = "cmd.exe"
                cmd_args = ["/c", command]
            else:
                exe = "powershell.exe"
                cmd_args = ["-NoProfile", "-NonInteractive", "-Command", command]

            proc = await asyncio.create_subprocess_exec(
                exe,
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                # Kill process tree
                try:
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            capture_output=True,
                        )
                    else:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    proc.kill()
                await proc.wait()
                duration_ms = int((time.monotonic() - start_time) * 1000)
                return ToolResult.failure(
                    "ToolTimeoutError",
                    f"Команда превысила таймаут ({timeout} сек)",
                    meta={"duration_ms": duration_ms},
                )

            duration_ms = int((time.monotonic() - start_time) * 1000)
            exit_code = proc.returncode or 0
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            # Truncate if too long
            max_chars = context.get("max_output_chars", 65536)
            truncated = False
            if len(stdout) > max_chars:
                half = max_chars // 2
                stdout = stdout[:half] + "\n...\n" + stdout[-half:]
                truncated = True
            if len(stderr) > max_chars:
                half = max_chars // 2
                stderr = stderr[:half] + "\n...\n" + stderr[-half:]
                truncated = True

            return ToolResult.success(
                {
                    "exit_code": exit_code,
                    "stdout": stdout,
                    "stderr": stderr,
                },
                meta={"duration_ms": duration_ms, "truncated": truncated},
            )
        except Exception as e:
            duration_ms = int((time.monotonic() - start_time) * 1000)
            return ToolResult.from_exception(e, meta={"duration_ms": duration_ms})


def _resolve_cwd(cwd: str | None, workspace: Path) -> Path:
    if not cwd:
        return workspace
    p = Path(cwd)
    if not p.is_absolute():
        p = workspace / p
    return p.resolve()
