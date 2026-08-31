"""Tests for shell tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from aisha.tools.shell import RunCommandTool, _check_dangerous


class TestCheckDangerous:
    def test_rm_rf(self) -> None:
        assert _check_dangerous("rm -rf /tmp") is not None

    def test_del_s(self) -> None:
        assert _check_dangerous("del /s /q C:\\temp") is not None

    def test_remove_item_recurse(self) -> None:
        assert _check_dangerous("Remove-Item -Recurse -Force C:\\temp") is not None

    def test_format(self) -> None:
        assert _check_dangerous("format D:") is not None

    def test_shutdown(self) -> None:
        assert _check_dangerous("shutdown /s") is not None

    def test_git_reset_hard(self) -> None:
        assert _check_dangerous("git reset --hard HEAD") is not None

    def test_git_push_force(self) -> None:
        assert _check_dangerous("git push --force origin main") is not None

    def test_safe_command(self) -> None:
        assert _check_dangerous("echo hello") is None

    def test_safe_ls(self) -> None:
        assert _check_dangerous("ls -la") is None

    def test_pip_install(self) -> None:
        assert _check_dangerous("pip install requests") is not None


class TestRunCommand:
    async def test_run_command_powershell(self, tmp_path: Path) -> None:
        ctx = {
            "workspace": tmp_path,
            "permission_mode": "ask",
            "read_only": False,
            "shell_type": "powershell",
            "shell_timeout": 30,
            "max_output_chars": 65536,
            "ask_fn": None,
        }
        tool = RunCommandTool()
        result = await tool.execute(
            {"command": "Write-Output 'hello'", "shell": "powershell"}, ctx
        )
        assert result.ok
        assert "hello" in result.data["stdout"]

    async def test_run_command_cmd(self, tmp_path: Path) -> None:
        ctx = {
            "workspace": tmp_path,
            "permission_mode": "ask",
            "read_only": False,
            "shell_type": "powershell",
            "shell_timeout": 30,
            "max_output_chars": 65536,
            "ask_fn": None,
        }
        tool = RunCommandTool()
        result = await tool.execute(
            {"command": "echo hello", "shell": "cmd"}, ctx
        )
        assert result.ok
        assert "hello" in result.data["stdout"]

    async def test_run_command_read_only(self, tmp_path: Path) -> None:
        ctx = {
            "workspace": tmp_path,
            "permission_mode": "ask",
            "read_only": True,
            "shell_type": "powershell",
        }
        tool = RunCommandTool()
        result = await tool.execute({"command": "echo test"}, ctx)
        assert not result.ok
        assert "read-only" in result.error["message"].lower()

    async def test_run_command_deny(self, tmp_path: Path) -> None:
        ctx = {
            "workspace": tmp_path,
            "permission_mode": "deny",
            "read_only": False,
            "shell_type": "powershell",
        }
        tool = RunCommandTool()
        result = await tool.execute({"command": "echo test"}, ctx)
        assert not result.ok
        assert "deny" in result.error["message"].lower()

    async def test_run_command_stderr(self, tmp_path: Path) -> None:
        ctx = {
            "workspace": tmp_path,
            "permission_mode": "ask",
            "read_only": False,
            "shell_type": "powershell",
            "shell_timeout": 30,
            "max_output_chars": 65536,
            "ask_fn": None,
        }
        tool = RunCommandTool()
        result = await tool.execute(
            {"command": "Write-Error 'oops'", "shell": "powershell"}, ctx
        )
        # PowerShell Write-Error still exits 0 but writes to stderr
        assert result.ok

    async def test_run_command_empty(self, tmp_path: Path) -> None:
        tool = RunCommandTool()
        with pytest.raises(Exception):
            tool.validate_args({"command": ""})

    async def test_auto_mode_dangerous_requires_confirmation(self, tmp_path: Path) -> None:
        ctx = {
            "workspace": tmp_path,
            "permission_mode": "auto",
            "read_only": False,
            "shell_type": "powershell",
            "confirm_fn": None,
        }
        tool = RunCommandTool()
        result = await tool.execute({"command": "shutdown /s"}, ctx)
        assert not result.ok
        assert "подтверждени" in result.error["message"].lower()

    async def test_dangerous_confirm_deny(self, tmp_path: Path) -> None:
        async def confirm(**kw: object) -> str:
            return "n"

        ctx = {
            "workspace": tmp_path,
            "permission_mode": "auto",
            "read_only": False,
            "shell_type": "powershell",
            "confirm_fn": confirm,
        }
        tool = RunCommandTool()
        result = await tool.execute({"command": "shutdown /s"}, ctx)
        assert not result.ok
        assert result.error["type"] == "ToolCancelledError"

    async def test_dangerous_confirm_allow(self, tmp_path: Path) -> None:
        async def confirm(**kw: object) -> str:
            return "y"

        ctx = {
            "workspace": tmp_path,
            "permission_mode": "auto",
            "read_only": False,
            "shell_type": "powershell",
            "shell_timeout": 30,
            "max_output_chars": 65536,
            "confirm_fn": confirm,
        }
        tool = RunCommandTool()
        result = await tool.execute({"command": "echo format"}, ctx)
        assert result.ok
        assert "format" in result.data["stdout"]

    async def test_dangerous_allow_remembered_for_session(self, tmp_path: Path) -> None:
        calls: list[str] = []

        async def confirm(**kw: object) -> str:
            calls.append("asked")
            return "a"

        ctx = {
            "workspace": tmp_path,
            "permission_mode": "auto",
            "read_only": False,
            "shell_type": "powershell",
            "shell_timeout": 30,
            "max_output_chars": 65536,
            "confirm_fn": confirm,
        }
        tool = RunCommandTool()
        first = await tool.execute({"command": "echo format"}, ctx)
        second = await tool.execute({"command": "echo format"}, ctx)
        assert first.ok
        assert second.ok
        assert len(calls) == 1
