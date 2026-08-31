"""Tests for file tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from aisha.tools.files import (
    EditFileTool,
    GlobTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def ctx(workspace: Path) -> dict:
    return {
        "workspace": workspace,
        "read_only": False,
        "allow_write_outside_workspace": False,
    }


class TestReadFile:
    async def test_read_basic(self, workspace: Path, ctx: dict) -> None:
        f = workspace / "test.txt"
        f.write_text("line1\nline2\nline3\n", encoding="utf-8")
        tool = ReadFileTool()
        result = await tool.execute({"path": "test.txt"}, ctx)
        assert result.ok
        assert "line1" in result.data["content"]
        assert result.data["total_lines"] == 3

    async def test_read_with_offset(self, workspace: Path, ctx: dict) -> None:
        f = workspace / "test.txt"
        f.write_text("line1\nline2\nline3\n", encoding="utf-8")
        tool = ReadFileTool()
        result = await tool.execute({"path": "test.txt", "offset": 1, "limit": 1}, ctx)
        assert result.ok
        assert "line2" in result.data["content"]
        assert "line1" not in result.data["content"]

    async def test_read_not_found(self, workspace: Path, ctx: dict) -> None:
        tool = ReadFileTool()
        result = await tool.execute({"path": "nonexistent.txt"}, ctx)
        assert not result.ok
        assert result.error["type"] == "FileNotFoundError"


class TestWriteFile:
    async def test_write_creates_file(self, workspace: Path, ctx: dict) -> None:
        tool = WriteFileTool()
        result = await tool.execute(
            {"path": "new.txt", "content": "hello"}, ctx
        )
        assert result.ok
        assert result.data["created"] is True
        assert (workspace / "new.txt").read_text(encoding="utf-8") == "hello"

    async def test_write_atomic(self, workspace: Path, ctx: dict) -> None:
        f = workspace / "existing.txt"
        f.write_text("old", encoding="utf-8")
        tool = WriteFileTool()
        result = await tool.execute(
            {"path": "existing.txt", "content": "new"}, ctx
        )
        assert result.ok
        assert f.read_text(encoding="utf-8") == "new"
        # No tmp files left
        assert not list(workspace.glob("*.tmp"))

    async def test_write_read_only(self, workspace: Path) -> None:
        ctx = {"workspace": workspace, "read_only": True}
        tool = WriteFileTool()
        result = await tool.execute({"path": "test.txt", "content": "x"}, ctx)
        assert not result.ok
        assert "read-only" in result.error["message"].lower()

    async def test_write_outside_workspace(self, workspace: Path) -> None:
        ctx = {
            "workspace": workspace,
            "read_only": False,
            "allow_write_outside_workspace": False,
        }
        tool = WriteFileTool()
        result = await tool.execute(
            {"path": "/tmp/outside.txt", "content": "x"}, ctx
        )
        assert not result.ok
        assert "workspace" in result.error["message"].lower()


class TestEditFile:
    async def test_edit_exact_replacement(self, workspace: Path, ctx: dict) -> None:
        f = workspace / "edit.txt"
        f.write_text("hello world", encoding="utf-8")
        tool = EditFileTool()
        result = await tool.execute(
            {"path": "edit.txt", "old_text": "world", "new_text": "python"}, ctx
        )
        assert result.ok
        assert f.read_text(encoding="utf-8") == "hello python"

    async def test_edit_no_match(self, workspace: Path, ctx: dict) -> None:
        f = workspace / "edit.txt"
        f.write_text("hello", encoding="utf-8")
        tool = EditFileTool()
        result = await tool.execute(
            {"path": "edit.txt", "old_text": "xyz", "new_text": "abc"}, ctx
        )
        assert not result.ok
        assert "не найден" in result.error["message"].lower()

    async def test_edit_multiple_matches(self, workspace: Path, ctx: dict) -> None:
        f = workspace / "edit.txt"
        f.write_text("a b a b a", encoding="utf-8")
        tool = EditFileTool()
        result = await tool.execute(
            {
                "path": "edit.txt",
                "old_text": "a",
                "new_text": "x",
                "expected_replacements": 1,
            },
            ctx,
        )
        assert not result.ok
        assert "совпадений" in result.error["message"].lower()


class TestListDir:
    async def test_list_dir(self, workspace: Path, ctx: dict) -> None:
        (workspace / "a.txt").write_text("a", encoding="utf-8")
        sub = workspace / "subdir"
        sub.mkdir()
        tool = ListDirTool()
        result = await tool.execute({"path": "."}, ctx)
        assert result.ok
        names = [e["name"] for e in result.data["entries"]]
        assert "a.txt" in names
        assert "subdir" in names

    async def test_list_dir_not_found(self, workspace: Path, ctx: dict) -> None:
        tool = ListDirTool()
        result = await tool.execute({"path": "nonexistent"}, ctx)
        assert not result.ok


class TestGlob:
    async def test_glob_pattern(self, workspace: Path, ctx: dict) -> None:
        (workspace / "a.py").write_text("a", encoding="utf-8")
        (workspace / "b.py").write_text("b", encoding="utf-8")
        (workspace / "c.txt").write_text("c", encoding="utf-8")
        tool = GlobTool()
        result = await tool.execute({"pattern": "*.py"}, ctx)
        assert result.ok
        assert len(result.data["matches"]) == 2

    async def test_glob_excludes_dirs(self, workspace: Path, ctx: dict) -> None:
        git_dir = workspace / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("x", encoding="utf-8")
        (workspace / "src.py").write_text("s", encoding="utf-8")
        tool = GlobTool()
        result = await tool.execute({"pattern": "**/*"}, ctx)
        assert result.ok
        for m in result.data["matches"]:
            assert ".git" not in m


class TestGrep:
    async def test_grep_search(self, workspace: Path, ctx: dict) -> None:
        (workspace / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
        (workspace / "b.py").write_text("def bar():\n    pass\n", encoding="utf-8")
        tool = GrepTool()
        result = await tool.execute({"pattern": "def foo"}, ctx)
        assert result.ok
        assert result.data["count"] == 1
        assert "foo" in result.data["matches"][0]["content"]

    async def test_grep_with_include(self, workspace: Path, ctx: dict) -> None:
        (workspace / "a.py").write_text("test", encoding="utf-8")
        (workspace / "b.txt").write_text("test", encoding="utf-8")
        tool = GrepTool()
        result = await tool.execute({"pattern": "test", "include": "*.py"}, ctx)
        assert result.ok
        assert result.data["count"] == 1

    async def test_grep_invalid_regex(self, workspace: Path, ctx: dict) -> None:
        tool = GrepTool()
        result = await tool.execute({"pattern": "[invalid"}, ctx)
        assert not result.ok
        assert "regex" in result.error["type"].lower()
