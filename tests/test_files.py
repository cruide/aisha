# Author: Tischenko A. (https://github.com/cruide)
from aisha.tools.files import (
    EditFileTool,
    GlobTool,
    GrepTool,
    ReadFileTool,
    WriteFileTool,
    _glob_match,
)


async def test_write_read_edit(ctx):
    (ctx.workspace / "src").mkdir()
    r = await WriteFileTool().run({"path": "src/a.py", "content": "x = 1\ny = 2\n"}, ctx)
    assert r.ok and r.data["action"] == "created"
    r = await ReadFileTool().run({"path": "src/a.py", "offset": 1, "limit": 1}, ctx)
    assert r.data["content"] == "y = 2\n" and r.data["lines_total"] == 2
    r = await EditFileTool().run(
        {"path": "src/a.py", "old_text": "x = 1", "new_text": "x = 42"}, ctx
    )
    assert r.ok and (ctx.workspace / "src" / "a.py").read_text() == "x = 42\ny = 2\n"
    r = await EditFileTool().run({"path": "src/a.py", "old_text": "= ", "new_text": "=="}, ctx)
    assert not r.ok and "2 matches" in r.error["message"]


async def test_path_traversal_blocked(ctx):
    from aisha.errors import ToolPermissionError
    from aisha.tools.base import ToolRegistry

    reg = ToolRegistry()
    reg.register(ReadFileTool())
    r = await reg.execute("read_file", {"path": "../outside.txt"}, ctx)
    assert not r.ok and r.error["type"] == ToolPermissionError.__name__


async def test_glob_blocks_paths_outside_workspace(ctx):
    outside = ctx.workspace.parent / "secret.env"
    outside.write_text("SECRET", encoding="utf-8")
    (ctx.workspace / "inside.py").write_text("ok", encoding="utf-8")
    r = await GlobTool().run({"pattern": "../secret.env"}, ctx)
    assert r.ok and r.data["files"] == []
    r = await GlobTool().run({"pattern": str(outside)}, ctx)
    assert not r.ok or r.data["files"] == []
    r = await GlobTool().run({"pattern": "*.py"}, ctx)
    assert r.data["files"] == ["inside.py"]


async def test_read_file_window_on_large_file(ctx):
    lines = [f"line {i:05d} " + "x" * 65 + "\n" for i in range(7000)]
    (ctx.workspace / "big.txt").write_text("".join(lines), encoding="utf-8")
    r = await ReadFileTool().run({"path": "big.txt", "offset": 100, "limit": 20}, ctx)
    assert r.ok
    assert r.data["returned"] == 20
    assert r.data["offset"] == 100
    assert r.data["total_known"] is False
    content = r.data["content"]
    assert len(content.splitlines()) == 20
    assert content.startswith("line 00100")
    assert r.meta["truncated"] is True


async def test_read_file_small_file_total_known(ctx):
    (ctx.workspace / "small.txt").write_text("a\nb\nc\n", encoding="utf-8")
    r = await ReadFileTool().run({"path": "small.txt"}, ctx)
    assert r.data["total_known"] is True
    assert r.data["lines_total"] == 3
    assert r.data["content"] == "a\nb\nc\n"


async def test_read_file_binary_detected(ctx):
    (ctx.workspace / "bin.txt").write_bytes(b"hello\x00world")
    r = await ReadFileTool().run({"path": "bin.txt"}, ctx)
    assert not r.ok and "binary" in r.error["message"]


async def test_edit_file_crlf_preserved(ctx):
    (ctx.workspace / "crlf.txt").write_bytes(b"line1\r\nline2\r\n")
    r = await EditFileTool().run(
        {"path": "crlf.txt", "old_text": "line1\nline2", "new_text": "foo\nbar"}, ctx
    )
    assert r.ok
    assert (ctx.workspace / "crlf.txt").read_bytes() == b"foo\r\nbar\r\n"


async def test_edit_file_near_match_hint_no_change(ctx):
    (ctx.workspace / "f.txt").write_text("hello world  \nfoo\n", encoding="utf-8")
    r = await EditFileTool().run(
        {"path": "f.txt", "old_text": "hello world\nfoo", "new_text": "x"}, ctx
    )
    assert not r.ok
    assert "near-match" in r.error["message"]
    assert (ctx.workspace / "f.txt").read_text() == "hello world  \nfoo\n"


async def test_edit_file_multiple_near_matches_not_unique(ctx):
    (ctx.workspace / "f.txt").write_text("abc  \ndef  \nabc  \ndef  \n", encoding="utf-8")
    r = await EditFileTool().run(
        {"path": "f.txt", "old_text": "abc\ndef", "new_text": "x"}, ctx
    )
    assert not r.ok
    assert "near-matches" in r.error["message"]


async def test_edit_file_too_large(ctx):
    ctx.config.tools.max_output_chars = 100
    big = "x" * (100 * 16 + 1)
    (ctx.workspace / "big.txt").write_text(big, encoding="utf-8")
    r = await EditFileTool().run(
        {"path": "big.txt", "old_text": "y", "new_text": "z"}, ctx
    )
    assert not r.ok
    assert "edit_file" in r.error["message"]
    assert "offset" not in r.error["message"]


async def test_grep_skips_excluded_dirs(ctx):
    (ctx.workspace / "node_modules").mkdir()
    (ctx.workspace / "node_modules" / "x.js").write_text("needle")
    (ctx.workspace / "a.js").write_text("needle here")
    r = await GrepTool().run({"pattern": "needle"}, ctx)
    assert [m["file"] for m in r.data["matches"]] == ["a.js"]


def test_glob_match_patterns():
    assert _glob_match("a.py", "*.py")
    assert not _glob_match("src/a.py", "*.py")
    assert _glob_match("src/a.py", "src/**/*.py")
    assert _glob_match("src/sub/a.py", "src/**/*.py")
    assert _glob_match("a.py", "**/*.py")
    assert _glob_match("a.py", "a.py")
    assert not _glob_match("a.py", "b.py")
    assert not _glob_match("a.py", "src/**/*.py")


async def test_glob_prunes_excluded_dirs(ctx):
    (ctx.workspace / "src").mkdir()
    (ctx.workspace / "src" / "a.py").write_text("")
    (ctx.workspace / "node_modules" / "huge").mkdir(parents=True)
    (ctx.workspace / "node_modules" / "huge" / "a.py").write_text("")
    r = await GlobTool().run({"pattern": "**/*.py"}, ctx)
    assert r.data["files"] == ["src/a.py"]
    r2 = await GlobTool().run({"pattern": "**/*.py", "include_ignored": True}, ctx)
    assert set(r2.data["files"]) == {"src/a.py", "node_modules/huge/a.py"}


async def test_grep_skips_external_file_symlink(ctx):
    outside = ctx.workspace.parent / "external.txt"
    outside.write_text("UNIQUEMARKER42", encoding="utf-8")
    (ctx.workspace / "inside.txt").write_text("needle", encoding="utf-8")
    link = ctx.workspace / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        import pytest

        pytest.skip("symlinks not available")
    r = await GrepTool().run({"pattern": "UNIQUEMARKER42"}, ctx)
    assert r.data["matches"] == []
