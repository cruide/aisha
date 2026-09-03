from aisha.tools.files import EditFileTool, GrepTool, ReadFileTool, WriteFileTool


async def test_write_read_edit(ctx):
    (ctx.workspace / "src").mkdir()
    r = await WriteFileTool().run({"path": "src/a.py", "content": "x = 1\ny = 2\n"}, ctx)
    assert r.ok and r.data["action"] == "создан"
    r = await ReadFileTool().run({"path": "src/a.py", "offset": 1, "limit": 1}, ctx)
    assert r.data["content"] == "y = 2\n" and r.data["lines_total"] == 2
    r = await EditFileTool().run({"path": "src/a.py", "old_text": "x = 1", "new_text": "x = 42"}, ctx)
    assert r.ok and (ctx.workspace / "src" / "a.py").read_text() == "x = 42\ny = 2\n"
    r = await EditFileTool().run({"path": "src/a.py", "old_text": "= ", "new_text": "=="}, ctx)
    assert not r.ok and "2 совпадений" in r.error["message"]


async def test_path_traversal_blocked(ctx):
    from aisha.errors import ToolPermissionError
    from aisha.tools.base import ToolRegistry

    reg = ToolRegistry()
    reg.register(ReadFileTool())
    r = await reg.execute("read_file", {"path": "../outside.txt"}, ctx)
    assert not r.ok and r.error["type"] == ToolPermissionError.__name__


async def test_grep_skips_excluded_dirs(ctx):
    (ctx.workspace / "node_modules").mkdir()
    (ctx.workspace / "node_modules" / "x.js").write_text("needle")
    (ctx.workspace / "a.js").write_text("needle here")
    r = await GrepTool().run({"pattern": "needle"}, ctx)
    assert [m["file"] for m in r.data["matches"]] == ["a.js"]
    