# Author: Tischenko A. (https://github.com/cruide)
"""File tools: read_file, write_file, edit_file, list_dir, glob, grep."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from aisha.errors import ToolPermissionError, ToolValidationError
from aisha.fsutil import atomic_write_text, human_size, is_inside
from aisha.tools.base import ConfirmRequest, Tool, ToolContext, ToolResult, require_confirmation

DEFAULT_EXCLUDES = frozenset({
    ".git", ".hg", ".svn", ".idea", ".vscode", "node_modules", "vendor",
    "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".ruff_cache",
})
MAX_GREP_FILE_BYTES = 2 * 1024 * 1024


def resolve_path(raw: str, ctx: ToolContext, *, write: bool) -> tuple[Path, bool]:
    """Resolve `raw` against the workspace. Returns (path, inside_workspace)."""
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ctx.workspace / path
    try:
        resolved = path.resolve()
    except OSError:
        resolved = Path(os.path.normpath(path))
    inside = is_inside(resolved, ctx.workspace)
    if not inside:
        allowed = (
            ctx.config.tools.allow_write_outside_workspace if write
            else ctx.config.tools.allow_read_outside_workspace
        )
        if not allowed:
            raise ToolPermissionError(f"Path outside workspace: {resolved}")
    return resolved, inside


def _read_text(path: Path, limit_bytes: int) -> str:
    size = path.stat().st_size
    if size > limit_bytes * 4:
        raise ToolValidationError(
            f"File is too large ({human_size(size)}); use offset/limit or grep"
        )
    data = path.read_bytes()
    if b"\x00" in data[:8192]:
        raise ToolValidationError("File appears to be binary")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


async def _confirm_outside_write(ctx: ToolContext, path: Path, action: str) -> None:
    await require_confirmation(ctx, ConfirmRequest(
        title="Write outside workspace",
        details=[("Action", action), ("Path", str(path))],
        reason="file is outside the workspace directory",
        key=f"write_outside:{path.parent}",
    ))


def _rel(path: Path, ctx: ToolContext) -> str:
    try:
        return path.relative_to(ctx.workspace).as_posix() or "."
    except ValueError:
        return str(path)


def _skip_dir(name: str, include_ignored: bool) -> bool:
    return not include_ignored and name in DEFAULT_EXCLUDES


class ReadFileTool(Tool):
    name = "read_file"
    read_only = True
    description = (
        "Read a text file (UTF-8). Required argument: path — file path relative to the workspace. "
        "Optional: offset (first line number, 0-based) and limit (how many lines to return, "
        "default 500). Always read a file with this tool before editing it. "
        "Example: read_file(path=\"src/main.py\", offset=0, limit=100)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path (relative to workspace)"},
            "offset": {"type": "integer", "description": "First line, 0-based"},
            "limit": {"type": "integer", "description": "Max lines to return"},
        },
        "required": ["path"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path, _ = resolve_path(args["path"], ctx, write=False)
        if not path.is_file():
            return ToolResult.failure("FileNotFoundError", f"File not found: {args['path']}")
        max_chars = ctx.config.tools.max_output_chars
        text = _read_text(path, max_chars)
        lines = text.splitlines(keepends=True)
        offset = max(0, int(args.get("offset", 0)))
        limit = max(1, int(args.get("limit", 500)))
        chunk = lines[offset:offset + limit]
        content = "".join(chunk)
        truncated = offset + len(chunk) < len(lines)
        if len(content) > max_chars:
            content, truncated = content[:max_chars], True
        data = {
            "path": _rel(path, ctx), "content": content, "lines_total": len(lines),
            "offset": offset, "returned": len(chunk), "size_bytes": path.stat().st_size,
        }
        summary = f"{len(lines)} lines, {human_size(path.stat().st_size)}"
        if truncated:
            summary += f" (showing {len(chunk)} from {offset})"
        return ToolResult.success(data, summary, truncated=truncated)


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Create a new file or fully overwrite an existing one (atomically). Required arguments: "
        "path (file path) and content (full file contents as a single string). "
        "Optional: create_dirs=true creates parent directories. For new files use write_file, "
        "for targeted edits of existing files — edit_file. Do not write files longer than ~300 "
        "lines in a single call: output is limited by tokens and will be truncated mid-way — "
        "write large files in parts (skeleton with write_file, then extend with edit_file). "
        "Example: write_file(path=\"notes.txt\", content=\"Hello\\n\")."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "create_dirs": {"type": "boolean",
                            "description": "Create parent directories (default true)"},
        },
        "required": ["path", "content"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path, inside = resolve_path(args["path"], ctx, write=True)
        if not inside:
            await _confirm_outside_write(ctx, path, "write_file")
        if path.is_dir():
            return ToolResult.failure("IsADirectoryError", f"Path is a directory: {args['path']}")
        if not path.parent.exists():
            if args.get("create_dirs", True):
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                return ToolResult.failure("FileNotFoundError",
                                          f"Directory does not exist: {path.parent}")
        existed = path.exists()
        content: str = args["content"]
        atomic_write_text(path, content)
        action = "overwritten" if existed else "created"
        data = {"path": _rel(path, ctx), "action": action, "bytes": len(content.encode("utf-8")),
                "lines": content.count("\n") + (1 if content and not content.endswith("\n") else 0)}
        return ToolResult.success(data, f"{action}, {data['lines']} lines")


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Targeted text replacement in an existing file. Required arguments: "
        "path (file path), old_text (exact fragment to replace, copied verbatim from read_file "
        "including indentation and line breaks) and new_text (replacement for old_text). "
        "old_text must occur exactly expected_replacements times (default 1), otherwise the "
        "file will not be changed. First read the file via read_file, then copy the exact "
        "fragment into old_text. Example: edit_file(path=\"src/app.py\", "
        "old_text=\"return 1\", new_text=\"return 2\")."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path (relative to workspace)"},
            "old_text": {"type": "string",
                         "description": "Exact fragment to replace (verbatim copy from file)"},
            "new_text": {"type": "string",
                         "description": "New text to replace old_text with"},
            "expected_replacements": {"type": "integer",
                                      "description": "How many times old_text should occur"},
        },
        "required": ["path", "old_text", "new_text"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path, inside = resolve_path(args["path"], ctx, write=True)
        if not inside:
            await _confirm_outside_write(ctx, path, "edit_file")
        if not path.is_file():
            return ToolResult.failure("FileNotFoundError", f"File not found: {args['path']}")
        old, new = args["old_text"], args["new_text"]
        if not old:
            return ToolResult.failure("ToolValidationError", "old_text cannot be empty")
        expected = int(args.get("expected_replacements", 1))
        text = _read_text(path, ctx.config.tools.max_output_chars * 4)
        count = text.count(old)
        if count == 0:
            return ToolResult.failure("ToolValidationError",
                                      "old_text not found in file; file unchanged")
        if count != expected:
            return ToolResult.failure(
                "ToolValidationError",
                f"Found {count} matches, expected {expected}; file unchanged. "
                "Refine old_text or set expected_replacements.",
            )
        atomic_write_text(path, text.replace(old, new))
        return ToolResult.success({"path": _rel(path, ctx), "replacements": count},
                                  f"{count} replacement(s)")


class ListDirTool(Tool):
    name = "list_dir"
    read_only = True
    description = (
        "List directory contents (names, type, size). Optional argument path "
        "(directory, default '.'). show_hidden=true shows hidden files; limit — max entries. "
        "Example: list_dir(path=\"src\")."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory, default '.'"},
            "show_hidden": {"type": "boolean"},
            "limit": {"type": "integer"},
        },
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path, _ = resolve_path(args.get("path", "."), ctx, write=False)
        if not path.is_dir():
            return ToolResult.failure("NotADirectoryError", f"Directory not found: {path}")
        limit = int(args.get("limit", 500))
        show_hidden = bool(args.get("show_hidden", False))
        entries: list[dict[str, Any]] = []
        with os.scandir(path) as it:
            items = sorted(it, key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
        for entry in items:
            if not show_hidden and entry.name.startswith("."):
                continue
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
                size = 0 if is_dir else entry.stat(follow_symlinks=False).st_size
            except OSError:
                is_dir, size = False, 0
            entries.append({"name": entry.name, "type": "dir" if is_dir else "file", "size": size})
        truncated = len(entries) > limit
        entries = entries[:limit]
        dirs = sum(1 for e in entries if e["type"] == "dir")
        return ToolResult.success(
            {"path": _rel(path, ctx), "entries": entries},
            f"{dirs} dirs, {len(entries) - dirs} files", truncated=truncated,
        )


class GlobTool(Tool):
    name = "glob"
    read_only = True
    description = (
        "Find files by glob pattern. Required argument: pattern, e.g. '**/*.py' or "
        "'src/**/*.php'. Optional: path — search base (default '.'). Returns a list of "
        "file paths. Example: glob(pattern=\"src/**/*.py\")."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "Search base, default '.'"},
            "limit": {"type": "integer"},
            "include_ignored": {"type": "boolean",
                                "description": "Do not exclude .git, node_modules, vendor, etc."},
        },
        "required": ["pattern"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        base, _ = resolve_path(args.get("path", "."), ctx, write=False)
        if not base.is_dir():
            return ToolResult.failure("NotADirectoryError", f"Directory not found: {base}")
        limit = int(args.get("limit", 200))
        include_ignored = bool(args.get("include_ignored", False))
        found: list[str] = []
        truncated = False
        allow_outside = ctx.config.tools.allow_read_outside_workspace
        try:
            for path in base.glob(args["pattern"]):
                try:
                    resolved = path.resolve()
                except OSError:
                    continue
                if not is_inside(resolved, ctx.workspace) and not allow_outside:
                    continue
                try:
                    rel_parts = resolved.relative_to(base).parts
                except ValueError:
                    rel_parts = ()
                if any(_skip_dir(p, include_ignored) for p in rel_parts[:-1]):
                    continue
                if not resolved.is_file():
                    continue
                if len(found) >= limit:
                    truncated = True
                    break
                found.append(_rel(resolved, ctx))
        except (ValueError, NotImplementedError, OSError) as exc:
            return ToolResult.failure("ToolValidationError", f"Invalid pattern: {exc}")
        found.sort()
        return ToolResult.success({"files": found, "count": len(found)},
                                  f"found {len(found)} files", truncated=truncated)


class GrepTool(Tool):
    name = "grep"
    read_only = True
    description = (
        "Regex search in file contents. Required argument: pattern (Python re regular "
        "expression). Optional: path (file or directory, default '.'), include (file name "
        "glob, e.g. '*.py'), ignore_case=true. Returns file, line number and match text. "
        "Example: grep(pattern=\"def foo\", include=\"*.py\", path=\"src\")."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression (Python re)"},
            "path": {"type": "string", "description": "File or directory, default '.'"},
            "include": {"type": "string", "description": "File glob, e.g. '*.py'"},
            "ignore_case": {"type": "boolean"},
            "limit": {"type": "integer", "description": "Max matches (default 100)"},
            "include_ignored": {"type": "boolean"},
        },
        "required": ["pattern"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            regex = re.compile(args["pattern"], re.IGNORECASE if args.get("ignore_case") else 0)
        except re.error as exc:
            return ToolResult.failure("ToolValidationError", f"Invalid regex: {exc}")
        root, _ = resolve_path(args.get("path", "."), ctx, write=False)
        include = args.get("include") or "*"
        limit = int(args.get("limit", 100))
        include_ignored = bool(args.get("include_ignored", False))
        matches: list[dict[str, Any]] = []
        files_scanned = 0
        truncated = False

        def iter_files():
            if root.is_file():
                yield root
                return
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if not _skip_dir(d, include_ignored)]
                for name in filenames:
                    if Path(name).match(include):
                        yield Path(dirpath) / name

        for file in iter_files():
            try:
                if file.stat().st_size > MAX_GREP_FILE_BYTES:
                    continue
                data = file.read_bytes()
            except OSError:
                continue
            if b"\x00" in data[:8192]:
                continue
            files_scanned += 1
            for lineno, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
                if regex.search(line):
                    if len(matches) >= limit:
                        truncated = True
                        break
                    matches.append({"file": _rel(file, ctx), "line": lineno,
                                    "text": line.strip()[:300]})
            if truncated:
                break
        return ToolResult.success(
            {"matches": matches, "count": len(matches), "files_scanned": files_scanned},
            f"found {len(matches)} matches in {files_scanned} files", truncated=truncated,
        )
