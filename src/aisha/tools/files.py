"""File-related tools: read, write, edit, list, glob, grep."""

from __future__ import annotations

__author__ = "Tischenko Alexander"

import fnmatch
import os
import re
import tempfile
from pathlib import Path

from aisha.tools.base import Tool, ToolResult, ToolValidationError

DEFAULT_EXCLUDES = {".git", ".idea", "node_modules", "vendor", "__pycache__"}
MAX_READ_CHARS = 1_000_000
MAX_LIST_ITEMS = 500
MAX_GLOB_RESULTS = 200
MAX_GREP_RESULTS = 200
MAX_GREP_FILE_SIZE = 1_000_000


def _resolve_path(path: str, workspace: Path) -> Path:
    """Resolve path relative to workspace."""
    p = Path(path)
    if not p.is_absolute():
        p = workspace / p
    return p.resolve()


def _is_inside(path: Path, base: Path) -> bool:
    """Check if path is inside base directory."""
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    half = max_chars // 2
    return text[:half] + "\n...\n" + text[-half:], True


class ReadFileTool(Tool):
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read file contents with optional offset and limit."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to workspace"},
                "offset": {"type": "integer", "description": "Line offset (0-based)", "default": 0},
                "limit": {"type": "integer", "description": "Max lines to read", "default": 500},
            },
            "required": ["path"],
        }

    def validate_args(self, args: dict) -> dict:
        if "offset" in args and args["offset"] < 0:
            raise ToolValidationError("offset must be >= 0")
        if "limit" in args and args["limit"] <= 0:
            raise ToolValidationError("limit must be > 0")
        return args

    async def execute(self, args: dict, context: dict) -> ToolResult:
        workspace: Path = context["workspace"]
        path = _resolve_path(args["path"], workspace)
        offset = args.get("offset", 0)
        limit = args.get("limit", 500)

        if not path.is_file():
            return ToolResult.failure("FileNotFoundError", f"Файл не найден: {args['path']}")

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult.failure(
                "UnicodeDecodeError",
                f"Не удалось декодировать файл как UTF-8: {args['path']}",
            )
        except Exception as e:
            return ToolResult.from_exception(e)

        lines = content.splitlines()
        total_lines = len(lines)
        selected = lines[offset : offset + limit]
        text = "\n".join(selected)
        truncated = len(text) > MAX_READ_CHARS
        text, _ = _truncate(text, MAX_READ_CHARS)

        return ToolResult.success(
            {
                "path": str(path.relative_to(workspace)),
                "content": text,
                "total_lines": total_lines,
                "offset": offset,
                "lines_read": len(selected),
                "truncated": truncated,
            }
        )


class WriteFileTool(Tool):
    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Create or overwrite a file. Supports atomic writes."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to workspace"},
                "content": {"type": "string", "description": "File content"},
                "create_parents": {
                    "type": "boolean",
                    "description": "Create parent directories if needed",
                    "default": True,
                },
            },
            "required": ["path", "content"],
        }

    async def execute(self, args: dict, context: dict) -> ToolResult:
        workspace: Path = context["workspace"]
        read_only: bool = context.get("read_only", False)
        allow_write_outside: bool = context.get("allow_write_outside_workspace", False)

        if read_only:
            return ToolResult.failure(
                "ToolPermissionError",
                "Режим read-only: запись файлов запрещена",
            )

        path = _resolve_path(args["path"], workspace)
        content = args["content"]
        create_parents = args.get("create_parents", True)

        if not _is_inside(path, workspace) and not allow_write_outside:
            return ToolResult.failure(
                "ToolPermissionError",
                f"Запись за пределами workspace запрещена: {path}",
            )

        created = not path.exists()

        try:
            if create_parents:
                path.parent.mkdir(parents=True, exist_ok=True)

            # Atomic write via temp file
            fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                os.replace(tmp_path, path)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

        except Exception as e:
            return ToolResult.from_exception(e)

        return ToolResult.success(
            {
                "path": str(path.relative_to(workspace)),
                "created": created,
                "bytes_written": len(content.encode("utf-8")),
            }
        )


class EditFileTool(Tool):
    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return "Perform exact text replacement in a file."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to workspace"},
                "old_text": {"type": "string", "description": "Exact text to find and replace"},
                "new_text": {"type": "string", "description": "Replacement text"},
                "expected_replacements": {
                    "type": "integer",
                    "description": "Expected number of replacements",
                    "default": 1,
                },
            },
            "required": ["path", "old_text", "new_text"],
        }

    def validate_args(self, args: dict) -> dict:
        if not args.get("old_text"):
            raise ToolValidationError("old_text must not be empty")
        return args

    async def execute(self, args: dict, context: dict) -> ToolResult:
        workspace: Path = context["workspace"]
        read_only: bool = context.get("read_only", False)
        allow_write_outside: bool = context.get("allow_write_outside_workspace", False)

        if read_only:
            return ToolResult.failure(
                "ToolPermissionError",
                "Режим read-only: редактирование файлов запрещено",
            )

        path = _resolve_path(args["path"], workspace)
        old_text = args["old_text"]
        new_text = args["new_text"]
        expected = args.get("expected_replacements", 1)

        if not _is_inside(path, workspace) and not allow_write_outside:
            return ToolResult.failure(
                "ToolPermissionError",
                f"Редактирование за пределами workspace запрещено: {path}",
            )

        if not path.is_file():
            return ToolResult.failure("FileNotFoundError", f"Файл не найден: {args['path']}")

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult.failure(
                "UnicodeDecodeError",
                f"Не удалось декодировать файл как UTF-8: {args['path']}",
            )
        except Exception as e:
            return ToolResult.from_exception(e)

        count = content.count(old_text)
        if count == 0:
            return ToolResult.failure(
                "TextNotFoundError",
                "old_text не найден в файле",
            )
        if count != expected:
            return ToolResult.failure(
                "UnexpectedMatchCount",
                f"Найдено {count} совпадений, ожидалось {expected}",
            )

        new_content = content.replace(old_text, new_text, 1)

        try:
            fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(new_content)
                os.replace(tmp_path, path)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            return ToolResult.from_exception(e)

        return ToolResult.success(
            {
                "path": str(path.relative_to(workspace)),
                "replacements": 1,
            }
        )


class ListDirTool(Tool):
    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List directory contents."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path (default: workspace root)",
                    "default": ".",
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files",
                    "default": False,
                },
            },
        }

    async def execute(self, args: dict, context: dict) -> ToolResult:
        workspace: Path = context["workspace"]
        path = _resolve_path(args.get("path", "."), workspace)
        show_hidden = args.get("show_hidden", False)

        if not path.is_dir():
            dir_path = args.get("path", ".")
            return ToolResult.failure(
                "NotADirectoryError", f"Не является директорией: {dir_path}"
            )

        try:
            entries = []
            for item in sorted(path.iterdir()):
                if not show_hidden and item.name.startswith("."):
                    continue
                entries.append(
                    {
                        "name": item.name,
                        "type": "dir" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else None,
                    }
                )
                if len(entries) >= MAX_LIST_ITEMS:
                    break

            truncated = len(list(path.iterdir())) > MAX_LIST_ITEMS
            return ToolResult.success(
                {
                    "path": (
                        str(path.relative_to(workspace))
                        if _is_inside(path, workspace)
                        else str(path)
                    ),
                    "entries": entries,
                    "truncated": truncated,
                }
            )
        except Exception as e:
            return ToolResult.from_exception(e)


class GlobTool(Tool):
    @property
    def name(self) -> str:
        return "glob"

    @property
    def description(self) -> str:
        return "Search files by pattern (glob)."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern (e.g. src/**/*.py)"},
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: workspace)",
                    "default": ".",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return",
                    "default": 100,
                },
                "exclude_dirs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Directory names to exclude",
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, args: dict, context: dict) -> ToolResult:
        workspace: Path = context["workspace"]
        base = _resolve_path(args.get("path", "."), workspace)
        pattern = args["pattern"]
        max_results = min(args.get("max_results", 100), MAX_GLOB_RESULTS)
        exclude_dirs = set(args.get("exclude_dirs", DEFAULT_EXCLUDES))

        if not base.is_dir():
            dir_path = args.get("path", ".")
            return ToolResult.failure(
                "NotADirectoryError", f"Директория не найдена: {dir_path}"
            )

        try:
            matches = []
            for root, dirs, files in os.walk(base):
                # Filter excluded dirs
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                root_path = Path(root)
                for fname in files:
                    fpath = root_path / fname
                    rel = fpath.relative_to(base)
                    if fnmatch.fnmatch(str(rel).replace("\\", "/"), pattern):
                        matches.append(str(rel))
                        if len(matches) >= max_results:
                            break
                if len(matches) >= max_results:
                    break

            return ToolResult.success(
                {
                    "pattern": pattern,
                    "matches": matches,
                    "truncated": len(matches) >= max_results,
                }
            )
        except Exception as e:
            return ToolResult.from_exception(e)


class GrepTool(Tool):
    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return "Search file contents using regex."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern"},
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in",
                    "default": ".",
                },
                "include": {
                    "type": "string",
                    "description": "File pattern to include (e.g. *.py)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max matches to return",
                    "default": 100,
                },
                "exclude_dirs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Directory names to exclude",
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, args: dict, context: dict) -> ToolResult:
        workspace: Path = context["workspace"]
        base = _resolve_path(args.get("path", "."), workspace)
        pattern = args["pattern"]
        include = args.get("include")
        max_results = min(args.get("max_results", 100), MAX_GREP_RESULTS)
        exclude_dirs = set(args.get("exclude_dirs", DEFAULT_EXCLUDES))

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return ToolResult.failure("RegexError", f"Некорректный regex: {e}")

        if base.is_file():
            files = [base]
        elif base.is_dir():
            files = []
            for root, dirs, filenames in os.walk(base):
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                for fname in filenames:
                    if include and not fnmatch.fnmatch(fname, include):
                        continue
                    fpath = Path(root) / fname
                    if fpath.stat().st_size > MAX_GREP_FILE_SIZE:
                        continue
                    files.append(fpath)
        else:
            search_path = args.get("path", ".")
            return ToolResult.failure(
                "PathNotFoundError", f"Путь не найден: {search_path}"
            )

        matches = []
        for fpath in files:
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(content.splitlines(), 1):
                    if regex.search(line):
                        file_str = (
                            str(fpath.relative_to(workspace))
                            if _is_inside(fpath, workspace)
                            else str(fpath)
                        )
                        matches.append(
                            {
                                "file": file_str,
                                "line": i,
                                "content": line.strip(),
                            }
                        )
                        if len(matches) >= max_results:
                            break
            except Exception:
                continue
            if len(matches) >= max_results:
                break

        return ToolResult.success(
            {
                "pattern": pattern,
                "matches": matches,
                "count": len(matches),
                "truncated": len(matches) >= max_results,
            }
        )
