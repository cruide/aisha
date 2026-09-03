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
            raise ToolPermissionError(f"Путь за пределами workspace: {resolved}")
    return resolved, inside


def _read_text(path: Path, limit_bytes: int) -> str:
    size = path.stat().st_size
    if size > limit_bytes * 4:
        raise ToolValidationError(
            f"Файл слишком большой ({human_size(size)}); используй offset/limit или grep"
        )
    data = path.read_bytes()
    if b"\x00" in data[:8192]:
        raise ToolValidationError("Файл выглядит бинарным")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


async def _confirm_outside_write(ctx: ToolContext, path: Path, action: str) -> None:
    await require_confirmation(ctx, ConfirmRequest(
        title="Запись за пределами workspace",
        details=[("Действие", action), ("Путь", str(path))],
        reason="файл находится вне рабочей директории",
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
        "Прочитать текстовый файл (UTF-8). offset — номер первой строки (с 0), "
        "limit — сколько строк вернуть (по умолчанию 500)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Путь к файлу (относительно workspace)"},
            "offset": {"type": "integer", "description": "Первая строка, с 0"},
            "limit": {"type": "integer", "description": "Максимум строк"},
        },
        "required": ["path"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path, _ = resolve_path(args["path"], ctx, write=False)
        if not path.is_file():
            return ToolResult.failure("FileNotFoundError", f"Файл не найден: {args['path']}")
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
        summary = f"{len(lines)} строк, {human_size(path.stat().st_size)}"
        if truncated:
            summary += f" (показано {len(chunk)} с {offset})"
        return ToolResult.success(data, summary, truncated=truncated)


class WriteFileTool(Tool):
    name = "write_file"
    description = "Создать файл или полностью перезаписать его содержимое (атомарно)."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "create_dirs": {"type": "boolean",
                            "description": "Создавать родительские каталоги (по умолчанию true)"},
        },
        "required": ["path", "content"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path, inside = resolve_path(args["path"], ctx, write=True)
        if not inside:
            await _confirm_outside_write(ctx, path, "write_file")
        if path.is_dir():
            return ToolResult.failure("IsADirectoryError", f"Это каталог: {args['path']}")
        if not path.parent.exists():
            if args.get("create_dirs", True):
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                return ToolResult.failure("FileNotFoundError",
                                          f"Каталог не существует: {path.parent}")
        existed = path.exists()
        content: str = args["content"]
        atomic_write_text(path, content)
        action = "перезаписан" if existed else "создан"
        data = {"path": _rel(path, ctx), "action": action, "bytes": len(content.encode("utf-8")),
                "lines": content.count("\n") + (1 if content and not content.endswith("\n") else 0)}
        return ToolResult.success(data, f"{action}, {data['lines']} строк")


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Точная замена фрагмента текста в файле. old_text должен встречаться ровно "
        "expected_replacements раз (по умолчанию 1), иначе файл не изменяется."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
            "expected_replacements": {"type": "integer"},
        },
        "required": ["path", "old_text", "new_text"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path, inside = resolve_path(args["path"], ctx, write=True)
        if not inside:
            await _confirm_outside_write(ctx, path, "edit_file")
        if not path.is_file():
            return ToolResult.failure("FileNotFoundError", f"Файл не найден: {args['path']}")
        old, new = args["old_text"], args["new_text"]
        if not old:
            return ToolResult.failure("ToolValidationError", "old_text не может быть пустым")
        expected = int(args.get("expected_replacements", 1))
        text = _read_text(path, ctx.config.tools.max_output_chars * 4)
        count = text.count(old)
        if count == 0:
            return ToolResult.failure("ToolValidationError",
                                      "old_text не найден в файле; файл не изменён")
        if count != expected:
            return ToolResult.failure(
                "ToolValidationError",
                f"Найдено {count} совпадений, ожидалось {expected}; файл не изменён. "
                "Уточни old_text или укажи expected_replacements.",
            )
        atomic_write_text(path, text.replace(old, new))
        return ToolResult.success({"path": _rel(path, ctx), "replacements": count},
                                  f"{count} замена(ы)")


class ListDirTool(Tool):
    name = "list_dir"
    read_only = True
    description = "Показать содержимое каталога (имена, тип, размер)."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Каталог, по умолчанию '.'"},
            "show_hidden": {"type": "boolean"},
            "limit": {"type": "integer"},
        },
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path, _ = resolve_path(args.get("path", "."), ctx, write=False)
        if not path.is_dir():
            return ToolResult.failure("NotADirectoryError", f"Каталог не найден: {path}")
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
            f"{dirs} каталогов, {len(entries) - dirs} файлов", truncated=truncated,
        )


class GlobTool(Tool):
    name = "glob"
    read_only = True
    description = "Найти файлы по маске, например '**/*.py' или 'src/**/*Controller.php'."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "База поиска, по умолчанию '.'"},
            "limit": {"type": "integer"},
            "include_ignored": {"type": "boolean",
                                "description": "Не исключать .git, node_modules, vendor и т.п."},
        },
        "required": ["pattern"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        base, _ = resolve_path(args.get("path", "."), ctx, write=False)
        if not base.is_dir():
            return ToolResult.failure("NotADirectoryError", f"Каталог не найден: {base}")
        limit = int(args.get("limit", 200))
        include_ignored = bool(args.get("include_ignored", False))
        found: list[str] = []
        truncated = False
        for path in base.glob(args["pattern"]):
            rel_parts = path.relative_to(base).parts
            if any(_skip_dir(p, include_ignored) for p in rel_parts[:-1]):
                continue
            if not path.is_file():
                continue
            if len(found) >= limit:
                truncated = True
                break
            found.append(_rel(path, ctx))
        found.sort()
        return ToolResult.success({"files": found, "count": len(found)},
                                  f"найдено {len(found)} файлов", truncated=truncated)


class GrepTool(Tool):
    name = "grep"
    read_only = True
    description = "Regex-поиск по содержимому файлов. include — маска имён файлов, например '*.php'."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Регулярное выражение (Python re)"},
            "path": {"type": "string", "description": "Файл или каталог, по умолчанию '.'"},
            "include": {"type": "string", "description": "Маска файлов, например '*.py'"},
            "ignore_case": {"type": "boolean"},
            "limit": {"type": "integer", "description": "Максимум совпадений (по умолчанию 100)"},
            "include_ignored": {"type": "boolean"},
        },
        "required": ["pattern"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            regex = re.compile(args["pattern"], re.IGNORECASE if args.get("ignore_case") else 0)
        except re.error as exc:
            return ToolResult.failure("ToolValidationError", f"Некорректное regex: {exc}")
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
            f"найдено {len(matches)} совпадений в {files_scanned} файлах", truncated=truncated,
        )
    