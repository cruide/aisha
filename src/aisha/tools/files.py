# Author: Tischenko A. (https://github.com/cruide)
"""File tools: read_file, write_file, edit_file, list_dir, glob, grep."""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
from pathlib import Path
from typing import Any

from rich.markup import escape

from aisha.errors import ToolPermissionError, ToolValidationError
from aisha.fsutil import atomic_write_text, human_size, is_inside
from aisha.tools.base import ConfirmRequest, Tool, ToolContext, ToolResult, require_confirmation

DEFAULT_EXCLUDES = frozenset({
    ".git", ".hg", ".svn", ".idea", ".vscode", "node_modules", "vendor", ".opencode",
    "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".aisha",
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


def _read_full_text(path: Path, limit_bytes: int) -> str:
    """Read the whole file; raise a clear error when it exceeds `limit_bytes`."""
    size = path.stat().st_size
    if size > limit_bytes:
        raise ToolValidationError(
            f"File is too large for edit_file ({human_size(size)}, limit "
            f"{human_size(limit_bytes)}); split it into smaller files first"
        )
    data = path.read_bytes()
    if b"\x00" in data[:8192]:
        raise ToolValidationError("File appears to be binary")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def _strip_blank_edges(text: str) -> str:
    """Drop leading/trailing whitespace-only lines (tolerates stray blank lines)."""
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _old_candidates(old: str) -> list[str]:
    """old_text variants to try, most exact first (verbatim, then blank-edge trimmed)."""
    candidates = [old]
    stripped = _strip_blank_edges(old)
    if stripped and stripped != old:
        candidates.append(stripped)
    return candidates


def _ws_tolerant_pattern(old: str) -> re.Pattern[str]:
    """Regex matching `old` while allowing trailing spaces/tabs on each line."""
    parts = [re.escape(line.rstrip(" \t")) + r"[ \t]*" for line in old.split("\n")]
    return re.compile("\n".join(parts))


def _apply_candidate(
    norm_text: str, candidate: str, norm_new: str, expected: int
) -> tuple[str | None, int]:
    """Try exact then whitespace-tolerant replacement of `candidate`.

    Returns ``(new_text, count)`` when exactly `expected` occurrences were found,
    otherwise ``(None, best_found)`` so the caller can report a useful count.
    """
    exact = norm_text.count(candidate)
    if exact == expected:
        return norm_text.replace(candidate, norm_new), expected
    matches = list(_ws_tolerant_pattern(candidate).finditer(norm_text))
    if len(matches) == expected:
        result = norm_text
        for match in reversed(matches):
            result = result[:match.start()] + norm_new + result[match.end():]
        return result, expected
    return None, max(exact, len(matches))


def _replace_fragment(
    norm_text: str, norm_old: str, norm_new: str, expected: int
) -> tuple[str | None, int]:
    """Replace `norm_old` tolerating line endings, trailing spaces and blank edges."""
    best_found = 0
    for candidate in _old_candidates(norm_old):
        result, found = _apply_candidate(norm_text, candidate, norm_new, expected)
        if result is not None:
            return result, found
        best_found = max(best_found, found)
    return None, best_found


def _find_closest_region(norm_text: str, norm_old: str, context: int = 2) -> tuple[int, str] | None:
    """Locate the region most similar to `norm_old` ignoring leading whitespace.

    Returns ``(1-based line number, snippet)`` for the first file line whose
    stripped text equals the first non-blank old_text line, or ``None``.
    """
    old_lines = norm_old.split("\n")
    first = next((line.strip() for line in old_lines if line.strip()), "")
    if not first:
        return None
    probe_len = max(1, len([line for line in old_lines if line.strip()]))
    file_lines = norm_text.split("\n")
    for idx, line in enumerate(file_lines):
        if line.strip() != first:
            continue
        low = max(0, idx - context)
        high = min(len(file_lines), idx + probe_len + context)
        return idx + 1, "\n".join(file_lines[low:high])
    return None


def _not_found_message(norm_text: str, norm_old: str) -> str:
    """Actionable 'old_text not found' error with a closest-match hint when possible."""
    msg = "old_text not found in file; file unchanged"
    region = _find_closest_region(norm_text, norm_old)
    if region is None:
        return msg + ". Re-read the file with read_file and copy the exact fragment."
    line_no, snippet = region
    if len(snippet) > 1500:
        snippet = snippet[:1500] + "…"
    return (
        f"{msg}. Closest match is near line {line_no} (check indentation and exact "
        f"characters):\n```\n{snippet}\n```\n"
        "Re-read the file and copy the exact fragment as old_text."
    )


def _read_window(
    path: Path, offset: int, limit: int, max_chars: int
) -> tuple[str, int, bool, int | None]:
    """Read a window of `path` starting at `offset` lines.

    Returns (content, returned_lines, truncated, lines_total). `lines_total` is None
    when the file was not fully read to EOF, so the exact line count is unknown.
    A line is never split: if it would exceed the char budget it is dropped and
    the result is marked truncated.
    """
    with path.open("rb") as fh:
        if b"\x00" in fh.read(8192):
            raise ToolValidationError("File appears to be binary")
    parts: list[str] = []
    returned = 0
    skipped = 0
    chars = 0
    truncated = False
    reached_eof = False
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for _ in range(offset):
            line = fh.readline()
            if line == "":
                reached_eof = True
                break
            skipped += 1
        if not reached_eof:
            for line in fh:
                if returned >= limit:
                    if fh.readline() == "":
                        reached_eof = True
                    else:
                        truncated = True
                    break
                line_chars = len(line)
                if returned > 0 and chars + line_chars > max_chars:
                    truncated = True
                    break
                parts.append(line)
                chars += line_chars
                returned += 1
            else:
                reached_eof = True
    lines_total = skipped + returned if reached_eof else None
    return "".join(parts), returned, truncated, lines_total


def _read_full(path: Path, max_chars: int) -> tuple[str, int, bool, int]:
    """Read the entire file at once.

    Returns (content, line_count, truncated, lines_total).
    truncated is True when the file exceeds max_chars.
    """
    with path.open("rb") as fh:
        if b"\x00" in fh.read(8192):
            raise ToolValidationError("File appears to be binary")
    data = path.read_text(encoding="utf-8", errors="replace")
    lines_total = data.count("\n") + (1 if data and not data.endswith("\n") else 0)
    if len(data) > max_chars:
        truncated = True
        content = data[:max_chars]
        returned = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    else:
        truncated = False
        content = data
        returned = lines_total
    return content, returned, truncated, lines_total


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


def _rel_to(path: Path, workspace: Path) -> str:
    try:
        return path.relative_to(workspace).as_posix() or "."
    except ValueError:
        return str(path)


def _glob_match(rel: str, pattern: str) -> bool:
    """Glob-match a relative POSIX path against a pattern (`*`, `**`, `?`, `[...]`)."""
    return _seg_match(rel.split("/"), pattern.split("/"))


def _seg_match(parts: list[str], pats: list[str]) -> bool:
    if not pats:
        return not parts
    head, tail = pats[0], pats[1:]
    if head == "**":
        if _seg_match(parts, tail):
            return True
        return bool(parts) and _seg_match(parts[1:], pats)
    if not parts:
        return False
    if fnmatch.fnmatchcase(parts[0], head):
        return _seg_match(parts[1:], tail)
    return False


def _glob_walk(base: Path, pattern: str, workspace: Path, include_ignored: bool,
               allow_outside: bool, limit: int) -> tuple[list[str], bool]:
    """Walk `base` pruning ignored dirs, returning matching files and a truncation flag."""
    found: list[str] = []
    truncated = False
    pat = pattern[2:] if pattern.startswith("./") else pattern
    pat = pat.rstrip("/")
    for dirpath, dirnames, filenames in os.walk(base):
        if not include_ignored:
            dirnames[:] = [d for d in dirnames if not _skip_dir(d, include_ignored)]
        for name in filenames:
            full = Path(dirpath) / name
            try:
                rel = full.relative_to(base).as_posix()
            except ValueError:
                continue
            if not _glob_match(rel, pat):
                continue
            try:
                resolved = full.resolve()
            except OSError:
                continue
            if not is_inside(resolved, workspace) and not allow_outside:
                continue
            if not resolved.is_file():
                continue
            if len(found) >= limit:
                truncated = True
                return found, truncated
            found.append(_rel_to(resolved, workspace))
    return found, truncated


def _grep_walk(root: Path, regex: re.Pattern, include: str, limit: int,
               include_ignored: bool, workspace: Path, allow_outside: bool
               ) -> tuple[list[dict[str, Any]], int, bool]:
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
            resolved = file.resolve()
        except OSError:
            continue
        if not is_inside(resolved, workspace) and not allow_outside:
            continue
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
                matches.append({"file": _rel_to(resolved, workspace), "line": lineno,
                                "text": line.strip()[:300]})
        if truncated:
            break
    return matches, files_scanned, truncated


class ReadFileTool(Tool):
    name = "read_file"
    read_only = True
    description = (
        "Read a UTF-8 text file. Use offset (0-based line) and limit for a window; "
        "the result includes the exact content, so copy edit_file.old_text from it. "
        "If truncated or incomplete, read the needed window again."
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

    FULL_READ_THRESHOLD = 98304

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path, _ = resolve_path(args["path"], ctx, write=False)
        if not path.is_file():
            return ToolResult.failure("FileNotFoundError", f"File not found: {args['path']}")
        max_chars = ctx.config.tools.max_output_chars
        offset = max(0, int(args.get("offset", 0)))
        limit = max(1, int(args.get("limit", 300)))
        full_read = (
            ctx.config.llm.context_window >= self.FULL_READ_THRESHOLD
            and offset == 0
            and "limit" not in args
        )
        try:
            if full_read:
                content, returned, truncated, lines_total = await asyncio.to_thread(
                    _read_full, path, max_chars
                )
            else:
                content, returned, truncated, lines_total = await asyncio.to_thread(
                    _read_window, path, offset, limit, max_chars
                )
        except ToolValidationError as exc:
            return ToolResult.failure("ToolValidationError", str(exc))
        size_bytes = path.stat().st_size
        fname = path.name
        data: dict[str, Any] = {
            "path": _rel(path, ctx), "content": content, "offset": offset,
            "returned": returned, "size_bytes": size_bytes,
            "lines_total": lines_total, "total_known": lines_total is not None,
        }
        if full_read and not truncated:
            summary = f"[bright_cyan]{escape(fname)}[/], {human_size(size_bytes)}"
        else:
            summary = (
                f"[bright_cyan]{escape(fname)}[/], {human_size(size_bytes)}"
                f" (showing {returned} from {offset})"
            )
        return ToolResult.success(data, summary, truncated=truncated)


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Create or fully overwrite a UTF-8 text file atomically; this replaces all existing content. "
        "Use edit_file for a small change. Parent directories are created by default."
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
        fname = path.name
        data = {"path": _rel(path, ctx), "action": action, "bytes": len(content.encode("utf-8")),
                "lines": content.count("\n") + (1 if content and not content.endswith("\n") else 0)}
        return ToolResult.success(
            data, f"[bright_cyan]{escape(fname)}[/], {action}, {data['lines']} lines"
        )


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Replace an exact fragment in an existing text file. First read the target region and copy "
        "old_text verbatim from the latest read_file.content: preserve spaces, indentation and newlines; "
        "do not add line numbers or code fences. Use a unique fragment. If not found, read again before retrying."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path (relative to workspace)"},
            "old_text": {"type": "string",
                         "description": "Verbatim fragment copied from read_file "
                                        "(exact whitespace/indentation, no line numbers)"},
            "new_text": {"type": "string",
                         "description": "Replacement text; keep the original indentation"},
            "expected_replacements": {"type": "integer",
                                      "description": "Exact number of occurrences to "
                                                     "replace (default 1)"},
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
        if not old.strip():
            return ToolResult.failure("ToolValidationError", "old_text cannot be empty")
        expected = int(args.get("expected_replacements", 1))
        max_chars = ctx.config.tools.max_output_chars
        try:
            text = await asyncio.to_thread(_read_full_text, path, max_chars * 16)
        except ToolValidationError as exc:
            return ToolResult.failure("ToolValidationError", str(exc))

        crlf = text.count("\r\n") > text.count("\n") - text.count("\r\n")
        norm_text = text.replace("\r\n", "\n")
        norm_old = old.replace("\r\n", "\n")
        norm_new = new.replace("\r\n", "\n")

        result, count = _replace_fragment(norm_text, norm_old, norm_new, expected)
        if result is None:
            if count == 0:
                return ToolResult.failure(
                    "ToolValidationError", _not_found_message(norm_text, norm_old)
                )
            return ToolResult.failure(
                "ToolValidationError",
                f"Found {count} matches, expected {expected}; file unchanged. "
                "Include more surrounding lines to make old_text unique, or set "
                "expected_replacements.",
            )
        if crlf:
            result = result.replace("\n", "\r\n")
        atomic_write_text(path, result)
        fname = path.name
        return ToolResult.success({"path": _rel(path, ctx), "replacements": count},
                                  f"[bright_cyan]{escape(fname)}[/], {count} replacement(s)")


class ListDirTool(Tool):
    name = "list_dir"
    read_only = True
    description = (
        "List files and directories in a path. Use show_hidden for dotfiles and limit to cap results."
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
        rel = _rel(path, ctx)
        return ToolResult.success(
            {"path": rel, "entries": entries},
            f"'[bright_cyan]{escape(rel)}[/]' {dirs} dirs, {len(entries) - dirs} files",
            truncated=truncated,
        )


class GlobTool(Tool):
    name = "glob"
    read_only = True
    description = (
        "Find file paths matching a glob pattern. Use path as the search base; patterns support *, **, ?, "
        "and [...]. Results exclude common generated/hidden directories unless include_ignored is true."
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
        allow_outside = ctx.config.tools.allow_read_outside_workspace
        found, truncated = await asyncio.to_thread(
            _glob_walk, base, args["pattern"], ctx.workspace, include_ignored,
            allow_outside, limit,
        )
        found.sort()
        base_rel = _rel(base, ctx)
        pat = args['pattern']
        summary = (
            f"[bright_cyan]{escape(base_rel)}[/] "
            f"[[bright_cyan]\"{escape(pat)}\"[/]] — found {len(found)} files"
        )
        return ToolResult.success(
            {"files": found, "count": len(found)}, summary, truncated=truncated,
        )


class GrepTool(Tool):
    name = "grep"
    read_only = True
    description = (
        "Search file contents with a Python regular expression. Use include to filter filenames and path "
        "to select a file or directory; set ignore_case when needed. Results contain file, line and text."
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
        allow_outside = ctx.config.tools.allow_read_outside_workspace
        matches, files_scanned, truncated = await asyncio.to_thread(
            _grep_walk, root, regex, include, limit, include_ignored, ctx.workspace,
            allow_outside,
        )
        root_rel = _rel(root, ctx)
        summary = (
            f"[bright_cyan]{escape(root_rel)}[/] [[bright_cyan]\"{escape(args['pattern'])}\"[/]] — "
            f"found {len(matches)} matches in {files_scanned} files"
        )
        return ToolResult.success(
            {"matches": matches, "count": len(matches), "files_scanned": files_scanned},
            summary, truncated=truncated,
        )
