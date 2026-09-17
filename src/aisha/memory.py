# Author: Tischenko A. (https://github.com/cruide)
"""Persistent memory blocks stored as Markdown.

Layout:
- global:  ``~/.aisha/memory/MEMORY.md``
- project: ``<workspace>/.aisha/memory/MEMORY.md`` (overrides global by label)

Each block is a level-2 heading followed by optional HTML-comment metadata and
the value text (everything up to the next ``## `` heading)::

    # Memory

    ## user_profile
    <!-- description: User and technical skills -->
    <!-- updated_at: 2026-09-02T19:30:46Z -->
    Name: Sanya
    Role: task author

Legacy ``*.json`` block files are imported once on startup and then removed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aisha.errors import ToolValidationError
from aisha.fsutil import atomic_write_text

NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
SCOPES = ("global", "project")
MEMORY_FILENAME = "MEMORY.md"
FILE_HEADER = "# Memory"
SECTION_RE = re.compile(r"^##\s+(.+?)\s*$")
META_RE = re.compile(r"^<!--\s*(description|updated_at):\s*(.*?)\s*-->\s*$")


@dataclass(slots=True)
class MemoryBlock:
    label: str
    description: str
    value: str
    scope: str
    updated_at: str


def parse_memory_text(text: str, scope: str) -> list[MemoryBlock]:
    """Parse a MEMORY.md body into blocks (order preserved)."""
    blocks: list[MemoryBlock] = []
    label: str | None = None
    description = ""
    updated_at = ""
    value_lines: list[str] = []
    meta_open = False

    def flush() -> None:
        if label is None:
            return
        blocks.append(MemoryBlock(
            label=label,
            description=description,
            value="\n".join(value_lines).strip("\n"),
            scope=scope,
            updated_at=updated_at,
        ))

    for line in text.splitlines():
        section = SECTION_RE.match(line)
        if section:
            flush()
            label = section.group(1).strip()
            description = ""
            updated_at = ""
            value_lines = []
            meta_open = True
            continue
        if label is None:
            continue  # preamble before the first block
        if line.startswith("\\##"):
            value_lines.append(line[1:])  # unescape a heading-like value line
            meta_open = False
            continue
        if meta_open:
            meta = META_RE.match(line)
            if meta:
                if meta.group(1) == "description":
                    description = meta.group(2)
                else:
                    updated_at = meta.group(2)
                continue
            if not line.strip():
                continue  # blank separator between metadata and value
            meta_open = False
        value_lines.append(line)
    flush()
    return blocks


def render_memory_text(blocks: list[MemoryBlock]) -> str:
    """Serialize blocks back to a MEMORY.md document."""
    lines = [FILE_HEADER, ""]
    if not blocks:
        lines.append("<!-- No memory blocks yet. -->")
        return "\n".join(lines) + "\n"
    for block in blocks:
        lines.append(f"## {block.label}")
        if block.description:
            lines.append(f"<!-- description: {block.description} -->")
        if block.updated_at:
            lines.append(f"<!-- updated_at: {block.updated_at} -->")
        if block.value:
            # Escape value lines that would otherwise look like a new block heading.
            lines.extend("\\" + ln if ln.startswith("##") else ln
                         for ln in block.value.split("\n"))
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


class MemoryStore:
    def __init__(
        self, global_dir: Path, project_dir: Path, *,
        max_block_chars: int, index_max_chars: int = 20_000,
    ) -> None:
        self.dirs = {"global": global_dir, "project": project_dir}
        self.max_block_chars = max_block_chars
        self.index_max_chars = index_max_chars
        self.errors: list[str] = []
        self._migration_errors: list[str] = []
        self._cache: tuple[tuple[Any, ...], list[MemoryBlock]] | None = None
        self._migrate_legacy_json()

    @staticmethod
    def validate_label(label: str) -> str:
        if not NAME_RE.match(label or ""):
            raise ToolValidationError(
                f"Invalid block name {label!r}: must match [a-zA-Z0-9][a-zA-Z0-9_-]{{0,63}}"
            )
        return label

    def _path(self, scope: str) -> Path:
        return self.dirs[scope] / MEMORY_FILENAME

    # ------------------------------------------------------------ file access
    def _read_scope(self, scope: str, errors: list[str] | None = None) -> list[MemoryBlock]:
        """Read the MEMORY.md of one scope straight from disk (no cache)."""
        path = self._path(scope)
        if not path.is_file():
            return []
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            if errors is not None:
                errors.append(f"{path}: {exc}")
            return []
        blocks: list[MemoryBlock] = []
        for block in parse_memory_text(text, scope):
            if NAME_RE.match(block.label):
                blocks.append(block)
            elif errors is not None:
                errors.append(f"{path}: invalid block name {block.label!r}, skipped")
        return blocks

    def _write_scope(self, scope: str, blocks: list[MemoryBlock]) -> None:
        directory = self.dirs[scope]
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self._path(scope), render_memory_text(blocks))

    def _dir_signature(self) -> tuple[Any, ...]:
        """Fingerprint of the memory files; changes when a block is added/removed/edited."""
        sig: list[Any] = []
        for scope in SCOPES:
            path = self._path(scope)
            try:
                sig.append((path.stat().st_size, path.stat().st_mtime_ns) if path.is_file()
                           else None)
            except OSError:
                sig.append(None)
        return tuple(sig)

    # --------------------------------------------------------------- legacy
    def _migrate_legacy_json(self) -> None:
        """Import old ``*.json`` blocks into MEMORY.md once, then delete them."""
        for scope in SCOPES:
            directory = self.dirs[scope]
            if not directory.is_dir():
                continue
            legacy = sorted(directory.glob("*.json"))
            if not legacy:
                continue
            merged = {b.label: b for b in self._read_scope(scope)}
            imported: list[Path] = []
            for path in legacy:
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    block = MemoryBlock(
                        label=str(raw["label"]),
                        description=str(raw.get("description", "")),
                        value=str(raw.get("value", "")),
                        scope=scope,
                        updated_at=str(raw.get("updated_at", "")),
                    )
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    self._migration_errors.append(f"{path}: failed to import ({exc})")
                    continue
                if not NAME_RE.match(block.label):
                    self._migration_errors.append(
                        f"{path}: invalid block name {block.label!r}, skipped"
                    )
                    continue
                merged.setdefault(block.label, block)
                imported.append(path)
            if imported:
                try:
                    self._write_scope(scope, sorted(merged.values(), key=lambda b: b.label))
                except OSError as exc:
                    self._migration_errors.append(f"{self._path(scope)}: {exc}")
                    continue
                for path in imported:
                    try:
                        path.unlink()
                    except OSError as exc:
                        self._migration_errors.append(f"{path}: {exc}")

    # ------------------------------------------------------------------ reads
    def _read_all(self) -> list[MemoryBlock]:
        blocks: dict[str, MemoryBlock] = {}
        self.errors = list(self._migration_errors)
        for scope in SCOPES:  # project overrides global
            for block in self._read_scope(scope, self.errors):
                blocks[block.label] = block
        return sorted(blocks.values(), key=lambda b: b.label)

    def list(self) -> list[MemoryBlock]:
        signature = self._dir_signature()
        if self._cache is None or self._cache[0] != signature:
            self._cache = (signature, self._read_all())
        return list(self._cache[1])

    def get(self, label: str) -> MemoryBlock | None:
        self.validate_label(label)
        for block in self.list():
            if block.label == label:
                return block
        return None

    # ----------------------------------------------------------------- writes
    def set(self, label: str, description: str, value: str, scope: str = "global") -> MemoryBlock:
        self.validate_label(label)
        if scope not in SCOPES:
            raise ToolValidationError(f"scope must be one of: {', '.join(SCOPES)}")
        if len(value) > self.max_block_chars:
            raise ToolValidationError(
                f"Block is too large ({len(value)} chars, limit {self.max_block_chars}). "
                "Compress the content first."
            )
        block = MemoryBlock(
            label=label,
            description=description.strip(),
            value=value,
            scope=scope,
            updated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        blocks = {b.label: b for b in self._read_scope(scope)}
        blocks[label] = block
        self._write_scope(scope, sorted(blocks.values(), key=lambda b: b.label))
        self._cache = None
        return block

    def replace(self, label: str, old: str, new: str, expected: int = 1) -> MemoryBlock:
        block = self.get(label)
        if block is None:
            raise ToolValidationError(f"Memory block not found: {label}")
        count = block.value.count(old)
        if count == 0:
            raise ToolValidationError("Replacement text not found in the block")
        if count != expected:
            raise ToolValidationError(f"Found {count} matches, expected {expected}")
        return self.set(label, block.description, block.value.replace(old, new), block.scope)

    # ----------------------------------------------------------------- index
    def index_text(self) -> str:
        blocks = self.list()
        if not blocks:
            return ""

        lines = [
            f"- {b.label} ({b.scope}) — {b.description or 'no description'}"
            for b in blocks
        ]

        text = "\n".join(lines)

        if len(text) <= self.index_max_chars:
            return text

        # Truncate and add a hint.
        remaining = len(lines)
        result: list[str] = []
        total = 0
        for line in lines:
            if total + len(line) + 1 > self.index_max_chars:
                break
            result.append(line)
            total += len(line) + 1
            remaining -= 1
        result.append(f"… +{remaining} more, use memory_list")
        return "\n".join(result)
