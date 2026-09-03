"""Persistent memory blocks: global (~/.aisha/memory) and project (<ws>/.aisha/memory)."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from aisha.errors import ToolValidationError
from aisha.fsutil import atomic_write_text

NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
SCOPES = ("global", "project")


@dataclass(slots=True)
class MemoryBlock:
    label: str
    description: str
    value: str
    scope: str
    updated_at: str


class MemoryStore:
    def __init__(self, global_dir: Path, project_dir: Path, *, max_block_chars: int) -> None:
        self.dirs = {"global": global_dir, "project": project_dir}
        self.max_block_chars = max_block_chars

    @staticmethod
    def validate_label(label: str) -> str:
        if not NAME_RE.match(label or ""):
            raise ToolValidationError(
                f"Недопустимое имя блока {label!r}: [a-zA-Z0-9][a-zA-Z0-9_-]{{0,63}}"
            )
        return label

    def _read(self, path: Path, scope: str) -> MemoryBlock | None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return MemoryBlock(
                label=str(raw["label"]),
                description=str(raw.get("description", "")),
                value=str(raw.get("value", "")),
                scope=scope,
                updated_at=str(raw.get("updated_at", "")),
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def list(self) -> list[MemoryBlock]:
        blocks: dict[str, MemoryBlock] = {}
        for scope in SCOPES:  # project overrides global
            directory = self.dirs[scope]
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.json")):
                block = self._read(path, scope)
                if block:
                    blocks[block.label] = block
        return sorted(blocks.values(), key=lambda b: b.label)

    def get(self, label: str) -> MemoryBlock | None:
        self.validate_label(label)
        for scope in reversed(SCOPES):
            path = self.dirs[scope] / f"{label}.json"
            if path.is_file():
                return self._read(path, scope)
        return None

    def set(self, label: str, description: str, value: str, scope: str = "global") -> MemoryBlock:
        self.validate_label(label)
        if scope not in SCOPES:
            raise ToolValidationError(f"scope должен быть одним из: {', '.join(SCOPES)}")
        if len(value) > self.max_block_chars:
            raise ToolValidationError(
                f"Блок слишком большой ({len(value)} символов, лимит {self.max_block_chars}). "
                "Сначала сожми содержимое."
            )
        block = MemoryBlock(
            label=label,
            description=description.strip(),
            value=value,
            scope=scope,
            updated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        directory = self.dirs[scope]
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            directory / f"{label}.json",
            json.dumps(asdict(block), ensure_ascii=False, indent=2),
        )
        return block

    def replace(self, label: str, old: str, new: str, expected: int = 1) -> MemoryBlock:
        block = self.get(label)
        if block is None:
            raise ToolValidationError(f"Блок памяти не найден: {label}")
        count = block.value.count(old)
        if count == 0:
            raise ToolValidationError("Текст для замены не найден в блоке")
        if count != expected:
            raise ToolValidationError(f"Найдено {count} совпадений, ожидалось {expected}")
        return self.set(label, block.description, block.value.replace(old, new), block.scope)

    def index_text(self) -> str:
        blocks = self.list()
        if not blocks:
            return ""
        return "\n".join(f"- {b.label} ({b.scope}) — {b.description or 'без описания'}" for b in blocks)
    