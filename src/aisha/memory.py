"""Memory management: global and project-scoped persistent blocks."""

from __future__ import annotations

__author__ = "Tischenko Alexander"

import json
import os
import re
import tempfile
import time
from pathlib import Path

BLOCK_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


class MemoryManager:
    """Manages persistent memory blocks stored as JSON files."""

    def __init__(self, global_dir: Path, project_dir: Path) -> None:
        self._global_dir = global_dir
        self._project_dir = project_dir
        self._global_dir.mkdir(parents=True, exist_ok=True)
        self._project_dir.mkdir(parents=True, exist_ok=True)

    def _block_path(self, label: str, scope: str) -> Path:
        base = self._project_dir if scope == "project" else self._global_dir
        return base / f"{label}.json"

    def list_blocks(self, scope: str = "all") -> list[dict[str, str]]:
        """List available memory blocks. Project blocks take priority."""
        blocks = []
        seen_labels = set()

        # Project first so it takes priority over global with the same label
        for s in ["project", "global"]:
            if scope != "all" and scope != s:
                continue
            base = self._project_dir if s == "project" else self._global_dir
            if not base.is_dir():
                continue
            for f in sorted(base.glob("*.json")):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    label = data.get("label", f.stem)
                    if label not in seen_labels:
                        seen_labels.add(label)
                        blocks.append(
                            {
                                "label": label,
                                "description": data.get("description", ""),
                                "scope": data.get("scope", s),
                            }
                        )
                except (json.JSONDecodeError, KeyError):
                    continue
        return blocks

    def get_block(self, label: str, scope: str = "all") -> dict | None:
        """Get a memory block by label. Project has priority."""
        scopes_to_check = ["project", "global"] if scope == "all" else [scope]
        for s in scopes_to_check:
            path = self._block_path(label, s)
            if path.is_file():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    return data
                except (json.JSONDecodeError, KeyError):
                    continue
        return None

    def set_block(
        self,
        label: str,
        value: str,
        description: str = "",
        scope: str = "global",
    ) -> None:
        """Create or overwrite a memory block atomically."""
        if not BLOCK_NAME_RE.match(label):
            raise ValueError(
                f"Invalid block label: {label}. "
                "Must match [a-zA-Z0-9][a-zA-Z0-9_-]{{0,63}}"
            )

        path = self._block_path(label, scope)
        data = {
            "label": label,
            "description": description,
            "value": value,
            "scope": scope,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        content = json.dumps(data, ensure_ascii=False, indent=2)

        # Atomic write
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

    def delete_block(self, label: str, scope: str = "all") -> bool:
        """Delete a memory block."""
        deleted = False
        scopes_to_check = ["project", "global"] if scope == "all" else [scope]
        for s in scopes_to_check:
            path = self._block_path(label, s)
            if path.is_file():
                path.unlink()
                deleted = True
        return deleted

    def get_summary(self) -> str:
        """Get compact summary of all memory blocks for system prompt."""
        blocks = self.list_blocks()
        if not blocks:
            return ""
        lines = []
        for b in blocks:
            desc = b["description"] or "(no description)"
            lines.append(f"{b['label']} [{b['scope']}] — {desc}")
        return "\n".join(lines)
