"""Skills management: search, index and load SKILL.md files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

SKILL_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


class SkillManager:
    """Manages skill discovery and loading."""

    def __init__(self, global_skills_dir: Path, project_skills_dir: Path) -> None:
        self._global_dir = global_skills_dir
        self._project_dir = project_skills_dir
        self._index: dict[str, dict[str, Any]] = {}
        self._loaded: dict[str, dict[str, Any]] = {}
        self._load_times: dict[str, float] = {}
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Scan skill directories and build index."""
        self._index.clear()
        for base_dir, scope in [
            (self._global_dir, "global"),
            (self._project_dir, "project"),
        ]:
            if not base_dir.is_dir():
                continue
            for skill_dir in base_dir.iterdir():
                if not skill_dir.is_dir():
                    continue
                skill_md = skill_dir / "SKILL.md"
                if not skill_md.is_file():
                    continue
                meta = self._parse_frontmatter(skill_md)
                if meta is None:
                    continue
                name = meta.get("name", skill_dir.name)
                if not SKILL_NAME_RE.match(name):
                    continue
                # Project has priority over global
                if name in self._index and scope == "global":
                    continue
                self._index[name] = {
                    "name": name,
                    "description": meta.get("description", ""),
                    "path": skill_md,
                    "scope": scope,
                    "mtime": skill_md.stat().st_mtime,
                }

    def _parse_frontmatter(self, path: Path) -> dict | None:
        """Parse YAML frontmatter from SKILL.md."""
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            return None

        if not content.startswith("---"):
            return None

        parts = content.split("---", 2)
        if len(parts) < 3:
            return None

        try:
            return yaml.safe_load(parts[1]) or {}
        except yaml.YAMLError:
            return None

    def rebuild_index(self) -> None:
        """Public method to rebuild the skill index."""
        self._rebuild_index()

    def get_index_lines(self) -> list[str]:
        """Get compact index for system prompt."""
        lines = []
        for name, info in sorted(self._index.items()):
            desc = info.get("description", "")
            lines.append(f"{name} — {desc}")
        return lines

    def load_skill(self, name: str) -> dict | None:
        """Load full skill content by name."""
        if name not in self._index:
            return None

        info = self._index[name]
        skill_path = info["path"]

        # Check if already loaded and file hasn't changed
        if name in self._loaded:
            current_mtime = skill_path.stat().st_mtime
            if current_mtime == self._load_times.get(name):
                return self._loaded[name]

        # Load full content
        try:
            content = skill_path.read_text(encoding="utf-8")
        except Exception:
            return None

        # Extract body (after frontmatter)
        body = content
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                body = parts[2].strip()

        result = {
            "name": info["name"],
            "description": info["description"],
            "content": body,
            "path": str(skill_path),
            "scope": info["scope"],
        }

        self._loaded[name] = result
        self._load_times[name] = skill_path.stat().st_mtime
        return result

    def get_all_descriptions(self) -> list[dict[str, str]]:
        """Get all skill names and descriptions."""
        return [
            {"name": name, "description": info["description"]}
            for name, info in sorted(self._index.items())
        ]
