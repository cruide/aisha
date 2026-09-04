# Author: Tischenko A. (https://github.com/cruide)
"""Skill discovery: <dir>/<name>/SKILL.md with YAML frontmatter (global + project)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from aisha.errors import ToolValidationError

NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    path: Path
    scope: str

    @property
    def directory(self) -> Path:
        return self.path.parent


def parse_skill_file(path: Path, scope: str) -> Skill:
    text = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    if not match:
        raise ToolValidationError(f"{path}: отсутствует YAML frontmatter")
    meta = yaml.safe_load(match.group(1)) or {}
    if not isinstance(meta, dict):
        raise ToolValidationError(f"{path}: frontmatter должен быть словарём")
    name = str(meta.get("name", "")).strip()
    description = str(meta.get("description", "")).strip()
    if not NAME_RE.match(name):
        raise ToolValidationError(f"{path}: некорректное поле name {name!r}")
    if not description:
        raise ToolValidationError(f"{path}: поле description обязательно")
    return Skill(name=name, description=description, path=path, scope=scope)


def skill_body(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    return (match.group(2) if match else text).strip()


class SkillIndex:
    def __init__(self, global_dir: Path, project_dir: Path) -> None:
        self.dirs = {"global": global_dir, "project": project_dir}
        self.skills: dict[str, Skill] = {}
        self.errors: list[str] = []

    def scan(self) -> None:
        self.skills.clear()
        self.errors.clear()
        for scope in ("global", "project"):  # project wins on conflict
            root = self.dirs[scope]
            if not root.is_dir():
                continue
            for path in sorted(root.glob("*/SKILL.md")):
                try:
                    skill = parse_skill_file(path, scope)
                except (ToolValidationError, OSError, yaml.YAMLError) as exc:
                    self.errors.append(str(exc))
                    continue
                self.skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def index_text(self) -> str:
        return "\n".join(f"- {s.name} — {s.description}" for s in self.skills.values())
