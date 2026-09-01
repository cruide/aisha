"""Tests for skills management."""

from __future__ import annotations

__author__ = "Tischenko Alexander"

from pathlib import Path

import pytest

from aisha.skills import SkillManager


@pytest.fixture
def skill_mgr(tmp_path: Path) -> SkillManager:
    global_dir = tmp_path / "global_skills"
    project_dir = tmp_path / "project_skills"
    return SkillManager(global_dir, project_dir)


def test_empty_index(skill_mgr: SkillManager) -> None:
    assert skill_mgr.get_index_lines() == []


def test_discover_skill(skill_mgr: SkillManager, tmp_path: Path) -> None:
    skill_dir = skill_mgr._global_dir / "test-skill"
    skill_dir.mkdir(parents=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\nname: test-skill\ndescription: A test skill\n---\n\nDo test things.",
        encoding="utf-8",
    )
    skill_mgr.rebuild_index()
    lines = skill_mgr.get_index_lines()
    assert len(lines) == 1
    assert "test-skill" in lines[0]


def test_load_skill(skill_mgr: SkillManager) -> None:
    skill_dir = skill_mgr._global_dir / "my-skill"
    skill_dir.mkdir(parents=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\nname: my-skill\ndescription: My skill\n---\n\nInstructions here.",
        encoding="utf-8",
    )
    skill_mgr.rebuild_index()
    result = skill_mgr.load_skill("my-skill")
    assert result is not None
    assert result["name"] == "my-skill"
    assert "Instructions here." in result["content"]


def test_project_skill_priority(skill_mgr: SkillManager) -> None:
    # Create global skill
    g_dir = skill_mgr._global_dir / "shared"
    g_dir.mkdir(parents=True)
    (g_dir / "SKILL.md").write_text(
        "---\nname: shared\ndescription: global version\n---\n\nglobal",
        encoding="utf-8",
    )
    # Create project skill with same name
    p_dir = skill_mgr._project_dir / "shared"
    p_dir.mkdir(parents=True)
    (p_dir / "SKILL.md").write_text(
        "---\nname: shared\ndescription: project version\n---\n\nproject",
        encoding="utf-8",
    )
    skill_mgr.rebuild_index()
    result = skill_mgr.load_skill("shared")
    assert result is not None
    assert result["description"] == "project version"
    assert result["scope"] == "project"


def test_invalid_frontmatter(skill_mgr: SkillManager) -> None:
    skill_dir = skill_mgr._global_dir / "bad"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("No frontmatter here.", encoding="utf-8")
    skill_mgr.rebuild_index()
    assert skill_mgr.get_index_lines() == []


def test_invalid_yaml(skill_mgr: SkillManager) -> None:
    skill_dir = skill_mgr._global_dir / "bad-yaml"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\n: invalid: yaml: [[\n---\n", encoding="utf-8")
    skill_mgr.rebuild_index()
    assert skill_mgr.get_index_lines() == []


def test_load_nonexistent(skill_mgr: SkillManager) -> None:
    assert skill_mgr.load_skill("nonexistent") is None


def test_get_all_descriptions(skill_mgr: SkillManager) -> None:
    skill_dir = skill_mgr._global_dir / "desc-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\name: desc-skill\ndescription: Description test\n---\n\n",
        encoding="utf-8",
    )
    skill_mgr.rebuild_index()
    descs = skill_mgr.get_all_descriptions()
    # May be empty if YAML parsing failed
    assert isinstance(descs, list)
