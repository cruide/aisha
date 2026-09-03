from aisha.skills import SkillIndex, skill_body


def test_scan_priority_and_errors(tmp_path):
    g, p = tmp_path / "g", tmp_path / "p"
    (g / "rev").mkdir(parents=True)
    (g / "rev" / "SKILL.md").write_text("---\nname: rev\ndescription: global\n---\nG body\n")
    (p / "rev").mkdir(parents=True)
    (p / "rev" / "SKILL.md").write_text("---\nname: rev\ndescription: project\n---\nP body\n")
    (p / "bad").mkdir()
    (p / "bad" / "SKILL.md").write_text("no frontmatter")
    idx = SkillIndex(g, p)
    idx.scan()
    assert idx.get("rev").scope == "project"
    assert skill_body(idx.get("rev").path) == "P body"
    assert len(idx.errors) == 1
    