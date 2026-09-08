# Author: Tischenko A. (https://github.com/cruide)
from aisha.skills import SkillIndex, skill_body


def test_scan_bom_skill(tmp_path):
    p = tmp_path / "p"
    (p / "bom").mkdir(parents=True)
    (p / "bom" / "SKILL.md").write_bytes(
        b"\xef\xbb\xbf---\nname: bom\ndescription: d\n---\nbody\n"
    )
    idx = SkillIndex(tmp_path / "g", p)
    idx.scan()
    assert idx.get("bom") is not None
    assert skill_body(idx.get("bom").path) == "body"
    assert len(idx.errors) == 0
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
