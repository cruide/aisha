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


def test_index_text_truncates(tmp_path):
    root = tmp_path / "skills"
    for i in range(10):
        d = root / f"s{i}"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\nname: s{i}\ndescription: desc{i}\n---\nbody\n", encoding="utf-8"
        )
    idx = SkillIndex(root, tmp_path / "proj", index_max_chars=40)
    idx.scan()
    text = idx.index_text()
    assert "use skill(name)" in text
    assert "s0" in text
    assert "s9" not in text
