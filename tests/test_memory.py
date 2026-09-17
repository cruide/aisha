# Author: Tischenko A. (https://github.com/cruide)
import json

import pytest

from aisha.errors import ToolValidationError
from aisha.memory import MEMORY_FILENAME, MemoryStore


def test_project_overrides_global_and_replace(tmp_path):
    store = MemoryStore(tmp_path / "g", tmp_path / "p", max_block_chars=100)
    store.set("style", "d", "global value", "global")
    store.set("style", "d", "project value", "project")
    assert store.get("style").value == "project value"
    assert [b.scope for b in store.list()] == ["project"]
    store.replace("style", "project", "new")
    assert store.get("style").value == "new value"
    with pytest.raises(ToolValidationError):
        store.set("bad name!", "d", "v")
    with pytest.raises(ToolValidationError):
        store.set("big", "d", "x" * 101)


def test_stored_as_markdown_with_metadata(tmp_path):
    store = MemoryStore(tmp_path / "g", tmp_path / "p", max_block_chars=1000)
    store.set("user_profile", "User info", "Name: Sanya\nRole: author", "global")
    text = (tmp_path / "g" / MEMORY_FILENAME).read_text(encoding="utf-8")
    assert text.startswith("# Memory\n")
    assert "## user_profile" in text
    assert "<!-- description: User info -->" in text
    assert "<!-- updated_at: " in text
    assert "Name: Sanya" in text

    # Re-reading a fresh store yields the same block.
    reread = MemoryStore(tmp_path / "g", tmp_path / "p", max_block_chars=1000)
    block = reread.get("user_profile")
    assert block.value == "Name: Sanya\nRole: author"
    assert block.description == "User info"
    assert block.updated_at


def test_value_heading_lines_are_escaped(tmp_path):
    store = MemoryStore(tmp_path / "g", tmp_path / "p", max_block_chars=1000)
    store.set("notes", "d", "## not a block\nnormal line", "global")
    assert store.get("notes").value == "## not a block\nnormal line"
    text = (tmp_path / "g" / MEMORY_FILENAME).read_text(encoding="utf-8")
    assert "\\## not a block" in text
    assert store.list() == [store.get("notes")]


def test_legacy_json_imported_and_removed(tmp_path):
    global_dir = tmp_path / "g"
    global_dir.mkdir()
    legacy = global_dir / "user_profile.json"
    legacy.write_text(json.dumps({
        "label": "user_profile", "description": "old", "value": "legacy value",
        "scope": "global", "updated_at": "2026-01-01T00:00:00Z",
    }), encoding="utf-8")

    store = MemoryStore(global_dir, tmp_path / "p", max_block_chars=1000)
    block = store.get("user_profile")
    assert block.value == "legacy value"
    assert block.description == "old"
    assert not legacy.exists()
    assert (global_dir / MEMORY_FILENAME).is_file()


def test_legacy_json_does_not_override_existing_block(tmp_path):
    global_dir = tmp_path / "g"
    store = MemoryStore(global_dir, tmp_path / "p", max_block_chars=1000)
    store.set("style", "new", "from markdown", "global")
    (global_dir / "style.json").write_text(json.dumps({
        "label": "style", "description": "old", "value": "from json",
    }), encoding="utf-8")

    reread = MemoryStore(global_dir, tmp_path / "p", max_block_chars=1000)
    assert reread.get("style").value == "from markdown"
    assert not (global_dir / "style.json").exists()


def test_index_text_truncates(tmp_path):
    store = MemoryStore(tmp_path / "g", tmp_path / "p", max_block_chars=1000,
                        index_max_chars=50)
    for i in range(10):
        store.set(f"block{i}", "description text here", f"value{i}")
    text = store.index_text()
    assert "use memory_list" in text
    assert "block0 (global)" in text
    assert "value0" not in text
    assert "value9" not in text
