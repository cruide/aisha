# Author: Tischenko A. (https://github.com/cruide)
import pytest

from aisha.errors import ToolValidationError
from aisha.memory import MemoryStore


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
