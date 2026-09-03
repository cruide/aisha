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
        