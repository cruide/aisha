"""Tests for memory management."""

from __future__ import annotations

__author__ = "Tischenko Alexander"

from pathlib import Path

import pytest

from aisha.memory import MemoryManager


@pytest.fixture
def mem(tmp_path: Path) -> MemoryManager:
    return MemoryManager(tmp_path / "global", tmp_path / "project")


def test_set_and_get_block(mem: MemoryManager) -> None:
    mem.set_block("test", "hello world", description="test block")
    block = mem.get_block("test")
    assert block is not None
    assert block["value"] == "hello world"
    assert block["label"] == "test"
    assert block["description"] == "test block"


def test_project_priority(mem: MemoryManager) -> None:
    mem.set_block("shared", "global value", scope="global")
    mem.set_block("shared", "project value", scope="project")
    block = mem.get_block("shared")
    assert block["value"] == "project value"


def test_list_blocks(mem: MemoryManager) -> None:
    mem.set_block("a", "value a", scope="global")
    mem.set_block("b", "value b", scope="project")
    blocks = mem.list_blocks()
    labels = [b["label"] for b in blocks]
    assert "a" in labels
    assert "b" in labels


def test_list_blocks_global_only(mem: MemoryManager) -> None:
    mem.set_block("a", "global", scope="global")
    mem.set_block("b", "project", scope="project")
    blocks = mem.list_blocks("global")
    assert len(blocks) == 1
    assert blocks[0]["label"] == "a"


def test_delete_block(mem: MemoryManager) -> None:
    mem.set_block("delme", "value")
    assert mem.get_block("delme") is not None
    mem.delete_block("delme")
    assert mem.get_block("delme") is None


def test_invalid_label(mem: MemoryManager) -> None:
    with pytest.raises(ValueError):
        mem.set_block("", "value")
    with pytest.raises(ValueError):
        mem.set_block("has spaces", "value")
    with pytest.raises(ValueError):
        mem.set_block("-starts-dash", "value")


def test_atomic_write(mem: MemoryManager) -> None:
    mem.set_block("atomic", "v1")
    mem.set_block("atomic", "v2")
    block = mem.get_block("atomic")
    assert block["value"] == "v2"
    # No tmp files
    global_dir = mem._global_dir
    assert not list(global_dir.glob("*.tmp"))


def test_get_summary(mem: MemoryManager) -> None:
    mem.set_block("style", "code style", description="Code preferences")
    summary = mem.get_summary()
    assert "style" in summary
    assert "Code preferences" in summary
