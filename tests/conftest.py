from pathlib import Path

import pytest

from aisha.config import load_config
from aisha.skills import SkillIndex
from aisha.tools.base import ToolContext


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws.resolve()


@pytest.fixture
def ctx(workspace: Path, monkeypatch) -> ToolContext:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: workspace.parent / "home"))
    config = load_config(workspace, env={})
    skills = SkillIndex(
        workspace.parent / "home" / ".aisha" / "skills", workspace / ".aisha" / "skills"
    )
    return ToolContext(workspace=workspace, config=config, memory=None, skills=skills, todos=[])
