from pathlib import Path

import pytest

from aisha.config import load_config
from aisha.errors import ConfigurationError


def test_defaults_and_priority(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    (home / ".aisha").mkdir(parents=True)
    (home / ".aisha" / "config.toml").write_text(
        '[llm]\ntemperature = 0.7\n[tools]\nshell_timeout = 30\n'
    )
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "aisha.toml").write_text("[tools]\nshell_timeout = 60\n")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    cfg = load_config(ws, env={"AISHA_MODEL": "x"}, cli={"llm": {"temperature": 0.1}})
    assert cfg.tools.shell_timeout == 60  # project > global
    assert cfg.server.model == "x"  # env
    assert cfg.llm.temperature == 0.1  # cli > everything
    assert cfg.tools.permission == "ask"


def test_project_cannot_enable_auto(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "aisha.toml").write_text('[tools]\npermission = "auto"\n')
    with pytest.raises(ConfigurationError, match="permission"):
        load_config(ws, env={})


def test_validation_errors(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    ws = tmp_path / "ws"
    ws.mkdir()
    with pytest.raises(ConfigurationError, match="base_url"):
        load_config(ws, env={"AISHA_SERVER_URL": "not-a-url"})
    with pytest.raises(ConfigurationError, match="max_output_tokens"):
        load_config(ws, env={"AISHA_MAX_OUTPUT_TOKENS": "999999"})


@pytest.mark.parametrize("key,value", [
    ("temperature", "true"),
    ("temperature", '"abc"'),
    ("context_soft_limit", "true"),
    ("context_soft_limit", '"abc"'),
    ("context_soft_limit", "1.5"),
])
def test_strict_float_validation(tmp_path: Path, monkeypatch, key, value):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "aisha.toml").write_text(f"[llm]\n{key} = {value}\n")
    with pytest.raises(ConfigurationError, match=key):
        load_config(ws, env={})


def test_sources_are_unique(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    ws = tmp_path / "ws"
    ws.mkdir()
    cfg = load_config(ws, env={"AISHA_MODEL": "x", "AISHA_PERMISSION": "auto"},
                      cli={"llm": {"temperature": 0.1}})
    assert len(cfg.sources) == len(set(cfg.sources))


def test_tool_guide_flag(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    ws = tmp_path / "ws"
    ws.mkdir()
    assert load_config(ws, env={}).llm.tool_guide is False
    (ws / "aisha.toml").write_text("[llm]\ntool_guide = true\n")
    assert load_config(ws, env={}).llm.tool_guide is True


def test_tool_guide_must_be_bool(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "aisha.toml").write_text('[llm]\ntool_guide = "yes"\n')
    with pytest.raises(ConfigurationError, match="tool_guide"):
        load_config(ws, env={})

