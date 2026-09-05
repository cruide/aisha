# Author: Tischenko A. (https://github.com/cruide)
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


def test_sampling_defaults_none(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    ws = tmp_path / "ws"
    ws.mkdir()
    cfg = load_config(ws, env={})
    assert cfg.llm.top_p is None
    assert cfg.llm.top_k is None
    assert cfg.llm.repeat_penalty is None
    assert cfg.llm.frequency_penalty is None


def test_sampling_params_set(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "aisha.toml").write_text(
        "[llm]\ntop_p = 0.9\ntop_k = 40\nrepeat_penalty = 1.1\nfrequency_penalty = 0.5\n"
    )
    cfg = load_config(ws, env={})
    assert cfg.llm.top_p == 0.9
    assert cfg.llm.top_k == 40
    assert cfg.llm.repeat_penalty == 1.1
    assert cfg.llm.frequency_penalty == 0.5


def test_api_key_priority(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    ws = tmp_path / "ws"
    ws.mkdir()
    assert load_config(ws, env={}).server.api_key == ""
    assert load_config(ws, env={"AISHA_API_KEY": "secret"}).server.api_key == "secret"
    (ws / "aisha.toml").write_text('[server]\napi_key = "project-key"\n')
    cfg = load_config(ws, env={"AISHA_API_KEY": "secret"})
    assert cfg.server.api_key == "secret"


@pytest.mark.parametrize("key,value", [
    ("top_p", "1.5"),
    ("top_p", '"abc"'),
    ("top_k", "0"),
    ("top_k", '"abc"'),
    ("repeat_penalty", "0"),
    ("frequency_penalty", "3.0"),
])
def test_sampling_validation(tmp_path: Path, monkeypatch, key, value):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "aisha.toml").write_text(f"[llm]\n{key} = {value}\n")
    with pytest.raises(ConfigurationError, match=key):
        load_config(ws, env={})


def test_skip_health_default_false(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    cfg = load_config(tmp_path / "ws", env={})
    assert cfg.server.skip_health is False


def test_skip_health_from_project_toml(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "aisha.toml").write_text("[server]\nskip_health = true\n")
    cfg = load_config(ws, env={})
    assert cfg.server.skip_health is True


def test_skip_health_from_global_toml(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    (home / ".aisha").mkdir(parents=True)
    (home / ".aisha" / "config.toml").write_text("[server]\nskip_health = true\n")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    cfg = load_config(tmp_path / "ws", env={})
    assert cfg.server.skip_health is True


def test_skip_health_from_env(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    cfg = load_config(tmp_path / "ws", env={"AISHA_SKIP_HEALTH": "true"})
    assert cfg.server.skip_health is True
    cfg = load_config(tmp_path / "ws", env={"AISHA_SKIP_HEALTH": "false"})
    assert cfg.server.skip_health is False


def test_skip_health_from_cli(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    cfg = load_config(tmp_path / "ws", env={}, cli={"server": {"skip_health": True}})
    assert cfg.server.skip_health is True


def test_skip_health_must_be_bool(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "aisha.toml").write_text('[server]\nskip_health = "yes"\n')
    with pytest.raises(ConfigurationError, match="skip_health"):
        load_config(ws, env={})

