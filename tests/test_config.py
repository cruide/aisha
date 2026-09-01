"""Tests for configuration loading and validation."""

from __future__ import annotations

__author__ = "Tischenko Alexander"

from pathlib import Path

import pytest

from aisha.config import Config, ConfigurationError, _validate_config, load_config


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


def test_default_config(workspace: Path) -> None:
    cfg = load_config(workspace)
    assert cfg.server.base_url == "http://localhost:8088"
    assert cfg.server.model == "Qwen3.5-9B-Q4_K_XL"
    assert cfg.tools.permission == "ask"
    assert cfg.tools.shell_type == "powershell"
    assert cfg.llm.context_window == 65536
    assert cfg.llm.max_output_tokens == 8192
    assert cfg.llm.context_soft_limit == 0.85


def test_global_toml(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = workspace / "home"
    home.mkdir()
    aisha_dir = home / ".aisha"
    aisha_dir.mkdir()
    config_toml = aisha_dir / "config.toml"
    config_toml.write_text(
        '[server]\nbase_url = "http://192.168.1.1:9090"\nmodel = "test-model"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    cfg = load_config(workspace)
    assert cfg.server.base_url == "http://192.168.1.1:9090"
    assert cfg.server.model == "test-model"


def test_project_toml(workspace: Path) -> None:
    config_toml = workspace / "aisha.toml"
    config_toml.write_text(
        '[server]\nmodel = "project-model"\n',
        encoding="utf-8",
    )
    cfg = load_config(workspace)
    assert cfg.server.model == "project-model"


def test_project_toml_cannot_set_auto(workspace: Path) -> None:
    config_toml = workspace / "aisha.toml"
    config_toml.write_text(
        '[tools]\npermission = "auto"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="permission=auto"):
        load_config(workspace)


def test_project_toml_cannot_enable_write_outside(workspace: Path) -> None:
    config_toml = workspace / "aisha.toml"
    config_toml.write_text(
        '[tools]\nallow_write_outside_workspace = true\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="запись за пределы"):
        load_config(workspace)


def test_project_toml_cannot_enable_read_outside(workspace: Path) -> None:
    config_toml = workspace / "aisha.toml"
    config_toml.write_text(
        '[tools]\nallow_read_outside_workspace = true\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="чтение за пределы"):
        load_config(workspace)


def test_cli_overrides(workspace: Path) -> None:
    cfg = load_config(workspace, {"model": "cli-model", "server_url": "http://1.2.3.4:5"})
    assert cfg.server.model == "cli-model"
    assert cfg.server.base_url == "http://1.2.3.4:5"


def test_env_overrides(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AISHA_MODEL", "env-model")
    monkeypatch.setenv("AISHA_SERVER_URL", "http://env:9999")
    cfg = load_config(workspace)
    assert cfg.server.model == "env-model"
    assert cfg.server.base_url == "http://env:9999"


def test_invalid_url(workspace: Path) -> None:
    cfg = Config()
    cfg.server.base_url = "ftp://invalid"
    with pytest.raises(ConfigurationError, match="URL"):
        _validate_config(cfg)


def test_invalid_permission(workspace: Path) -> None:
    cfg = Config()
    cfg.tools.permission = "invalid"
    with pytest.raises(ConfigurationError, match="permission"):
        _validate_config(cfg)


def test_invalid_shell_type(workspace: Path) -> None:
    cfg = Config()
    cfg.tools.shell_type = "bash"
    with pytest.raises(ConfigurationError, match="shell_type"):
        _validate_config(cfg)


def test_context_soft_limit_range(workspace: Path) -> None:
    cfg = Config()
    cfg.llm.context_soft_limit = 0.1
    with pytest.raises(ConfigurationError, match="context_soft_limit"):
        _validate_config(cfg)

    cfg.llm.context_soft_limit = 0.99
    with pytest.raises(ConfigurationError, match="context_soft_limit"):
        _validate_config(cfg)


def test_max_output_tokens_exceeds_context(workspace: Path) -> None:
    cfg = Config()
    cfg.llm.max_output_tokens = 70000
    cfg.llm.context_window = 65536
    with pytest.raises(ConfigurationError, match="max_output_tokens"):
        _validate_config(cfg)


def test_empty_model_name(workspace: Path) -> None:
    cfg = Config()
    cfg.server.model = ""
    with pytest.raises(ConfigurationError, match="Имя модели"):
        _validate_config(cfg)


def test_read_only_mode(workspace: Path) -> None:
    cfg = load_config(workspace, {"read_only": True})
    assert cfg.tools.read_only is True


def test_read_only_kept_with_permission_auto(workspace: Path) -> None:
    cfg = load_config(workspace, {"read_only": True, "permission": "auto"})
    assert cfg.tools.read_only is True
    assert cfg.tools.permission == "auto"


def test_invalid_toml(workspace: Path) -> None:
    config_toml = workspace / "aisha.toml"
    config_toml.write_text("[invalid\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="TOML"):
        load_config(workspace)
