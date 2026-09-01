"""Configuration loading, merging and validation for aisha agent."""

from __future__ import annotations

__author__ = "Tischenko Alexander"

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""


@dataclass
class ServerConfig:
    base_url: str = "http://localhost:8088"
    model: str = "Qwen3.5-9B-Q4_K_XL"
    connect_timeout: int = 5
    request_timeout: int = 600


@dataclass
class LlmConfig:
    temperature: float = 0.2
    max_output_tokens: int = 8192
    context_window: int = 65536
    context_soft_limit: float = 0.85
    max_tool_iterations: int = 25


@dataclass
class ToolsConfig:
    shell: bool = True
    web_search: bool = True
    permission: str = "ask"
    shell_type: str = "powershell"
    shell_timeout: int = 120
    max_output_chars: int = 65536
    allow_read_outside_workspace: bool = False
    allow_write_outside_workspace: bool = False
    read_only: bool = False


@dataclass
class WebConfig:
    timeout: int = 20
    max_results: int = 8
    max_page_bytes: int = 2097152
    max_content_chars: int = 50000
    allow_private_hosts: bool = False


@dataclass
class MemoryConfig:
    enabled: bool = True
    max_block_chars: int = 30000


@dataclass
class UiConfig:
    theme: str = "dark"
    stream: bool = True
    show_reasoning: bool = False
    input_history: str = "~/.aisha/input_history.txt"


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    web: WebConfig = field(default_factory=WebConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    ui: UiConfig = field(default_factory=UiConfig)


def _toml_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _load_toml_file(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path, "rb") as f:
        try:
            return tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ConfigurationError(f"Ошибка TOML в файле {path}: {e}") from e


def _apply_section(target: dict, source: dict, section: str, filename: str) -> None:
    if section not in source:
        return
    if not isinstance(source[section], dict):
        raise ConfigurationError(
            f"Секция [{section}] в файле {filename} должна быть словарём"
        )
    if section not in target:
        target[section] = {}
    target[section].update(source[section])


def _validate_config(cfg: Config, filename: str = "<defaults>") -> None:
    # Validate server URL
    parsed = urlparse(cfg.server.base_url)
    if parsed.scheme not in ("http", "https"):
        raise ConfigurationError(
            f"Некорректный URL сервера в {filename}: {cfg.server.base_url}"
        )

    # Validate permission
    if cfg.tools.permission not in ("auto", "ask", "deny"):
        raise ConfigurationError(
            f"Недопустимое значение permission в {filename}: {cfg.tools.permission}"
        )

    # Validate shell type
    if cfg.tools.shell_type not in ("powershell", "cmd"):
        raise ConfigurationError(
            f"Недопустимое значение shell_type в {filename}: {cfg.tools.shell_type}"
        )

    # Validate timeouts and limits
    if cfg.server.connect_timeout <= 0:
        raise ConfigurationError(
            f"connect_timeout должен быть положительным в {filename}"
        )
    if cfg.server.request_timeout <= 0:
        raise ConfigurationError(
            f"request_timeout должен быть положительным в {filename}"
        )
    if cfg.tools.shell_timeout <= 0:
        raise ConfigurationError(
            f"shell_timeout должен быть положительным в {filename}"
        )
    if cfg.tools.max_output_chars <= 0:
        raise ConfigurationError(
            f"max_output_chars должен быть положительным в {filename}"
        )
    if cfg.web.timeout <= 0:
        raise ConfigurationError(f"web timeout должен быть положительным в {filename}")
    if cfg.web.max_results <= 0:
        raise ConfigurationError(
            f"web max_results должен быть положительным в {filename}"
        )
    if cfg.web.max_page_bytes <= 0:
        raise ConfigurationError(
            f"web max_page_bytes должен быть положительным в {filename}"
        )
    if cfg.web.max_content_chars <= 0:
        raise ConfigurationError(
            f"web max_content_chars должен быть положительным в {filename}"
        )
    if cfg.memory.max_block_chars <= 0:
        raise ConfigurationError(
            f"memory max_block_chars должен быть положительным в {filename}"
        )

    # Validate context_soft_limit range
    if not (0.5 <= cfg.llm.context_soft_limit <= 0.95):
        raise ConfigurationError(
            f"context_soft_limit должен быть от 0.5 до 0.95 в {filename}"
        )

    # Validate max_output_tokens < context_window
    if cfg.llm.max_output_tokens >= cfg.llm.context_window:
        raise ConfigurationError(
            f"max_output_tokens ({cfg.llm.max_output_tokens}) должен быть меньше "
            f"context_window ({cfg.llm.context_window}) в {filename}"
        )

    # Validate model name
    if not cfg.server.model or not cfg.server.model.strip():
        raise ConfigurationError(f"Имя модели обязательно в {filename}")


def _build_config(
    global_cfg: dict,
    project_cfg: dict,
    env: dict[str, str],
    cli_overrides: dict[str, str],
) -> Config:
    """Merge configuration from all sources with proper priority."""
    cfg = Config()

    # Apply global config
    merged: dict = {}
    _apply_section(merged, global_cfg, "server", "global")
    _apply_section(merged, global_cfg, "llm", "global")
    _apply_section(merged, global_cfg, "tools", "global")
    _apply_section(merged, global_cfg, "web", "global")
    _apply_section(merged, global_cfg, "memory", "global")
    _apply_section(merged, global_cfg, "ui", "global")

    # Apply project config (with security restrictions)
    project_merged: dict = {}
    _apply_section(project_merged, project_cfg, "server", "aisha.toml")
    _apply_section(project_merged, project_cfg, "llm", "aisha.toml")
    _apply_section(project_merged, project_cfg, "tools", "aisha.toml")
    _apply_section(project_merged, project_cfg, "web", "aisha.toml")
    _apply_section(project_merged, project_cfg, "memory", "aisha.toml")
    _apply_section(project_merged, project_cfg, "ui", "aisha.toml")

    # Security: project config cannot weaken safety on its own
    if "permission" in project_merged.get("tools", {}):
        if project_merged["tools"]["permission"] == "auto":
            raise ConfigurationError(
                "Проектный конфиг не может установить permission=auto. "
                "Используйте глобальную конфигурацию или CLI-аргумент."
            )
    if "allow_write_outside_workspace" in project_merged.get("tools", {}):
        if project_merged["tools"]["allow_write_outside_workspace"]:
            raise ConfigurationError(
                "Проектный конфиг не может включить запись за пределы workspace."
            )
    if "allow_read_outside_workspace" in project_merged.get("tools", {}):
        if project_merged["tools"]["allow_read_outside_workspace"]:
            raise ConfigurationError(
                "Проектный конфиг не может включить чтение за пределы workspace."
            )

    merged.update(
        {
            section: {**merged.get(section, {}), **project_merged.get(section, {})}
            for section in ["server", "llm", "tools", "web", "memory", "ui"]
            if section in project_merged
        }
    )

    # Apply to dataclass
    if "server" in merged:
        for k, v in merged["server"].items():
            if hasattr(cfg.server, k):
                setattr(cfg.server, k, v)
    if "llm" in merged:
        for k, v in merged["llm"].items():
            if hasattr(cfg.llm, k):
                setattr(cfg.llm, k, v)
    if "tools" in merged:
        for k, v in merged["tools"].items():
            if hasattr(cfg.tools, k):
                setattr(cfg.tools, k, v)
    if "web" in merged:
        for k, v in merged["web"].items():
            if hasattr(cfg.web, k):
                setattr(cfg.web, k, v)
    if "memory" in merged:
        for k, v in merged["memory"].items():
            if hasattr(cfg.memory, k):
                setattr(cfg.memory, k, v)
    if "ui" in merged:
        for k, v in merged["ui"].items():
            if hasattr(cfg.ui, k):
                setattr(cfg.ui, k, v)

    # Apply environment variables
    env_map = {
        "AISHA_SERVER_URL": ("server", "base_url"),
        "AISHA_MODEL": ("server", "model"),
        "AISHA_PERMISSION": ("tools", "permission"),
        "AISHA_SHELL": ("tools", "shell_type"),
        "AISHA_CONTEXT_WINDOW": ("llm", "context_window"),
        "AISHA_MAX_OUTPUT_TOKENS": ("llm", "max_output_tokens"),
    }
    for env_key, (section, field_name) in env_map.items():
        val = env.get(env_key)
        if val is not None:
            section_obj = getattr(cfg, section)
            current = getattr(section_obj, field_name)
            if isinstance(current, int):
                try:
                    setattr(section_obj, field_name, int(val))
                except ValueError:
                    raise ConfigurationError(
                        f"Переменная {env_key} должна быть числом: {val}"
                    )
            else:
                setattr(section_obj, field_name, val)

    # Apply CLI overrides
    cli_map = {
        "server_url": ("server", "base_url"),
        "model": ("server", "model"),
        "permission": ("tools", "permission"),
        "shell": ("tools", "shell_type"),
        "context_window": ("llm", "context_window"),
        "max_output_tokens": ("llm", "max_output_tokens"),
    }
    for cli_key, (section, field_name) in cli_map.items():
        val = cli_overrides.get(cli_key)
        if val is not None:
            section_obj = getattr(cfg, section)
            current = getattr(section_obj, field_name)
            if isinstance(current, int):
                try:
                    setattr(section_obj, field_name, int(val))
                except ValueError:
                    raise ConfigurationError(
                        f"CLI-аргумент {cli_key} должен быть числом: {val}"
                    )
            elif isinstance(current, float):
                try:
                    setattr(section_obj, field_name, float(val))
                except ValueError:
                    raise ConfigurationError(
                        f"CLI-аргумент {cli_key} должен быть числом: {val}"
                    )
            else:
                setattr(section_obj, field_name, val)

    # CLI read-only flag (independent of permission mode)
    if cli_overrides.get("read_only"):
        cfg.tools.read_only = True

    # CLI permission override (highest priority)
    if cli_overrides.get("permission"):
        cfg.tools.permission = cli_overrides["permission"]

    return cfg


def load_config(
    workspace: Path,
    cli_overrides: dict[str, str] | None = None,
) -> Config:
    """Load and merge configuration from all sources.

    Priority (highest to lowest):
    1. CLI arguments
    2. Environment variables
    3. Project aisha.toml
    4. Global ~/.aisha/config.toml
    5. Built-in defaults
    """
    cli_overrides = cli_overrides or {}

    # Global config
    global_dir = Path.home() / ".aisha"
    global_path = global_dir / "config.toml"
    global_cfg = _load_toml_file(global_path)

    # Project config
    project_path = workspace / "aisha.toml"
    project_cfg = _load_toml_file(project_path)

    # Environment variables
    env = dict(os.environ)

    cfg = _build_config(global_cfg, project_cfg, env, cli_overrides)

    # Validate
    _validate_config(cfg, str(global_path))
    if project_cfg:
        _validate_config(cfg, str(project_path))

    return cfg
