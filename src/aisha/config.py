"""Configuration: defaults <- global TOML <- project TOML <- env <- CLI, with validation."""

from __future__ import annotations

import copy
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from aisha.errors import ConfigurationError

PERMISSIONS = ("auto", "ask", "deny")
SHELLS = ("powershell", "cmd")

DEFAULTS: dict[str, dict[str, Any]] = {
    "server": {
        "base_url": "http://localhost:8088",
        "model": "Qwen3.5-9B-Q4_K_XL",
        "api_key": "",
        "connect_timeout": 5.0,
        "request_timeout": 600.0,
    },
    "llm": {
        "temperature": 0.7,
        "top_p": None,
        "top_k": None,
        "repeat_penalty": None,
        "frequency_penalty": None,
        "max_output_tokens": 32768,
        "context_window": 32768,
        "context_soft_limit": 0.85,
        "max_tool_iterations": 25,
        "tool_guide": False,
    },
    "tools": {
        "shell": True,
        "web_search": True,
        "permission": "ask",
        "shell_type": "powershell",
        "shell_timeout": 120,
        "max_output_chars": 65536,
        "allow_read_outside_workspace": False,
        "allow_write_outside_workspace": False,
    },
    "web": {
        "timeout": 20,
        "max_results": 8,
        "max_page_bytes": 2_097_152,
        "max_content_chars": 50_000,
        "allow_private_hosts": False,
    },
    "memory": {"enabled": True, "max_block_chars": 30_000},
    "ui": {
        "theme": "dark",
        "stream": True,
        "show_reasoning": False,
        "debug": False,
        "input_history": "~/.aisha/input_history.txt",
    },
}

ENV_VARS: dict[str, tuple[str, str, type]] = {
    "AISHA_SERVER_URL": ("server", "base_url", str),
    "AISHA_MODEL": ("server", "model", str),
    "AISHA_API_KEY": ("server", "api_key", str),
    "AISHA_PERMISSION": ("tools", "permission", str),
    "AISHA_SHELL": ("tools", "shell_type", str),
    "AISHA_CONTEXT_WINDOW": ("llm", "context_window", int),
    "AISHA_MAX_OUTPUT_TOKENS": ("llm", "max_output_tokens", int),
}


@dataclass(slots=True)
class ServerConfig:
    base_url: str
    model: str
    api_key: str
    connect_timeout: float
    request_timeout: float


@dataclass(slots=True)
class LLMConfig:
    temperature: float
    top_p: float | None
    top_k: int | None
    repeat_penalty: float | None
    frequency_penalty: float | None
    max_output_tokens: int
    context_window: int
    context_soft_limit: float
    max_tool_iterations: int
    tool_guide: bool


@dataclass(slots=True)
class ToolsConfig:
    shell: bool
    web_search: bool
    permission: str
    shell_type: str
    shell_timeout: int
    max_output_chars: int
    allow_read_outside_workspace: bool
    allow_write_outside_workspace: bool


@dataclass(slots=True)
class WebConfig:
    timeout: float
    max_results: int
    max_page_bytes: int
    max_content_chars: int
    allow_private_hosts: bool


@dataclass(slots=True)
class MemoryConfig:
    enabled: bool
    max_block_chars: int


@dataclass(slots=True)
class UIConfig:
    theme: str
    stream: bool
    show_reasoning: bool
    debug: bool
    input_history: str


@dataclass(slots=True)
class Config:
    server: ServerConfig
    llm: LLMConfig
    tools: ToolsConfig
    web: WebConfig
    memory: MemoryConfig
    ui: UIConfig
    workspace: Path
    read_only: bool = False
    sources: list[str] = field(default_factory=list)

    @property
    def home_dir(self) -> Path:
        return Path.home() / ".aisha"

    @property
    def project_dir(self) -> Path:
        return self.workspace / ".aisha"


def global_config_path() -> Path:
    return Path.home() / ".aisha" / "config.toml"


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(f"{path}: ошибка разбора TOML: {exc}") from exc
    except OSError as exc:
        raise ConfigurationError(f"{path}: не удалось прочитать файл: {exc}") from exc


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _check_structure(data: dict[str, Any], source: str) -> None:
    for section, values in data.items():
        if section not in DEFAULTS:
            raise ConfigurationError(f"{source}: неизвестная секция [{section}]")
        if not isinstance(values, dict):
            raise ConfigurationError(f"{source}: секция [{section}] должна быть таблицей")
        for key in values:
            if key not in DEFAULTS[section]:
                raise ConfigurationError(f"{source}: [{section}] неизвестный параметр '{key}'")


def _check_project_security(project: dict[str, Any], current: dict[str, Any], source: str) -> None:
    """A project-level aisha.toml must never weaken security silently."""
    tools = project.get("tools", {})
    if tools.get("permission") == "auto":
        raise ConfigurationError(
            f'{source}: [tools] permission = "auto" нельзя задавать в проектном конфиге '
            "(только глобально или через --permission auto)"
        )
    for key in ("allow_read_outside_workspace", "allow_write_outside_workspace"):
        if tools.get(key) is True:
            raise ConfigurationError(
                f"{source}: [tools] {key} = true разрешён только в глобальном конфиге"
            )
    if tools.get("shell") is True and not current["tools"]["shell"]:
        raise ConfigurationError(
            f"{source}: [tools] shell запрещён глобально и не может быть включён проектом"
        )


def _validate(data: dict[str, Any], source: str) -> None:
    def fail(section: str, key: str, msg: str) -> None:
        raise ConfigurationError(f"{source}: [{section}] {key}: {msg}")

    def positive(section: str, key: str) -> None:
        value = data[section][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            fail(section, key, f"ожидается положительное число, получено {value!r}")

    def number_in_range(section: str, key: str, lo: float, hi: float) -> None:
        value = data[section][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            fail(section, key, f"ожидается число от {lo} до {hi}, получено {value!r}")
        if not lo <= float(value) <= hi:
            fail(section, key, f"ожидается значение от {lo} до {hi}")

    srv, llm, tools = data["server"], data["llm"], data["tools"]
    url = urlparse(str(srv["base_url"]))
    if url.scheme not in ("http", "https") or not url.netloc:
        fail("server", "base_url", f"некорректный URL {srv['base_url']!r}")
    if not str(srv["model"]).strip():
        fail("server", "model", "имя модели обязательно")
    if srv["api_key"] is not None and not isinstance(srv["api_key"], str):
        fail("server", "api_key", "ожидается строка")
    for key in ("connect_timeout", "request_timeout"):
        positive("server", key)
    for key in ("max_output_tokens", "context_window", "max_tool_iterations"):
        positive("llm", key)
    number_in_range("llm", "temperature", 0.0, 2.0)
    number_in_range("llm", "context_soft_limit", 0.5, 0.95)
    if llm["top_p"] is not None:
        number_in_range("llm", "top_p", 0.0, 1.0)
    if llm["top_k"] is not None:
        if isinstance(llm["top_k"], bool) or not isinstance(llm["top_k"], int) \
                or llm["top_k"] <= 0:
            fail("llm", "top_k", f"ожидается положительное целое, получено {llm['top_k']!r}")
    if llm["repeat_penalty"] is not None:
        if isinstance(llm["repeat_penalty"], bool) \
                or not isinstance(llm["repeat_penalty"], (int, float)) \
                or llm["repeat_penalty"] <= 0:
            fail("llm", "repeat_penalty",
                 f"ожидается положительное число, получено {llm['repeat_penalty']!r}")
    if llm["frequency_penalty"] is not None:
        number_in_range("llm", "frequency_penalty", -2.0, 2.0)
    if llm["max_output_tokens"] > llm["context_window"]:
        fail("llm", "max_output_tokens", "не должен превышать context_window")
    if tools["permission"] not in PERMISSIONS:
        fail("tools", "permission", f"допустимо: {', '.join(PERMISSIONS)}")
    if tools["shell_type"] not in SHELLS:
        fail("tools", "shell_type", f"допустимо: {', '.join(SHELLS)}")
    for key in ("shell_timeout", "max_output_chars"):
        positive("tools", key)
    for key in ("timeout", "max_results", "max_page_bytes", "max_content_chars"):
        positive("web", key)
    positive("memory", "max_block_chars")
    bool_keys = (
        ("tools", "shell"), ("tools", "web_search"),
        ("tools", "allow_read_outside_workspace"), ("tools", "allow_write_outside_workspace"),
        ("web", "allow_private_hosts"), ("memory", "enabled"),
        ("ui", "stream"), ("ui", "show_reasoning"), ("ui", "debug"),
        ("llm", "tool_guide"),
    )
    for section, key in bool_keys:
        if not isinstance(data[section][key], bool):
            fail(section, key, "ожидается true/false")


def load_config(
    workspace: Path,
    *,
    cli: dict[str, dict[str, Any]] | None = None,
    read_only: bool = False,
    env: dict[str, str] | None = None,
) -> Config:
    """Build the effective configuration for `workspace`."""
    env = os.environ if env is None else env
    data = copy.deepcopy(DEFAULTS)
    sources: list[str] = []

    gpath = global_config_path()
    if gpath.is_file():
        gdata = _load_toml(gpath)
        _check_structure(gdata, str(gpath))
        _deep_merge(data, gdata)
        _validate(data, str(gpath))
        sources.append(str(gpath))

    ppath = workspace / "aisha.toml"
    if ppath.is_file():
        pdata = _load_toml(ppath)
        _check_structure(pdata, str(ppath))
        _check_project_security(pdata, data, str(ppath))
        _deep_merge(data, pdata)
        _validate(data, str(ppath))
        sources.append(str(ppath))

    for name, (section, key, cast) in ENV_VARS.items():
        raw = env.get(name)
        if raw is None or raw == "":
            continue
        try:
            data[section][key] = cast(raw)
        except ValueError as exc:
            raise ConfigurationError(f"{name}: некорректное значение {raw!r}") from exc
        sources.append(name)
    _validate(data, "переменные окружения")

    if cli:
        _deep_merge(data, cli)
        _validate(data, "аргументы командной строки")
        sources.append("CLI")

    sources = list(dict.fromkeys(sources))

    return Config(
        server=ServerConfig(**data["server"]),
        llm=LLMConfig(**data["llm"]),
        tools=ToolsConfig(**data["tools"]),
        web=WebConfig(**data["web"]),
        memory=MemoryConfig(**data["memory"]),
        ui=UIConfig(**data["ui"]),
        workspace=workspace,
        read_only=read_only,
        sources=sources,
    )
