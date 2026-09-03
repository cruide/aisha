"""Tool ABC, registry, result envelope, argument validation and confirmation protocol."""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aisha.errors import (
    AishaError,
    ToolCancelledError,
    ToolPermissionError,
    ToolValidationError,
)

if TYPE_CHECKING:
    from aisha.config import Config
    from aisha.memory import MemoryStore
    from aisha.skills import SkillIndex


@dataclass(slots=True)
class ConfirmRequest:
    title: str
    details: list[tuple[str, str]]
    reason: str
    key: str  # "allow for session" key


ConfirmFn = Callable[[ConfirmRequest], Awaitable[str | None]]
AskFn = Callable[[str, list[str], bool], Awaitable[str]]


@dataclass
class ToolContext:
    workspace: Path
    config: Config
    memory: MemoryStore | None
    skills: SkillIndex
    todos: list[dict[str, str]]
    confirm: ConfirmFn | None = None
    ask: AskFn | None = None
    interactive: bool = True
    session_allowed: set[str] = field(default_factory=set)
    loaded_skills: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    ok: bool
    data: Any = None
    error: dict[str, str] | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    summary: str = ""  # one-line human summary for the UI (not sent to the model)

    @classmethod
    def success(cls, data: Any, summary: str = "", **meta: Any) -> ToolResult:
        return cls(ok=True, data=data, summary=summary, meta=dict(meta))

    @classmethod
    def failure(cls, error_type: str, message: str, **meta: Any) -> ToolResult:
        return cls(ok=False, error={"type": error_type, "message": message},
                   summary=message, meta=dict(meta))

    def to_json(self) -> str:
        return json.dumps(
            {"ok": self.ok, "data": self.data, "error": self.error, "meta": self.meta},
            ensure_ascii=False,
        )


async def require_confirmation(ctx: ToolContext, request: ConfirmRequest) -> None:
    """Ask the user; 'a' remembers approval for the session, 'y' once, anything else cancels."""
    if request.key in ctx.session_allowed:
        return
    if ctx.confirm is None:
        raise ToolPermissionError(
            f"{request.title}: требуется подтверждение, но интерактивный режим недоступен"
        )
    answer = await ctx.confirm(request)
    if answer == "a":
        ctx.session_allowed.add(request.key)
        return
    if answer == "y":
        return
    raise ToolCancelledError(f"{request.title}: отклонено пользователем")


_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "integer": (int,), "number": (int, float),
    "boolean": (bool,), "array": (list,), "object": (dict,),
}


def validate_args(schema: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Validate against a minimal JSON-schema subset; unknown keys are dropped."""
    if not isinstance(args, dict):
        raise ToolValidationError("аргументы должны быть JSON-объектом")
    props: dict[str, Any] = schema.get("properties", {})
    missing = [k for k in schema.get("required", []) if args.get(k) is None]
    if missing:
        raise ToolValidationError(f"отсутствуют обязательные аргументы: {', '.join(missing)}")
    clean: dict[str, Any] = {}
    for key, value in args.items():
        if key not in props or value is None:
            continue
        spec = props[key]
        expected = spec.get("type")
        if expected in _TYPES:
            ok = isinstance(value, _TYPES[expected])
            if expected in ("integer", "number") and isinstance(value, bool):
                ok = False
            # Lenient coercion for small models that stringify scalars.
            if not ok and isinstance(value, str):
                stripped = value.strip()
                if expected == "integer" and stripped.lstrip("-").isdigit():
                    value, ok = int(stripped), True
                elif expected == "number":
                    try:
                        value, ok = float(stripped), True
                    except ValueError:
                        pass
                elif expected == "boolean" and stripped.lower() in ("true", "false"):
                    value, ok = stripped.lower() == "true", True
            if not ok and expected == "string" and isinstance(value, (int, float)):
                value, ok = str(value), True
            if not ok:
                raise ToolValidationError(
                    f"аргумент '{key}': ожидается {expected}, получено {type(value).__name__}"
                )
        if "enum" in spec and value not in spec["enum"]:
            raise ToolValidationError(
                f"аргумент '{key}': допустимые значения {', '.join(map(str, spec['enum']))}"
            )
        clean[key] = value
    return clean


class Tool(ABC):
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    read_only: bool = False  # True => allowed in --read-only mode

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult: ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Инструмент уже зарегистрирован: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __iter__(self):
        return iter(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self, *, read_only: bool = False) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values() if not read_only or t.read_only]

    async def execute(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        started = time.perf_counter()
        tool = self.get(name)
        if tool is None:
            result = ToolResult.failure("UnknownTool", f"Неизвестный инструмент: {name}")
        else:
            try:
                if ctx.config.read_only and not tool.read_only:
                    raise ToolPermissionError(f"Режим read-only: инструмент {name} недоступен")
                result = await tool.run(validate_args(tool.parameters, args), ctx)
            except asyncio.CancelledError:
                raise
            except AishaError as exc:
                result = ToolResult.failure(type(exc).__name__, str(exc))
            except Exception as exc:  # tool bugs must not kill the session
                result = ToolResult.failure(type(exc).__name__, f"{exc}")
        result.meta.setdefault("duration_ms", int((time.perf_counter() - started) * 1000))
        return result
    