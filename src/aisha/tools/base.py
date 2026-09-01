"""Base tool class and tool registry for aisha agent."""

from __future__ import annotations

__author__ = "Tischenko Alexander"

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class ToolValidationError(Exception):
    """Raised when tool arguments are invalid."""


class ToolPermissionError(Exception):
    """Raised when tool execution is not permitted."""


class ToolTimeoutError(Exception):
    """Raised when a tool execution times out."""


class ToolCancelledError(Exception):
    """Raised when a user cancels a tool operation."""


@dataclass
class ToolResult:
    """Standardized tool execution result."""

    ok: bool
    data: Any = None
    error: dict | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "data": self.data,
            "error": self.error,
            "meta": self.meta,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    @staticmethod
    def success(data: Any, meta: dict | None = None) -> ToolResult:
        return ToolResult(ok=True, data=data, meta=meta or {})

    @staticmethod
    def failure(error_type: str, message: str, meta: dict | None = None) -> ToolResult:
        return ToolResult(
            ok=False,
            error={"type": error_type, "message": message},
            meta=meta or {},
        )

    @staticmethod
    def from_exception(exc: Exception, meta: dict | None = None) -> ToolResult:
        return ToolResult(
            ok=False,
            error={"type": type(exc).__name__, "message": str(exc)},
            meta=meta or {},
        )


class Tool(ABC):
    """Base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in tool_calls."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Description shown to the model."""

    @property
    @abstractmethod
    def parameters_schema(self) -> dict:
        """JSON Schema for tool parameters."""

    def to_openai_tool(self) -> dict:
        """Convert to OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def validate_args(self, args: dict) -> dict:
        """Validate and normalize arguments. Raises ToolValidationError on failure."""
        return args

    def format_args(self, args: dict) -> str:
        """Return a short human-readable summary of the arguments for the UI."""
        for key in ("query", "url", "pattern", "command", "question", "path", "label", "name"):
            if key in args and args[key] is not None:
                return str(args[key])
        return ""

    @abstractmethod
    async def execute(self, args: dict, context: dict) -> ToolResult:
        """Execute the tool with given arguments and context.

        Context contains:
            workspace: Path - current workspace
            config: Config - current config
            permission_mode: str - auto/ask/deny
            read_only: bool
            ask_fn: callable - for user confirmation
        """

    def __repr__(self) -> str:
        return f"Tool({self.name})"


class ToolRegistry:
    """Registry for all available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def openai_tools(self) -> list[dict]:
        return [t.to_openai_tool() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
