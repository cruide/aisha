"""Extra tools: todowrite, ask_user, memory tools, skill tool."""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable

from aisha.tools.base import Tool, ToolResult, ToolValidationError


class TodoItem:
    def __init__(self, text: str, status: str = "pending") -> None:
        self.text = text
        self.status = status


class TodoWriteTool(Tool):
    """Manages session task list."""

    def __init__(self) -> None:
        self.items: list[TodoItem] = []

    @property
    def name(self) -> str:
        return "todowrite"

    @property
    def description(self) -> str:
        return "Manage session task list with status tracking."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "done", "cancelled"],
                            },
                        },
                        "required": ["text", "status"],
                    },
                    "description": "Updated task list",
                },
            },
            "required": ["items"],
        }

    def validate_args(self, args: dict) -> dict:
        valid_statuses = {"pending", "in_progress", "done", "cancelled"}
        for item in args.get("items", []):
            if item.get("status") not in valid_statuses:
                raise ToolValidationError(f"Invalid status: {item.get('status')}")
        return args

    def format_args(self, args: dict) -> str:
        items = args.get("items", [])
        return f"{len(items)} задач"

    async def execute(self, args: dict, context: dict) -> ToolResult:
        self.items = [TodoItem(i["text"], i["status"]) for i in args["items"]]
        return ToolResult.success(
            {"items": [{"text": i.text, "status": i.status} for i in self.items]}
        )

    def get_summary(self) -> str:
        if not self.items:
            return ""
        lines = []
        for i, item in enumerate(self.items, 1):
            status_icon = {
                "pending": "[ ]",
                "in_progress": "[~]",
                "done": "[x]",
                "cancelled": "[-]",
            }.get(item.status, "[?]")
            lines.append(f"{i}. {status_icon} {item.text}")
        return "\n".join(lines)


class AskUserTool(Tool):
    """Ask the user a question during agent loop."""

    @property
    def name(self) -> str:
        return "ask_user"

    @property
    def description(self) -> str:
        return "Ask the user a question. Pauses the agent loop."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Question to ask"},
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Answer options",
                },
                "allow_free_text": {
                    "type": "boolean",
                    "description": "Allow free-text input",
                    "default": True,
                },
            },
            "required": ["question"],
        }

    async def execute(self, args: dict, context: dict) -> ToolResult:
        ask_fn: Callable[[str, list[str] | None], Awaitable[str | None]] | None = (
            context.get("ask_fn")
        )
        if not ask_fn:
            return ToolResult.failure(
                "NotInteractiveError",
                "Интерактивный ввод недоступен в неинтерактивном режиме",
            )

        question = args["question"]
        options = args.get("options")
        try:
            answer = await ask_fn(question, options)
            if answer is None:
                return ToolResult.failure(
                    "ToolCancelledError", "Пользователь отменил ввод"
                )
            return ToolResult.success({"answer": answer})
        except Exception as e:
            return ToolResult.from_exception(e)


class MemoryListTool(Tool):
    """List memory blocks."""

    def __init__(self, memory_manager: Any) -> None:
        self._memory = memory_manager

    @property
    def name(self) -> str:
        return "memory_list"

    @property
    def description(self) -> str:
        return "List available memory blocks with their descriptions."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": ["all", "global", "project"],
                    "default": "all",
                },
            },
        }

    async def execute(self, args: dict, context: dict) -> ToolResult:
        scope = args.get("scope", "all")
        blocks = self._memory.list_blocks(scope)
        return ToolResult.success({"blocks": blocks})


class MemoryGetTool(Tool):
    """Get a specific memory block."""

    def __init__(self, memory_manager: Any) -> None:
        self._memory = memory_manager

    @property
    def name(self) -> str:
        return "memory_get"

    @property
    def description(self) -> str:
        return "Read the full content of a memory block."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "Memory block label"},
            },
            "required": ["label"],
        }

    async def execute(self, args: dict, context: dict) -> ToolResult:
        label = args["label"]
        block = self._memory.get_block(label)
        if block is None:
            return ToolResult.failure(
                "MemoryBlockNotFoundError",
                f"Блок памяти '{label}' не найден",
            )
        return ToolResult.success(block)


class MemorySetTool(Tool):
    """Create or overwrite a memory block."""

    def __init__(self, memory_manager: Any) -> None:
        self._memory = memory_manager

    @property
    def name(self) -> str:
        return "memory_set"

    @property
    def description(self) -> str:
        return "Create or completely overwrite a memory block."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "Block label"},
                "value": {"type": "string", "description": "Block content"},
                "description": {
                    "type": "string",
                    "description": "Block description",
                    "default": "",
                },
                "scope": {
                    "type": "string",
                    "enum": ["global", "project"],
                    "default": "global",
                },
            },
            "required": ["label", "value"],
        }

    def validate_args(self, args: dict) -> dict:
        label = args.get("label", "")
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$", label):
            raise ToolValidationError(
                f"Invalid memory block label: {label}. "
                "Must match [a-zA-Z0-9][a-zA-Z0-9_-]{{0,63}}"
            )
        return args

    async def execute(self, args: dict, context: dict) -> ToolResult:
        read_only = context.get("read_only", False)
        if read_only:
            return ToolResult.failure(
                "ToolPermissionError",
                "Режим read-only: запись памяти запрещена",
            )

        label = args["label"]
        value = args["value"]
        max_block_chars = context.get("max_block_chars", 30000)
        if len(value) > max_block_chars:
            return ToolResult.failure(
                "ValueTooLong",
                f"Блок превышает лимит ({len(value)}/{max_block_chars} символов). "
                "Сожмите содержимое перед записью.",
            )

        try:
            self._memory.set_block(
                label=label,
                value=value,
                description=args.get("description", ""),
                scope=args.get("scope", "global"),
            )
            return ToolResult.success({"label": label, "scope": args.get("scope", "global")})
        except Exception as e:
            return ToolResult.from_exception(e)


class MemoryReplaceTool(Tool):
    """Replace text inside a memory block."""

    def __init__(self, memory_manager: Any) -> None:
        self._memory = memory_manager

    @property
    def name(self) -> str:
        return "memory_replace"

    @property
    def description(self) -> str:
        return "Replace exact text inside an existing memory block."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "Block label"},
                "old_text": {"type": "string", "description": "Exact text to replace"},
                "new_text": {"type": "string", "description": "Replacement text"},
            },
            "required": ["label", "old_text", "new_text"],
        }

    def validate_args(self, args: dict) -> dict:
        if not args.get("old_text"):
            raise ToolValidationError("old_text must not be empty")
        return args

    async def execute(self, args: dict, context: dict) -> ToolResult:
        read_only = context.get("read_only", False)
        if read_only:
            return ToolResult.failure(
                "ToolPermissionError",
                "Режим read-only: изменение памяти запрещено",
            )

        label = args["label"]
        block = self._memory.get_block(label)
        if block is None:
            return ToolResult.failure(
                "MemoryBlockNotFoundError",
                f"Блок памяти '{label}' не найден",
            )

        old_text = args["old_text"]
        new_text = args["new_text"]
        value = block["value"]

        if old_text not in value:
            return ToolResult.failure(
                "TextNotFoundError",
                "old_text не найден в блоке памяти",
            )

        new_value = value.replace(old_text, new_text, 1)
        try:
            self._memory.set_block(
                label=label,
                value=new_value,
                description=block.get("description", ""),
                scope=block.get("scope", "global"),
            )
            return ToolResult.success({"label": label})
        except Exception as e:
            return ToolResult.from_exception(e)


class SkillTool(Tool):
    """Load a skill by name."""

    def __init__(self, skill_manager: Any) -> None:
        self._skills = skill_manager

    @property
    def name(self) -> str:
        return "skill"

    @property
    def description(self) -> str:
        return "Load a skill by name and return its instructions."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name"},
            },
            "required": ["name"],
        }

    async def execute(self, args: dict, context: dict) -> ToolResult:
        name = args["name"]
        skill = self._skills.load_skill(name)
        if skill is None:
            return ToolResult.failure(
                "SkillNotFoundError",
                f"Скилл '{name}' не найден",
            )
        return ToolResult.success(skill)
