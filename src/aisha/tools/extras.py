"""Auxiliary tools: todowrite, ask_user, memory_*, skill."""

from __future__ import annotations

from typing import Any

from aisha.errors import ToolPermissionError, ToolValidationError
from aisha.skills import skill_body
from aisha.tools.base import Tool, ToolContext, ToolResult

TODO_STATUSES = ("pending", "in_progress", "done", "cancelled")


class TodoWriteTool(Tool):
    name = "todowrite"
    read_only = True
    description = (
        "Обновить список задач текущей сессии (полная замена списка). Обязательный аргумент: "
        "items — массив объектов вида {text: строка, status: pending|in_progress|done|cancelled}. "
        "Пример: todowrite(items=[{text: 'написать тест', status: 'pending'}])."
    )
    parameters = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "status": {"type": "string", "enum": list(TODO_STATUSES)},
                    },
                    "required": ["text", "status"],
                },
            }
        },
        "required": ["items"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        items: list[dict[str, str]] = []
        for raw in args["items"]:
            if not isinstance(raw, dict) or not str(raw.get("text", "")).strip():
                raise ToolValidationError("каждый элемент должен содержать непустой text")
            status = str(raw.get("status", "pending"))
            if status not in TODO_STATUSES:
                raise ToolValidationError(f"недопустимый status {status!r}")
            items.append({"text": str(raw["text"]).strip(), "status": status})
        ctx.todos[:] = items
        if ctx.on_system_change:
            ctx.on_system_change()
        done = sum(1 for t in items if t["status"] == "done")
        return ToolResult.success({"items": items}, f"{done}/{len(items)} выполнено")


class AskUserTool(Tool):
    name = "ask_user"
    read_only = True
    description = (
        "Задать пользователю уточняющий вопрос и дождаться ответа. Обязательный аргумент: "
        "question — текст вопроса. Необязательные: options (список вариантов) и allow_free_text. "
        "Используй, когда задача неоднозначна, вместо того чтобы гадать."
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "options": {"type": "array", "items": {"type": "string"}},
            "allow_free_text": {"type": "boolean"},
        },
        "required": ["question"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.ask is None or not ctx.interactive:
            raise ToolPermissionError("ask_user недоступен в неинтерактивном режиме")
        options = [str(o) for o in args.get("options") or []]
        answer = await ctx.ask(args["question"], options, bool(args.get("allow_free_text", True)))
        return ToolResult.success({"answer": answer}, f"ответ: {answer[:60]}")


def _store(ctx: ToolContext):
    if ctx.memory is None:
        raise ToolPermissionError("Память отключена в конфигурации")
    return ctx.memory


class MemoryListTool(Tool):
    name = "memory_list"
    read_only = True
    description = "Список блоков постоянной памяти с описаниями. Аргументы не требуются."
    parameters = {"type": "object", "properties": {}}

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        store = _store(ctx)
        blocks = [{"label": b.label, "description": b.description, "scope": b.scope,
                   "updated_at": b.updated_at} for b in store.list()]
        return ToolResult.success({"blocks": blocks, "errors": list(store.errors)},
                                  f"{len(blocks)} блоков")


class MemoryGetTool(Tool):
    name = "memory_get"
    read_only = True
    description = "Прочитать содержимое блока памяти. Обязательный аргумент: label — имя блока."
    parameters = {"type": "object", "properties": {"label": {"type": "string"}},
                  "required": ["label"]}

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        block = _store(ctx).get(args["label"])


        if block is None:
            return ToolResult.failure("NotFound", f"Блок памяти не найден: {args['label']}")
        return ToolResult.success(
            {"label": block.label, "description": block.description, "value": block.value,
             "scope": block.scope, "updated_at": block.updated_at},
            f"{len(block.value)} символов",
        )


class MemorySetTool(Tool):
    name = "memory_set"
    description = (
        "Создать или полностью перезаписать блок памяти. Обязательные аргументы: label (имя), "
        "description (краткое назначение) и value (содержимое). scope: global (предпочтения "
        "пользователя) или project (правила текущего проекта). Сохраняй только устойчивые факты, "
        "не сохраняй секреты."
    )
    parameters = {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "description": {"type": "string"},
            "value": {"type": "string"},
            "scope": {"type": "string", "enum": ["global", "project"]},
        },
        "required": ["label", "description", "value"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        block = _store(ctx).set(args["label"], args["description"], args["value"],
                                args.get("scope", "global"))
        if ctx.on_system_change:
            ctx.on_system_change()
        return ToolResult.success({"label": block.label, "scope": block.scope,
                                   "chars": len(block.value)}, f"{block.label} ({block.scope})")


class MemoryReplaceTool(Tool):
    name = "memory_replace"
    description = (
        "Точная замена текста внутри блока памяти. Обязательные аргументы: label (имя блока), "
        "old_text (точный заменяемый фрагмент) и new_text (новый текст)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
            "expected_replacements": {"type": "integer"},
        },
        "required": ["label", "old_text", "new_text"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        block = _store(ctx).replace(args["label"], args["old_text"], args["new_text"],
                                    int(args.get("expected_replacements", 1)))
        if ctx.on_system_change:
            ctx.on_system_change()
        return ToolResult.success({"label": block.label, "chars": len(block.value)},
                                  f"{block.label} обновлён")


class SkillTool(Tool):
    name = "skill"
    read_only = True
    description = "Загрузить полный текст скилла по имени из индекса. Обязательный аргумент: name."
    parameters = {"type": "object", "properties": {"name": {"type": "string"}},
                  "required": ["name"]}

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        skill = ctx.skills.get(args["name"])
        if skill is None:
            return ToolResult.failure("NotFound", f"Скилл не найден: {args['name']}")
        mtime = skill.path.stat().st_mtime
        if ctx.loaded_skills.get(skill.name) == mtime:
            return ToolResult.success(
                {"name": skill.name, "already_loaded": True,
                 "note": "Скилл уже загружен в этой сессии и не изменялся."},
                "уже загружен",
            )
        ctx.loaded_skills[skill.name] = mtime
        body = skill_body(skill.path)
        return ToolResult.success(
            {"name": skill.name, "scope": skill.scope, "directory": str(skill.directory),
             "content": body},
            f"{skill.name}, {len(body)} символов",
        )
