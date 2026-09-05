# Author: Tischenko A. (https://github.com/cruide)
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
        "Update the current session's task list (full replacement). Required argument: "
        "items — array of objects like {text: string, status: pending|in_progress|done|cancelled}. "
        "Example: todowrite(items=[{text: 'write tests', status: 'pending'}])."
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
                raise ToolValidationError("each item must contain a non-empty text")
            status = str(raw.get("status", "pending"))
            if status not in TODO_STATUSES:
                raise ToolValidationError(f"invalid status {status!r}")
            items.append({"text": str(raw["text"]).strip(), "status": status})
        ctx.todos[:] = items
        if ctx.on_system_change:
            ctx.on_system_change()
        done = sum(1 for t in items if t["status"] == "done")
        return ToolResult.success({"items": items}, f"{done}/{len(items)} done")


class AskUserTool(Tool):
    name = "ask_user"
    read_only = True
    description = (
        "Ask the user a clarifying question and wait for an answer. Required argument: "
        "question — the question text. Optional: options (list of choices) and allow_free_text. "
        "Use when a task is ambiguous instead of guessing."
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
            raise ToolPermissionError("ask_user is not available in non-interactive mode")
        options = [str(o) for o in args.get("options") or []]
        answer = await ctx.ask(args["question"], options, bool(args.get("allow_free_text", True)))
        return ToolResult.success({"answer": answer}, f"answer: {answer[:60]}")


def _store(ctx: ToolContext):
    if ctx.memory is None:
        raise ToolPermissionError("Memory is disabled in the configuration")
    return ctx.memory


class MemoryListTool(Tool):
    name = "memory_list"
    read_only = True
    description = "List persistent memory blocks with descriptions. No arguments required."
    parameters = {"type": "object", "properties": {}}

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        store = _store(ctx)
        blocks = [{"label": b.label, "description": b.description, "scope": b.scope,
                   "updated_at": b.updated_at} for b in store.list()]
        return ToolResult.success({"blocks": blocks, "errors": list(store.errors)},
                                  f"{len(blocks)} blocks")


class MemoryGetTool(Tool):
    name = "memory_get"
    read_only = True
    silent = True
    description = "Read the contents of a memory block. Required argument: label — block name."
    parameters = {"type": "object", "properties": {"label": {"type": "string"}},
                  "required": ["label"]}

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        block = _store(ctx).get(args["label"])


        if block is None:
            return ToolResult.failure("NotFound", f"Memory block not found: {args['label']}")
        return ToolResult.success(
            {"label": block.label, "description": block.description, "value": block.value,
             "scope": block.scope, "updated_at": block.updated_at},
            f"{len(block.value)} chars",
        )


class MemorySetTool(Tool):
    name = "memory_set"
    description = (
        "Create or fully overwrite a memory block. Required arguments: label (name), "
        "description (brief purpose) and value (contents). scope: global (user preferences) "
        "or project (current project rules). Save only durable facts, do not store secrets."
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
        "Exact text replacement inside a memory block. Required arguments: label (block name), "
        "old_text (exact fragment to replace) and new_text (replacement text)."
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
                                  f"{block.label} updated")


class SkillTool(Tool):
    name = "skill"
    read_only = True
    description = "Load a skill's full text by name from the index. Required argument: name."
    parameters = {"type": "object", "properties": {"name": {"type": "string"}},
                  "required": ["name"]}

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        skill = ctx.skills.get(args["name"])
        if skill is None:
            return ToolResult.failure("NotFound", f"Skill not found: {args['name']}")
        mtime = skill.path.stat().st_mtime
        if ctx.loaded_skills.get(skill.name) == mtime:
            return ToolResult.success(
                {"name": skill.name, "already_loaded": True,
                 "note": "Skill is already loaded in this session and has not changed."},
                "already loaded",
            )
        ctx.loaded_skills[skill.name] = mtime
        body = skill_body(skill.path)
        return ToolResult.success(
            {"name": skill.name, "scope": skill.scope, "directory": str(skill.directory),
             "content": body},
            f"{skill.name}, {len(body)} chars",
        )
