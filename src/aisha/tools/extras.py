# Author: Tischenko A. (https://github.com/cruide)
"""Auxiliary tools: todowrite, ask_user, memory_*, skill."""

from __future__ import annotations

from typing import Any

from rich.markup import escape

from aisha.errors import ToolPermissionError, ToolValidationError
from aisha.skills import skill_body
from aisha.tools.base import Tool, ToolContext, ToolResult

TODO_STATUSES = ("pending", "in_progress", "done", "cancelled")


class TodoWriteTool(Tool):
    name = "todowrite"
    read_only = True
    description = (
        "Replace the entire current task list. REQUIRED JSON argument: todos — an array of "
        "objects; every object must contain text and status. Valid statuses: pending, "
        "in_progress, done, cancelled. Use for work with at least three steps, keep exactly "
        "one item in_progress while work remains, and send the complete list on every call."
    )
    parameters = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "REQUIRED. Complete replacement list; each object needs text "
                               "and status.",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Task description."},
                        "status": {"type": "string", "enum": list(TODO_STATUSES),
                                   "description": "pending, in_progress, done, or cancelled."},
                    },
                    "required": ["text", "status"],
                },
            }
        },
        "required": ["todos"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        items: list[dict[str, str]] = []
        for raw in args["todos"]:
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
    interactive_only = True
    description = (
        "Ask the user for a decision or missing information and wait for the answer. "
        "REQUIRED JSON argument: question — the exact question as a string. "
        "Provide options for a choice; allow_free_text defaults to true. Interactive mode only."
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "REQUIRED. Question shown to user."},
            "options": {"type": "array", "items": {"type": "string"},
                        "description": "Optional answer choices as strings."},
            "allow_free_text": {"type": "boolean", "description": "Allow a custom answer "
                                "in addition to options (default true)."},
        },
        "required": ["question"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.ask is None or not ctx.interactive:
            raise ToolPermissionError("ask_user is not available in non-interactive mode")
        options = [str(o) for o in args.get("options") or []]
        answer = await ctx.ask(args["question"], options, bool(args.get("allow_free_text", True)))
        return ToolResult.success({"answer": answer}, f"answer: {escape(answer[:60])}")


def _store(ctx: ToolContext):
    if ctx.memory is None:
        raise ToolPermissionError("Memory is disabled in the configuration")
    return ctx.memory


class MemoryListTool(Tool):
    name = "memory_list"
    read_only = True
    description = (
        "List memory labels, descriptions and scopes, not contents. No arguments. "
        "Use memory_get to read a block; project blocks shadow global blocks with the same label."
    )
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
    description = (
        "Read a persistent memory block by its exact label from the memory index. "
        "Returns value, description and scope; read before updating existing facts."
    )
    parameters = {"type": "object", "properties": {
        "label": {"type": "string", "description": "REQUIRED. Exact label from memory_list."}},
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
        "Create or fully overwrite durable memory. Use global for user preferences and "
        "project for this workspace. "
        "Do not store secrets; keep the description brief and the value focused."
    )
    parameters = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "description": "REQUIRED. Stable block identifier."},
            "description": {"type": "string", "description": "REQUIRED. Brief index summary."},
            "value": {"type": "string", "description": "REQUIRED. Complete block contents."},
            "scope": {"type": "string", "enum": ["global", "project"],
                      "description": "Storage scope (default global)."},
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
        "Replace exact text in an existing memory block. First memory_get the current value "
        "and copy old_text "
        "verbatim; use a unique fragment and expected_replacements when needed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "description": "REQUIRED. Exact memory label."},
            "old_text": {"type": "string", "description": "REQUIRED. Exact text copied from "
                         "memory_get value."},
            "new_text": {"type": "string", "description": "REQUIRED. Replacement text."},
            "expected_replacements": {"type": "integer", "minimum": 1,
                                      "description": "Exact occurrence count (default 1)."},
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
    description = (
        "Load task-specific instructions using an exact name from the skills index, "
        "not a file path. "
        "Load a relevant skill before working; unchanged skills are only returned once per session."
    )
    parameters = {"type": "object", "properties": {
        "name": {"type": "string", "description": "REQUIRED. Exact skill name from the "
                 "skills index, not a path."}},
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
