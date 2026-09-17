"""Tests for tool argument validation and failure diagnostics."""

import pytest

from aisha.errors import ToolValidationError
from aisha.tools.base import Tool, ToolContext, ToolResult, validate_args


class SchemaTool(Tool):
    """Test tool for schema enrichment."""

    name = "schema_tool"
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["path"],
    }

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Return the validated arguments."""
        return ToolResult.success(args)


def test_validate_args_enforces_numeric_bounds() -> None:
    schema = {
        "properties": {
            "count": {"type": "integer", "minimum": 1, "maximum": 10},
        }
    }
    assert validate_args(schema, {"count": "10"}) == {"count": 10}
    with pytest.raises(ToolValidationError, match="at least 1"):
        validate_args(schema, {"count": 0})
    with pytest.raises(ToolValidationError, match="at most 10"):
        validate_args(schema, {"count": 11})


def test_validate_args_rejects_oversized_integer_string() -> None:
    schema = {"properties": {"count": {"type": "integer"}}}
    with pytest.raises(ToolValidationError, match="invalid integer"):
        validate_args(schema, {"count": "9" * 5000})


def test_missing_required_arguments_explains_retry_shape() -> None:
    schema = {
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    with pytest.raises(
        ToolValidationError, match=r'Minimum valid arguments: \{"path":"<string>"\}'
    ):
        validate_args(schema, {}, "read_file")


def test_tool_schema_marks_required_properties_explicitly() -> None:
    schema = SchemaTool().schema()["function"]["parameters"]
    assert schema["properties"]["path"]["description"] == "REQUIRED."
    assert "description" not in schema["properties"]["limit"]
    assert schema["additionalProperties"] is False
    assert "description" not in SchemaTool.parameters["properties"]["path"]
