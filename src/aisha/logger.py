# Author: Tischenko A. (https://github.com/cruide)
"""Write logical LLM payloads, responses, and tool traces to a debug log."""

from __future__ import annotations

import json
import logging
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _short_id() -> str:
    """Generate a short 6-char hex session identifier."""
    return uuid.uuid4().hex[:6]


def _log_filename() -> str:
    """Return log filename: aisha_YYYY_MM_DD_<short_id>.log."""
    now = datetime.now(timezone.utc)
    return f"aisha_{now.strftime('%Y_%m_%d')}_{_short_id()}.log"


class DebugLogger:
    """Writes structured debug logs for LLM interactions to a file.

    Log file is created in ``<workspace>/.aisha/logs/`` with the naming pattern
    ``aisha_YYYY_MM_DD_<6-char-hex>.log``.  All methods are no-ops when
    the logger has not been started (i.e. debug mode is off).
    """

    def __init__(self) -> None:
        self._logger: logging.Logger | None = None
        self._path: Path | None = None

    @property
    def path(self) -> Path | None:
        """Return the log file path, or None if not started."""
        return self._path

    def start(self, workspace: Path) -> Path:
        """Create the log directory and open the log file.

        Returns the resolved log file path.
        """
        log_dir = workspace / ".aisha" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._path = log_dir / _log_filename()

        self._logger = logging.getLogger(f"aisha.debug.{id(self)}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        handler = logging.FileHandler(
            self._path, encoding="utf-8", delay=True,
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)

        self._log("=" * 72)
        self._log(f"AISHA DEBUG LOG — {datetime.now(timezone.utc).isoformat()}")
        self._log(f"Session: {self._path.stem}")
        self._log("=" * 72)
        return self._path

    def close(self) -> None:
        """Close every handler without allowing one failure to stop cleanup."""
        if self._logger is None:
            return
        logger = self._logger
        try:
            for handler in logger.handlers[:]:
                try:
                    handler.close()
                except Exception:
                    pass
                finally:
                    logger.removeHandler(handler)
        finally:
            self._logger = None

    def _log(self, text: str) -> None:
        if self._logger is not None:
            self._logger.debug(text)

    def _dump_json(self, label: str, obj: Any) -> None:
        """Pretty-print a JSON-serialisable object under a label."""
        self._log(f"\n--- {label} ---")
        try:
            self._log(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            self._log(repr(obj))

    # ------------------------------------------------------------------ LLM
    def log_request(
        self,
        payload: dict[str, Any],
        *,
        est_tokens: int,
        message_count: int,
    ) -> None:
        """Log an outgoing LLM chat request."""
        self._log(f"\n{'=' * 72}")
        self._log(f"→ LLM REQUEST — messages: {message_count}, ~{est_tokens} tokens")
        self._log(f"  model: {payload.get('model', '?')}")
        self._log(f"  temperature: {payload.get('temperature', '?')}")
        self._log(f"  max_tokens: {payload.get('max_tokens', '?')}")
        if payload.get("tools"):
            tool_names = [
                t.get("function", {}).get("name", "?") for t in payload["tools"]
            ]
            self._log(f"  tools: {', '.join(tool_names)}")
        self._dump_json("payload", payload)

    def log_response(
        self,
        response: Any,
        *,
        produced_chars: int,
    ) -> None:
        """Log an LLM response (ChatResponse or equivalent)."""
        self._log(f"\n{'─' * 72}")
        self._log("← LLM RESPONSE")
        self._log(f"  finish_reason: {getattr(response, 'finish_reason', '?')}")
        self._log(f"  interrupted: {getattr(response, 'interrupted', False)}")
        content = getattr(response, "content", "")
        reasoning = getattr(response, "reasoning", "")
        tool_calls = getattr(response, "tool_calls", [])
        usage = getattr(response, "usage", None)
        if content:
            self._log(f"  content ({len(content)} chars):")
            self._log(content)
        if reasoning:
            self._log(f"  reasoning ({len(reasoning)} chars):")
            self._log(reasoning)
        if tool_calls:
            self._log(f"  tool_calls ({len(tool_calls)}):")
            for tc in tool_calls:
                self._log(f"    [{tc.id}] {tc.name}({tc.arguments})")
        if usage:
            self._dump_json("usage", usage)
        self._log(f"  produced: ~{produced_chars} chars")

    # --------------------------------------------------------------- tools
    def log_tool_call(
        self,
        name: str,
        args: dict[str, Any],
        call_id: str,
    ) -> None:
        """Log a tool call before execution."""
        self._log(f"\n  ▸ TOOL CALL: {name} (id={call_id})")
        self._dump_json("    args", args)

    def log_tool_result(
        self,
        name: str,
        call_id: str,
        result_json: str,
        *,
        ok: bool,
    ) -> None:
        """Log a tool result after execution."""
        status = "OK" if ok else "FAIL"
        self._log(f"\n  ◂ TOOL RESULT: {name} (id={call_id}) [{status}]")
        try:
            obj = json.loads(result_json)
            self._log(json.dumps(obj, indent=2, ensure_ascii=False, default=str)[:4000])
        except (json.JSONDecodeError, TypeError):
            self._log(result_json[:4000])

    def log_exception(self, exc: Exception, *, tool_name: str) -> None:
        """Log a tool implementation failure with its full traceback."""
        self._log(f"\n  ! TOOL EXCEPTION: {tool_name}")
        self._log("".join(traceback.format_exception(exc)).rstrip())


# Module-level singleton — initialised in cli.py when --debug is active.
debug_logger = DebugLogger()
