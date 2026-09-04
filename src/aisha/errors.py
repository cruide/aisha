# Author: Tischenko A. (https://github.com/cruide)
"""Exception hierarchy shared by all modules."""


class AishaError(Exception):
    """Base class for all agent errors."""


class ConfigurationError(AishaError):
    """Invalid or conflicting configuration."""


class ServerUnavailableError(AishaError):
    """llama-server is not reachable or not ready."""


class ProtocolError(AishaError):
    """Server returned malformed SSE/JSON or an unexpected HTTP status."""


class ContextOverflowError(AishaError):
    """Request does not fit into the model context."""


class ToolValidationError(AishaError):
    """Tool arguments failed schema validation."""


class ToolPermissionError(AishaError):
    """Operation is forbidden by the current permission mode."""


class ToolTimeoutError(AishaError):
    """Tool execution exceeded its time limit."""


class ToolCancelledError(AishaError):
    """User cancelled the operation."""
