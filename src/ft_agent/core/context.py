"""Compatibility wrapper for agent_core.core.context."""

from agent_core.core.context import (
    AgentEvent,
    RunContext,
    get_current_context,
    reset_current_context,
    set_current_context,
)

__all__ = [
    "AgentEvent",
    "RunContext",
    "get_current_context",
    "reset_current_context",
    "set_current_context",
]

