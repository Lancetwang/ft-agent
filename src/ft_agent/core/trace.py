"""Compatibility wrapper for agent_core.core.trace."""

from agent_core.core.trace import (
    DEFAULT_TRACE_CATEGORIES,
    TRACE_KEY,
    TraceEvent,
    TraceOptions,
    TraceRecorder,
    clear_trace_recorder,
    format_trace_event,
    get_trace_recorder,
    make_trace_options,
    reset_current_trace_recorder,
    set_current_trace_recorder,
)

__all__ = [
    "DEFAULT_TRACE_CATEGORIES",
    "TRACE_KEY",
    "TraceEvent",
    "TraceOptions",
    "TraceRecorder",
    "clear_trace_recorder",
    "format_trace_event",
    "get_trace_recorder",
    "make_trace_options",
    "reset_current_trace_recorder",
    "set_current_trace_recorder",
]
