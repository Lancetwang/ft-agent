"""ft-agent package."""

from ft_agent.agent import Agent
from ft_agent.core import (
    Action,
    AgentEvent,
    CallableNode,
    ExecResult,
    Flow,
    FlowError,
    FlowRunResult,
    Node,
    Payload,
    RunContext,
    TraceEvent,
    TraceOptions,
    TraceRecorder,
    format_trace_event,
    get_current_context,
    make_trace_options,
    reset_current_context,
    set_current_context,
)

__all__ = [
    "Action",
    "Agent",
    "AgentEvent",
    "CallableNode",
    "ExecResult",
    "Flow",
    "FlowError",
    "FlowRunResult",
    "Node",
    "Payload",
    "RunContext",
    "TraceEvent",
    "TraceOptions",
    "TraceRecorder",
    "__version__",
    "format_trace_event",
    "get_current_context",
    "make_trace_options",
    "reset_current_context",
    "set_current_context",
]

__version__ = "0.1.0"
