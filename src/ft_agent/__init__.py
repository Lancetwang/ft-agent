"""ft-agent package."""

from ft_agent.agent import Agent
from ft_agent.core import (
    Action,
    CallableNode,
    ExecResult,
    Flow,
    FlowError,
    FlowRunResult,
    Node,
    Payload,
    TraceEvent,
    TraceOptions,
    TraceRecorder,
    format_trace_event,
    make_trace_options,
)

__all__ = [
    "Action",
    "Agent",
    "CallableNode",
    "ExecResult",
    "Flow",
    "FlowError",
    "FlowRunResult",
    "Node",
    "Payload",
    "TraceEvent",
    "TraceOptions",
    "TraceRecorder",
    "__version__",
    "format_trace_event",
    "make_trace_options",
]

__version__ = "0.1.0"
