"""ft-agent package."""

from ft_agent.agent import Agent
from ft_agent.core import Action, CallableNode, ExecResult, Flow, FlowError, FlowRunResult, Node, Payload

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
    "__version__",
]

__version__ = "0.1.0"
