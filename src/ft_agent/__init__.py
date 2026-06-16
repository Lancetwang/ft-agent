"""ft-agent package."""

from ft_agent.agent import Agent
from ft_agent.core import CallableNode, Context, Flow, FlowError, FlowRunResult, Node, NodeResult

__all__ = [
    "Agent",
    "CallableNode",
    "Context",
    "Flow",
    "FlowError",
    "FlowRunResult",
    "Node",
    "NodeResult",
    "__version__",
]

__version__ = "0.1.0"
