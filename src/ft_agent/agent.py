from ft_agent.core.flow import Flow, FlowRunResult
from ft_agent.core.node import Context


class Agent:
    def __init__(self, flow: Flow) -> None:
        self.flow = flow

    def run(self, context: Context | None = None) -> FlowRunResult:
        return self.flow.run(context)
