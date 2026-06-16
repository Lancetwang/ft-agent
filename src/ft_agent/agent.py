from ft_agent.core.flow import Flow, FlowRunResult
from ft_agent.core.node import Payload


class Agent:
    def __init__(self, flow: Flow) -> None:
        self.flow = flow

    def run(self, payload: Payload = None) -> FlowRunResult:
        return self.flow.run(payload)
