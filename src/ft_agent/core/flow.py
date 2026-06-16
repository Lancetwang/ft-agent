from dataclasses import dataclass
from typing import Any

from ft_agent.core.node import Action, Node, Payload


class FlowError(RuntimeError):
    pass


@dataclass(frozen=True)
class FlowRunResult:
    action: Action | None
    payload: Payload
    path: list[str]


class Flow:
    def __init__(self, start: Node | None = None) -> None:
        self.start = start

    def run(self, payload: Any = None, *, max_steps: int = 100) -> FlowRunResult:
        current = self.start
        last_action: Action | None = None
        path: list[str] = []

        for _ in range(max_steps):
            if current is None:
                return FlowRunResult(action=last_action, payload=payload, path=path)

            path.append(current.__class__.__name__)
            last_action, payload = current._exec(payload)
            current = current.successors.get(last_action)

        raise FlowError(f"Flow exceeded max_steps={max_steps}.")
