from typing import Any

from ft_agent.core import ExecResult, Node, Payload
from ft_agent.tools.executor import ToolExecutor


class ToolCallNode(Node):
    def __init__(
        self,
        *,
        executor: ToolExecutor | None = None,
        assistant_key: str = "assistant_message",
        messages_key: str = "history",
        results_key: str = "tool_results",
        next_action: str = "chat",
    ) -> None:
        super().__init__()
        self.executor = executor or ToolExecutor()
        self.assistant_key = assistant_key
        self.messages_key = messages_key
        self.results_key = results_key
        self.next_action = next_action

    def exec(self, payload: Payload) -> ExecResult:
        state: dict[str, Any] = dict(payload or {})
        assistant_message = state.get(self.assistant_key, {})
        tool_calls = self.executor.parse_tool_calls(assistant_message)
        results = self.executor.execute_all(tool_calls)

        state[self.results_key] = results
        messages = state.setdefault(self.messages_key, [])
        for result in results:
            messages.append(result.to_message())

        return self.next_action, state
