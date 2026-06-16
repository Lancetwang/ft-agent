from collections.abc import Callable, Sequence

from ft_agent.core import ExecResult, Node, Payload
from ft_agent.llm.deepseek import DeepSeekLLM, Message


class LLMNode(Node):
    def __init__(
        self,
        *,
        llm: DeepSeekLLM,
        messages: Callable[[Payload], Sequence[Message]],
        output_key: str = "answer",
        action: str = "default",
    ) -> None:
        super().__init__()
        self.llm = llm
        self.messages = messages
        self.output_key = output_key
        self.action = action

    def exec(self, payload: Payload) -> ExecResult:
        state = dict(payload or {})
        content = self.llm.chat(self.messages(state), temperature=0)
        state[self.output_key] = content
        return self.action, state
