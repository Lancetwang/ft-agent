from collections.abc import Callable, Mapping, Sequence
from typing import Any

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
        chat_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.messages = messages
        self.output_key = output_key
        self.action = action
        self.chat_kwargs = dict(chat_kwargs or {})

    def exec(self, payload: Payload) -> ExecResult:
        state = dict(payload or {})
        chat_kwargs = dict(self.chat_kwargs)
        chat_kwargs.update(state.get("chat_kwargs", {}))
        content = self.llm.chat(self.messages(state), **chat_kwargs)
        state[self.output_key] = content
        return self.action, state
