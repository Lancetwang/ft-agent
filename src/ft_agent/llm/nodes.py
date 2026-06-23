from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agent_core import ExecResult, Node, Payload
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


class ToolAwareLLMNode(Node):
    def __init__(
        self,
        *,
        llm: DeepSeekLLM,
        messages: Callable[[Payload], Sequence[Message]],
        tools: Callable[[Payload], Sequence[Mapping[str, Any]]] | Sequence[Mapping[str, Any]],
        assistant_key: str = "assistant_message",
        output_key: str = "answer",
        messages_key: str = "history",
        tool_action: str = "tool_call",
        done_action: str = "done",
        chat_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.messages = messages
        self.tools = tools
        self.assistant_key = assistant_key
        self.output_key = output_key
        self.messages_key = messages_key
        self.tool_action = tool_action
        self.done_action = done_action
        self.chat_kwargs = dict(chat_kwargs or {})

    def exec(self, payload: Payload) -> ExecResult:
        state = dict(payload or {})
        chat_kwargs = dict(self.chat_kwargs)
        chat_kwargs.update(state.get("chat_kwargs", {}))
        assistant_message = self.llm.chat_message(
            self.messages(state),
            tools=self._tools(state),
            **chat_kwargs,
        )
        state[self.assistant_key] = assistant_message
        state.setdefault(self.messages_key, []).append(assistant_message)

        if assistant_message.get("tool_calls"):
            return self.tool_action, state

        state[self.output_key] = assistant_message.get("content", "")
        return self.done_action, state

    def _tools(self, payload: Payload) -> Sequence[Mapping[str, Any]]:
        if callable(self.tools):
            return self.tools(payload)
        return self.tools
